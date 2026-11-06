"""Shared helpers for the fine-tuning sweep and the PGD attack.

Model building (victim + recovered weights), re-initialization of the layers to
be fine-tuned, freezing, data transforms, evaluation and one training epoch.
"""

import copy

import torch
import torch.nn as nn
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collect_conv_linear_layers(model):
    """Ordered {name: layer} of Conv2d/Linear layers + the name order.

    This order (named_modules) is the order CPA extracts layers in, so "the
    first K layers" below means the first K layers the attack recovered.
    """
    layers = {
        name: layer
        for name, layer in model.named_modules()
        if isinstance(layer, (nn.Conv2d, nn.Linear))
    }
    return layers, list(layers.keys())


def build_transforms(mean, std):
    """Train (augmented) and test transforms.

    The channel statistics are load-bearing AND per-architecture: ResNet-18 and
    VGG-11-BN were trained with the CIFAR-10 stats, the small CNN with
    Normalize(0.5). archs.Arch carries the right pair per victim.
    """
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, test_transform


def evaluate_fp32(model, test_loader):
    """Test accuracy (%) of an fp32 model; restores its train/eval mode."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if was_training:
        model.train()
    return 100 * correct / total


def evaluate_fp16(model, test_loader):
    """Test accuracy (%) of a half-precision copy -- the precision the victim
    accuracies (92.59 / 92.78 / 84.36) were measured in."""
    half_copy = copy.deepcopy(model).half()
    half_copy.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, predicted = half_copy(inputs.half()).max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def load_recovered_model(arch):
    """The victim with every conv/linear WEIGHT replaced by its recovered value.

    Starts from the victim checkpoint, so BatchNorm parameters/buffers (never
    attacked) and biases (never extracted) are the victim's, then overwrites
    every weight tensor with the CPA-recovered one (models/recovered_{arch}.pt,
    written by build_recovered.py from the extraction pipeline's per-filter
    outputs). Returned in fp32; the recovered values are fp16 and upcast
    losslessly.

    Only `.weight` comes from the extraction, so a layer left frozen keeps the
    victim's bias -- see reinitialize_layers for the other half of that rule.
    """
    model = arch.builder().to(device)
    model.load_state_dict(torch.load(arch.checkpoint))
    recovered = torch.load(arch.recovered)
    layers, layer_order = collect_conv_linear_layers(model)
    if set(recovered) != set(layer_order):
        raise ValueError(f"{arch.recovered} holds {sorted(recovered)}, "
                         f"model has {layer_order}")
    for name in layer_order:
        layer = layers[name]
        with torch.no_grad():
            layer.weight.data.copy_(recovered[name].reshape(layer.weight.shape)
                                    .to(layer.weight.dtype).to(layer.weight.device))
    print(f"Loaded recovered weights for {len(layer_order)} layers from {arch.recovered}")
    return model


def reinitialize_layers(model, trainable_names, init_layer):
    """Discard the to-be-fine-tuned layers' recovered values: weight AND bias are
    re-drawn with the architecture's own constructor init (`init_layer`), so a
    re-initialized layer starts from the same distribution as the from-scratch
    surrogate."""
    layers, _ = collect_conv_linear_layers(model)
    for name in trainable_names:
        if name not in layers:
            raise KeyError(f"'{name}' is not a Conv2d/Linear layer of this model")
        layer = layers[name]
        init_layer(layer)
        print(f"Re-initialized {name} ({tuple(layer.weight.shape)})")


def freeze_all_but(model, trainable_names):
    """Freeze every parameter, then re-enable only the trainable layers'.

    BatchNorm affine parameters end up frozen here; bnPrefix re-enables the ones
    paired with a trainable layer. Running statistics still update, because the
    model is put in train() mode for each epoch.
    """
    layers, _ = collect_conv_linear_layers(model)
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = []
    for name in trainable_names:
        for param in layers[name].parameters():
            param.requires_grad = True
            trainable_params.append(param)

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_trainable:,} of {n_total:,} params "
          f"({100 * n_trainable / n_total:.2f}%) across {trainable_names}")
    return trainable_params


def snapshot_frozen(model, names):
    """Clone the named layers' weights so the freeze can be proven afterwards."""
    layers, _ = collect_conv_linear_layers(model)
    return {name: layers[name].weight.detach().clone() for name in names}


def assert_frozen_unchanged(model, snapshot):
    """Fail loudly if a supposedly-frozen layer moved during training."""
    layers, _ = collect_conv_linear_layers(model)
    for name, before in snapshot.items():
        after = layers[name].weight.detach()
        if not torch.equal(before, after):
            drift = (after - before).abs().max().item()
            raise AssertionError(
                f"Frozen layer {name} changed during training (max drift {drift:.3e})"
            )
    print(f"Freeze audit passed: {', '.join(snapshot)} are bit-identical to their "
          f"pre-training values")


def train_one_epoch(model, train_loader, criterion, optimizer):
    """One pass over the train split. Returns (mean loss, train accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / total, 100 * correct / total
