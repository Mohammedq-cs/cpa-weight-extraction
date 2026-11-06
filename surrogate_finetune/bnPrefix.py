"""BatchNorm handling for the "extracted prefix" threat model.

BatchNorm is known exactly as far as the conv prefix is. For a cell with
num_recovered = K, the BN paired with each conv at index < K keeps the victim's
affine params and stays frozen; the BN paired with each conv at index >= K is
re-initialized (scale 1, shift 0), has its running stats reset, and is trained
alongside that conv. The given information is then a prefix in BN exactly as it
is in conv weights, so the victim's BN cannot do work the extracted weights are
supposed to be doing.

Running statistics of the PREFIX BN are left to recalibrate under
model.train(): the attacker holds his own weights and his own data and can
compute equivalent stats himself. Freezing them instead would model a strictly
stronger attacker and inflate the transfer result. Every cell records
bn_prefix_stats='recalibrated'.
"""

import torch.nn as nn

from common import collect_conv_linear_layers


def paired_bn_names(model):
    """Map each Conv2d/Linear layer name to the BatchNorm that consumes it.

    Derived structurally -- the BatchNorm2d immediately following a conv in
    named_modules() order -- rather than by string surgery on the layer name.
    ResNet's BasicBlock declares conv1, bn1, relu, conv2, bn2, downsample in that
    order and downsample is Sequential(conv1x1, norm_layer), so traversal order
    and the naming convention agree; deriving it means an architecture change
    cannot silently desync them.

    `fc` maps to None: it is the last module and has no BatchNorm.
    """
    pairs = {}
    pending_conv = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if pending_conv is not None:
                pairs[pending_conv] = None
            pending_conv = name
        elif isinstance(module, nn.BatchNorm2d) and pending_conv is not None:
            conv = dict(model.named_modules())[pending_conv]
            if module.num_features != conv.out_channels:
                raise ValueError(
                    f"{name} has {module.num_features} features but follows "
                    f"{pending_conv} with {conv.out_channels} out channels; "
                    "the traversal-order pairing is not valid for this model")
            pairs[pending_conv] = name
            pending_conv = None
    if pending_conv is not None:
        pairs[pending_conv] = None

    layers, order = collect_conv_linear_layers(model)
    if set(pairs) != set(order):
        raise ValueError("paired_bn_names disagrees with collect_conv_linear_layers")
    return pairs


def suffix_bn_names(model, trainable_names):
    """BN modules belonging to the re-initialized suffix.

    `trainable_names` is layer_order[K:], so this IS the prefix/suffix split --
    it is derived from the caller's split rather than recomputed from K, which
    keeps the two from ever disagreeing.
    """
    pairs = paired_bn_names(model)
    return [pairs[n] for n in trainable_names if pairs.get(n) is not None]


def reinitialize_bn(model, bn_names):
    """Discard a BN's learned scale/shift AND its recorded statistics.

    Affine reset matches what the architectures' constructors do.
    reset_running_stats() is the part that actually matters for the threat model:
    running_mean/var are recorded from the victim's activations, so leaving them
    would leak victim information into a layer the attacker never recovered.
    """
    modules = dict(model.named_modules())
    for name in bn_names:
        bn = modules[name]
        nn.init.constant_(bn.weight, 1)
        nn.init.constant_(bn.bias, 0)
        bn.reset_running_stats()
    if bn_names:
        print(f"Re-initialized {len(bn_names)} BN layers: {bn_names}")


def build_source_model(base_builder, num_recovered, layer_order):
    """sweepRepair.build_cell_model, plus the BN-prefix rule.

    Returns (model, trainable_names, trainable_params, reinit_bn_names).

    The suffix BN params must be appended to the SAME list the caller hands to
    the optimizer. freeze_all_but sets requires_grad=False on every parameter and
    re-enables only conv/linear ones, and it reaches them through
    collect_conv_linear_layers, which filters to Conv2d/Linear -- so BN is
    structurally unreachable through it and would silently never train.
    """
    model, trainable, params = base_builder(num_recovered, layer_order)
    if num_recovered < 0:
        # True from-scratch: nothing was given, BN included. Already all-random
        # and all-trainable, so the prefix rule has nothing to do.
        return model, trainable, params, []

    bn_names = suffix_bn_names(model, trainable)
    reinitialize_bn(model, bn_names)

    modules = dict(model.named_modules())
    for name in bn_names:
        for param in modules[name].parameters():
            param.requires_grad = True
            params.append(param)
    return model, trainable, params, bn_names


def snapshot_params(model, names):
    """Clone the parameters of arbitrary modules, for a freeze audit.

    common.snapshot_frozen cannot do this: it indexes
    collect_conv_linear_layers(model)[name].weight, so a BN name raises
    KeyError, and it clones only `.weight` while BN also has `.bias`.

    Parameters only, deliberately. running_mean/running_var are SUPPOSED to move
    even on frozen-affine prefix BN (see the module docstring), so auditing the
    full state_dict would fail on every run.
    """
    modules = dict(model.named_modules())
    return {f'{name}.{pname}': param.detach().clone()
            for name in names
            for pname, param in modules[name].named_parameters()}


def assert_params_unchanged(model, snapshot):
    """Verify every audited parameter is bit-identical to its snapshot.

    This is the claim the BN-prefix arm rests on -- prefix BN stayed at ground
    truth while suffix BN moved -- so it is checked, not assumed.
    """
    current = dict(model.named_parameters())
    for key, saved in snapshot.items():
        now = current[key]
        if not now.detach().cpu().equal(saved.cpu()):
            raise AssertionError(f"{key} changed during training but was frozen")
