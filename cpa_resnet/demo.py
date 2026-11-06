"""
Entry point for first-weight-known multi-patch CPA extraction on ResNet-18.

Derived from vgg_e2e/demoSignAmbiguous.py but:
  * loads the CIFAR-adapted ResNet-18 (from resnet18.pt) instead of VGG-11,
  * drops every sign-correction CLI flag (no oracle pass, no terminal
    per-class-accuracy fix) -- the per-filter extractor is seeded with the TRUE
    first weight so recovered filters are not sign-ambiguous,
  * calls `runWeightsRetrievalSimulationFirstWeightKnown` from this pipeline's
    simulation.py.
"""

import argparse
import copy

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from utils import save_model_weights, calculate_accuracy
from modelUtils import zero_out_model_weights
from datasets import CIFAR5MDataset, CIFAR10_ROOT, CIFAR5M_ROOT
from simulation import runWeightsRetrievalSimulationFirstWeightKnown
from convNet import resnet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="First-weight-known multi-patch CPA extraction on ResNet-18."
    )
    parser.add_argument("--model_save_path", required=True,
                        help="Path to save the extracted model.")
    parser.add_argument("--num_traces", type=int, required=True)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--leakage_model", type=str, default='HW',
                        choices=['HW', 'HD'])
    parser.add_argument("--exper_name", type=str, default='first_weight_known_experiment')
    parser.add_argument("--dataset_name", type=str, default='cifar10',
                        choices=['cifar10', 'cifar6m', 'random'])
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--start_at_layer", type=str, default=None,
                        help="Testing fast-forward / resume: skip extraction/eval for all layers "
                             "before this one, only loading their already-recovered weights into "
                             "the model. Requires earlier layers' per-filter pickles to exist on "
                             "disk. Use this to resume a run from the layer that was in progress.")
    parser.add_argument("--keep_layer_memmaps", action="store_true",
                        help="Keep each layer's transient sim+baseline patch memmaps on disk after "
                             "extraction instead of deleting them. Default is to delete them so "
                             "disk use stays bounded (leaving them accumulates ~tens of GB per "
                             "layer and exhausts the quota mid-run).")
    args = parser.parse_args()

    traces_use = int(args.num_traces)
    model_save_path = args.model_save_path
    noise_level = args.noise_level
    leakage_model = args.leakage_model
    exper_name = args.exper_name
    dataset_name = args.dataset_name
    seed = args.seed
    start_at_layer = args.start_at_layer
    cleanup_memmaps = not args.keep_layer_memmaps

    # General CIFAR-10 channel stats -- NOT Normalize(0.5).
    # resnet18.pt was trained with these stats: verified CIFAR-10 test accuracy is
    # 92.59% with these vs 86.54% under Normalize(0.5), so 0.5 is a train/inference
    # mismatch that corrupts the very activations the CPA attack leaks from. Keep
    # these values; do not revert to 0.5.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))
    ])

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Using deterministic CuDNN: {torch.backends.cudnn.deterministic}")

    g = torch.Generator()
    g.manual_seed(seed)

    # Two models: `true_model` stays the ground-truth device
    # and supplies the BASELINE (measured) leakage; `working_model` starts zeroed
    # and is filled in layer by layer as weights are recovered -- it supplies the
    # SIMULATED (attacker-side) activations that DRIVE patch selection.
    true_model = resnet18().to(device)
    true_model.load_state_dict(torch.load('resnet18.pt'))
    true_model = true_model.half()
    true_model.eval()

    print(f"Seed={seed}")
    print(f"CuDNN enabled = {torch.backends.cudnn.enabled}")
    save_model_weights(true_model, './model_weights')

    working_model = copy.deepcopy(true_model)
    zero_out_model_weights(working_model)
    working_model.eval()

    cifar10_test_data = CIFAR10(root=CIFAR10_ROOT, train=False, transform=transform, download=True)
    cifar10_test_loader = DataLoader(
        cifar10_test_data, batch_size=10_000, shuffle=True,
    )
    print(f"OGModel = {calculate_accuracy(true_model, cifar10_test_loader)}")

    train_dataset = CIFAR5MDataset(
        root_dir=CIFAR5M_ROOT,
        parts=["part0.npz"],
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=2_000_000, shuffle=False, num_workers=40,
    )
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    exp_name = f'seed{seed}/{exper_name}/{leakage_model}/{dataset_name}'
    input_tensor = images[:traces_use, ]
    numTraces = input_tensor.shape[0]
    print(f"\nnumTraces = {numTraces}, noiseLevel= {noise_level}, leakageModel= {leakage_model}\n")

    bsInpsPath = f'{exper_name}/{dataset_name}'
    print(f"Zerod Model = {calculate_accuracy(working_model, cifar10_test_loader)}")

    # No up-front baseline generation: patch selection is decided per layer from
    # the SIMULATED (working_model) activations, and both the simulated and the
    # baseline (true_model) patch memmaps for that selection are written together
    # inside the per-layer step (see simulation._generate_sim_inputs_and_submit).
    accLayers, modifiedModel = runWeightsRetrievalSimulationFirstWeightKnown(
        working_model, true_model, input_tensor, numTraces,
        noise_level=noise_level, leakage_model=leakage_model,
        exp_name=exp_name, bsInpsPath=bsInpsPath,
        start_at_layer=start_at_layer, cleanup_memmaps=cleanup_memmaps,
    )

    torch.save(modifiedModel.state_dict(), model_save_path)
    print(accLayers)
    print(f"ModifiedModel = {calculate_accuracy(modifiedModel, cifar10_test_loader)}")
