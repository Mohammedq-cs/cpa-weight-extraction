import numpy as np
import itertools
from utils import *
from attack import *
from datasets import *
from modelUtils import *
from simulation import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from ipdb import set_trace
import multiprocessing
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageNet
from patchInputs import generate_baseline_inputs_hooked
from convNet import *
import torchvision
import torchvision.transforms as transforms
import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_shuffle_mask(num_elemnts, seed: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(num_elemnts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run CPA with sign-ambiguous extraction + reactive sign correction (VGG-11)."
    )
    parser.add_argument("--model_save_path", required=True,
                        help="Path to save your extracted model.")
    parser.add_argument("--num_traces", type=int, required=True)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--leakage_model", type=str, default='HW',
                        choices=['HW', 'HD'])
    parser.add_argument("--exper_name", type=str, default='sign_ambiguous_experiment')
    parser.add_argument("--dataset_name", type=str, default='cifar10',
                        choices=['cifar10', 'cifar6m', 'random'])
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--g_idx", type=int, default=0,
                        help="Index of the downstream oracle filter g used per layer for reactive sign correction. Default: 0.")
    parser.add_argument("--fc_num_probes", type=int, default=8,
                        help="Number of downstream FC units used as voting probes for the FC-oracle sign-correction path (only the last conv uses this). Default: 8.")
    parser.add_argument("--start_at_layer", type=str, default=None,
                        help="Testing fast-forward: skip extraction/sign-correction/eval for all layers before this one, only loading their already-recovered weights into the model. Requires earlier layers' per-filter pickles to exist on disk. E.g. --start_at_layer features.18.")
    parser.add_argument("--save_unique_counts", action="store_true",
                        help="Diagnostic: per layer, save the per-weight input distinct-value counts over the saved patches, for the simulated inputs and the baseline inputs, under ./uniqueCounts/N{noise}/{exper_name}/{numTraces}/{layer}.unique_counts_diag.pkl.")

    args = parser.parse_args()
    traces_use = int(args.num_traces)
    model_save_path = args.model_save_path
    noise_level = args.noise_level
    leakage_model = args.leakage_model
    exper_name = args.exper_name
    dataset_name = args.dataset_name
    seed = args.seed
    g_idx = args.g_idx
    fc_num_probes = args.fc_num_probes
    start_at_layer = args.start_at_layer
    save_unique_counts = args.save_unique_counts

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

    model = cifar10_vgg11_bn().to(device)
    model.load_state_dict(torch.load('cifar10_vgg11_bn-eaeebf42.pt'))
    model = model.half()
    model.eval()

    print(f"Seed={seed}")
    print(f"CuDNN enabled = {torch.backends.cudnn.enabled}")
    save_model_weights(model, './model_weights')

    cifar10_test_data = CIFAR10(root=CIFAR10_ROOT, train=False, transform=transform, download=True)
    cifar10_test_loader = DataLoader(
        cifar10_test_data, batch_size=10_000, shuffle=True,
    )
    print(f"OGModel = {calculate_accuracy(model, cifar10_test_loader)}")

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
    # Per-weight highest-unique-value patch selection + column-major memmaps,
    # written once from the REAL model (selection persisted for the sim step).
    generate_baseline_inputs_hooked(
        model, input_tensor, numTraces,
        base_dir=f'./baseLineInps/{bsInpsPath}',
    )

    zero_out_model_weights(model)
    print(f"Zerod Model = {calculate_accuracy(model, cifar10_test_loader)}")

    accLayers, modifiedModel = runWeightsRetrievalSimulationSignAmbiguous(
        model, input_tensor, numTraces,
        noise_level=noise_level, leakage_model=leakage_model,
        exp_name=exp_name, bsInpsPath=bsInpsPath,
        g_idx=g_idx, fc_num_probes=fc_num_probes,
        start_at_layer=start_at_layer,
        test_loader=cifar10_test_loader,
        save_unique_counts=save_unique_counts,
    )

    torch.save(modifiedModel.state_dict(), model_save_path)
    print(accLayers)
    print(f"ModifiedModel = {calculate_accuracy(modifiedModel, cifar10_test_loader)}")
