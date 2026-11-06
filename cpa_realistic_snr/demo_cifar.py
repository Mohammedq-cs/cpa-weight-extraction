import os
import numpy as np
import itertools
from utils import *
from attack import *
from datasets import *
from modelUtils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
import time
from ipdb import set_trace
import multiprocessing
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageNet
from generateFilterHW import *
from convNet import *
import patchInputsMaterialized as PM
import torchvision
import torchvision.transforms as transforms
import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_shuffle_mask(num_elemnts, seed: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.permutation(num_elemnts)
    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CPA.")
    parser.add_argument("--model_save_path", required=True, help="Path to save your extracted Model.")
    parser.add_argument("--num_traces", type=int, required=True)
    parser.add_argument('--noise_level', type=float, default=0.0, help='')
    parser.add_argument(
        '--leakage_model',
        type=str,
        default='HW',
        help='Leakage model to use (default: HW). Options: HW, HD',
        choices=['HW', 'HD']
    )
    parser.add_argument(
        '--exper_name',
        type=str,
        default='accuracy_experiment',
        help='Path to the experiment file (default: accuracy_experiment).'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='cifar10',
        help='Dataset to use (default: cifar10). Options: cifar10, cifar6m, random',
        choices=['cifar10', 'cifar6m', 'random']
    )
    parser.add_argument('--seed', type=int, default=8, help='')
    parser.add_argument('--multipatch', action='store_true',
                        help='Materialize per-weight best-activation patches (union, '
                             'column-major memmaps) instead of the single top-left patch.')
    parser.add_argument('--layers', type=str, default='conv1',
                        help='Comma-separated layers to materialize in multipatch mode '
                             '(default: conv1).')
    parser.add_argument('--selection_images', type=int, default=8000,
                        help='Images used to pick each weight\'s best patch (multipatch).')
    parser.add_argument('--max_union', type=int, default=0,
                        help='Cap the per-layer patch union to this many patches (0 = no cap).')

    args = parser.parse_args()
    traces_use = int(args.num_traces)
    model_save_path = args.model_save_path
    noise_level = args.noise_level
    leakage_model = args.leakage_model
    exper_name = args.exper_name
    dataset_name = args.dataset_name
    seed = args.seed

    # original transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # augmented transform: small horizontal flips + small shifts
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=0,
            translate=(2/32, 2/32)   # up to 2-pixel shift
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Using deterministic CuDNN: {torch.backends.cudnn.deterministic}")

    g = torch.Generator()
    g.manual_seed(seed)

    model = Net().to(device)
    model.load_state_dict(torch.load('model_cifar.pt'))
    model = model.half()
    model.eval()

    print(f"Seed={seed}")
    print(f"CuDNN enabled = {torch.backends.cudnn.enabled}")
    save_model_weights(model, './model_weights')

    cifar10_test_data = CIFAR10(root=CIFAR10_ROOT, train=False, transform=transform, download=True)
    cifar10_test_loader = DataLoader(
        cifar10_test_data,
        batch_size=10_000,
        shuffle=True
    )

    print(f"OGModel = {calculate_accuracy(model, cifar10_test_loader)}")

    # -------------------------------
    # ORIGINAL 6M DATASET
    # -------------------------------
    train_dataset = CIFAR5MDataset(
        root_dir=CIFAR5M_ROOT,
        parts=["part0.npz", "part1.npz", "part2.npz", "part3.npz", "part4.npz", "part5.npz"],
        transform=transform
    )

    # -------------------------------
    # FULL AUGMENTED COPY OF SAME 6M
    # -------------------------------
    aug_dataset_full = CIFAR5MDataset(
        root_dir=CIFAR5M_ROOT,
        parts=["part0.npz", "part1.npz", "part2.npz", "part3.npz", "part4.npz", "part5.npz"],
        transform=aug_transform
    )

    # -------------------------------
    # EXTRA 240K AUGMENTED SUBSET
    # to reach exactly 12,240,000
    # -------------------------------
    target_size = 12_240_000
    base_size = len(train_dataset)

    remaining = target_size - 2 * base_size
    assert remaining >= 0, f"Target {target_size} is too small for base_size={base_size}"

    extra_indices = list(range(remaining))
    extra_aug_subset = Subset(aug_dataset_full, extra_indices)

    final_dataset = ConcatDataset([
        train_dataset,      # 6,000,000
        aug_dataset_full,   # 6,000,000
        extra_aug_subset    #   240,000
    ])

    print(f"base_size = {base_size}")
    print(f"final_dataset size = {len(final_dataset)}")

    # if you want to cap by args.num_traces, use a subset here
    if traces_use < len(final_dataset):
        final_dataset = Subset(final_dataset, range(traces_use))
        print(f"Using only first {len(final_dataset)} traces")

    train_loader = DataLoader(
        final_dataset,
        batch_size=100_000,
        shuffle=False,
        num_workers=40,
    )

    exp_name = f'seed{seed}/{exper_name}/{leakage_model}/{dataset_name}'
    input_tensor = train_loader

    numTraces = len(train_loader.dataset)
    print(f"\nnumTraces = {numTraces}, noiseLevel= {noise_level}, leakageModel= {leakage_model}\n")

    bsInpsPath = f'{exper_name}/{dataset_name}'

    if args.multipatch:
        layer_names = [s for s in args.layers.split(',') if s]
        # Representative selection sample: seeded random subset (CIFAR-5M is i.i.d.,
        # but draw randomly across the whole set to be safe).
        sel_rng = np.random.default_rng(seed)
        n_sel = min(args.selection_images, numTraces)
        sel_idx = sel_rng.permutation(numTraces)[:n_sel].tolist()
        sel_loader = DataLoader(Subset(final_dataset, sel_idx), batch_size=2000,
                                shuffle=False, num_workers=20)
        print(f"MULTIPATCH baseline for layers={layer_names}, "
              f"selection_images={n_sel}, max_union={args.max_union or None}")
        PM.generate_baseline_materialized(
            model, train_loader, numTraces,
            base_dir=f'./baseLineInps/{bsInpsPath}',
            layer_names=layer_names,
            selection_images=n_sel,
            fwd_batch=2000,
            epochs=1,
            selection_source=sel_loader,
            max_union=(args.max_union or None),
        )
        print("Done: multipatch baseline materialized. Exiting (extraction runs separately).")
        exit(0)

    runFilterHW(model, input_tensor, numTraces, exp_name=bsInpsPath)
    print("Done: baseline inputs generated. Exiting (extraction runs separately).")
    exit(0)   # success path -- see note in demo_random.py about afterok dependencies
