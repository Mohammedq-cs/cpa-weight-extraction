"""
Input-distribution comparison entry point (single-node, in-process).

Runs the multi-patch CPA pipeline over one of
cifar10 / cifar6m / random / c6m_low_acc / c6m_high_acc and
saves the recovered .pth -- built on the in-process pipeline (run_inproc +
ColumnProvider + multi-GPU pool), exactly like demoInproc.py:
  * NO baseline/sim memmaps on NFS; both models held in memory, activations
    served on the fly,
  * per-filter extraction runs in-process (inline single GPU, or the node's GPUs
    via the worker pool), not as SLURM array jobs,
  * the fc2 sign-flip fix is applied at the end.
Only the INPUT selection differs from demoInproc.py -- the extraction math and
orchestration are identical.

The working model is zeroed (not noised) as the starting state: run_inproc
overwrites every layer's weights with the recovered ones before that layer's
activation is ever read, and a layer's input depends only on PRIOR layers, so the
initial weights of a layer never affect its extraction (verified equal to the
noised-init multipatch path).
"""
import os
import sys
import copy
import pickle
import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from utils import calculate_accuracy, save_model_weights
from datasets import CIFAR5MDataset, CIFAR10_ROOT, CIFAR5M_ROOT
from modelUtils import zero_out_model_weights
from convNet import Net
from simulationInproc import run_inproc, InlineExecutor
from lastLayerFix import correct_last_layer_signs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_executor(num_gpus):
    if num_gpus and num_gpus > 1:
        from workerPool import PoolExecutor
        return PoolExecutor(num_gpus)
    return InlineExecutor()


def load_bucket_indices(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    indices = data["indices"]
    print(f"Loaded {indices.shape[0]} indices ; info: {data.get('description', '')}")
    return indices


def make_bucket_loader(train_dataset, bucket_idxs, batch_size=30_000, num_workers=8):
    subset = Subset(train_dataset, bucket_idxs.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def get_shuffle_mask(num_elements, seed=8):
    return np.random.default_rng(seed).permutation(num_elements)


def select_input_tensor(dataset_name, transform, model, test_loader, seed):
    """Build the attack input_tensor for the chosen distribution (mirrors the
    multipatch inpDist.py dataset branch)."""
    if dataset_name == 'cifar10':
        images_test, _ = next(iter(test_loader))
        train_loader = DataLoader(
            CIFAR10(root=CIFAR10_ROOT, train=True, transform=transform, download=True),
            batch_size=60_000, shuffle=False)
        images_train, _ = next(iter(train_loader))
        images = torch.cat((images_train, images_test), dim=0)
        return images[get_shuffle_mask(images.shape[0], seed=seed)]

    if dataset_name == 'random':
        return torch.rand(200_000, 3, 32, 32) * 2 - 1   # U(-1, 1): uniform pixels after Normalize(0.5, 0.5)

    # cifar6m / c6m_low_acc / c6m_high_acc all read CIFAR5M part0
    ds = CIFAR5MDataset(root_dir=CIFAR5M_ROOT, parts=["part0.npz"], transform=transform)
    if dataset_name == 'cifar6m':
        loader = DataLoader(ds, batch_size=2_000_000, shuffle=False, num_workers=16)
    else:
        idx_file = {'c6m_low_acc': "./buckets_pickles/low_indices.pkl",
                    'c6m_high_acc': "./buckets_pickles/high_indices.pkl"}[dataset_name]
        loader = make_bucket_loader(ds, load_bucket_indices(idx_file),
                                    batch_size=30_000, num_workers=16)
        print(f"Bucket Loader Acc = {calculate_accuracy(model, loader)}")
    images, _ = next(iter(loader))
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="In-process single-node multi-patch CPA over a chosen input distribution.")
    parser.add_argument("--model_save_path", required=True)
    parser.add_argument("--num_traces", type=int, required=True)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--leakage_model", type=str, default='HW', choices=['HW', 'HD'])
    parser.add_argument("--exper_name", type=str, default='inp_dis_comp')
    parser.add_argument("--dataset_name", type=str, default='cifar10',
                        choices=['cifar10', 'cifar6m', 'random', 'c6m_low_acc', 'c6m_high_acc'])
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--layers_subset", type=str, default=None)
    args = parser.parse_args()

    seed = args.seed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- frozen true model (baseline activations + true weights) ----
    true_model = Net().to(device)
    true_model.load_state_dict(torch.load('model_cifar.pt'))
    true_model = true_model.half().eval()
    save_model_weights(true_model, './model_weights')

    cifar10_test = CIFAR10(root=CIFAR10_ROOT, train=False, transform=transform, download=True)
    test_loader = DataLoader(cifar10_test, batch_size=10_000, shuffle=False)
    print(f"OGModel = {calculate_accuracy(true_model, test_loader)}")
    sys.stdout.flush()

    # ---- attack inputs for the chosen distribution ----
    input_tensor = select_input_tensor(args.dataset_name, transform, true_model, test_loader, seed)
    input_tensor = input_tensor[:int(args.num_traces)]
    numTraces = input_tensor.shape[0]
    print(f"numTraces={numTraces}, noise={args.noise_level}, leak={args.leakage_model}, "
          f"dataset={args.dataset_name}, gpus={args.num_gpus}")
    sys.stdout.flush()

    # ---- zeroed working model (simulated activations, updated per layer) ----
    working_model = copy.deepcopy(true_model)
    zero_out_model_weights(working_model)
    print(f"Zerod Model = {calculate_accuracy(working_model, test_loader)}")
    sys.stdout.flush()

    # Same path scheme as demoInproc.py:
    #   exp_name = seed{seed}/{exper_name}/{leakage}/{dataset}
    exp_name = f"seed{seed}/{args.exper_name}/{args.leakage_model}/{args.dataset_name}"
    output_dir = args.output_dir or f"./N{args.noise_level}/{exp_name}output/{numTraces}"
    corr_dir = f"./N{args.noise_level}/{exp_name}Corr/{numTraces}"
    scripts_dir = f"./scriptsOut/{exp_name}/{numTraces}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(corr_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)
    layers_subset = args.layers_subset.split(",") if args.layers_subset else None

    executor = build_executor(args.num_gpus)
    accLayers, modifiedModel = run_inproc(
        true_model, working_model, input_tensor, numTraces,
        noise_level=args.noise_level, leakage_model=args.leakage_model,
        output_dir=output_dir, seed=seed, executor=executor,
        layers_subset=layers_subset, corr_dir=corr_dir, scripts_dir=scripts_dir,
    )
    print("per-layer acc:", [round(float(a), 4) for a in accLayers])
    sys.stdout.flush()

    print(f"ModifiedModel (pre-fix) = {calculate_accuracy(modifiedModel, test_loader)}")
    correct_last_layer_signs(modifiedModel, modifiedModel.fc2, test_loader, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.model_save_path)), exist_ok=True)
    torch.save(modifiedModel.state_dict(), args.model_save_path)
    print(f"ModifiedModel (post-fix) = {calculate_accuracy(modifiedModel, test_loader)}")
    sys.stdout.flush()
    if hasattr(executor, "shutdown"):
        executor.shutdown()
