"""
Entry point for the single-node, in-process multi-patch CPA extraction.

Compatible with submit_experiment.sh, plus --num_gpus and --output_dir. Key
properties of this in-process path:
  * NO baseline/sim memmaps are written to NFS -- both the frozen true_model and
    the zeroed working_model are held in memory and their activations are
    captured + served on the fly (columnProvider).
  * per-filter extraction runs in-process (inline, single GPU) or across the
    node's GPUs (worker pool), not as SLURM array jobs.
  * after extraction, the last-layer (fc2) sign-flip fix is applied.
The extraction math itself is unchanged.
"""
import os
import sys
import copy
import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="In-process single-node multi-patch CPA.")
    parser.add_argument("--model_save_path", required=True)
    parser.add_argument("--num_traces", type=int, required=True)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--leakage_model", type=str, default='HW', choices=['HW', 'HD'])
    parser.add_argument("--exper_name", type=str, default='accuracy_experiment')
    parser.add_argument("--dataset_name", type=str, default='cifar6m',
                        choices=['cifar10', 'cifar6m', 'random'])
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="checkpoint dir (default ./out_inproc/...)")
    parser.add_argument("--layers_subset", type=str, default=None,
                        help="comma-separated layer names to restrict to (debug)")
    args = parser.parse_args()

    seed = args.seed
    numTraces = int(args.num_traces)
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
    true_model = true_model.half()
    true_model.eval()
    save_model_weights(true_model, './model_weights')

    cifar10_test = CIFAR10(root=CIFAR10_ROOT, train=False, transform=transform, download=True)
    test_loader = DataLoader(cifar10_test, batch_size=10_000, shuffle=False)
    print(f"OGModel = {calculate_accuracy(true_model, test_loader)}")
    sys.stdout.flush()

    # ---- inputs (CIFAR5M traces) ----
    train_dataset = CIFAR5MDataset(root_dir=CIFAR5M_ROOT, parts=["part0.npz"], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=2_000_000, shuffle=False, num_workers=16)
    images, _ = next(iter(train_loader))
    input_tensor = images[:numTraces]
    numTraces = input_tensor.shape[0]
    print(f"numTraces={numTraces}, noise={args.noise_level}, leak={args.leakage_model}, gpus={args.num_gpus}")
    sys.stdout.flush()

    # ---- zeroed working model (simulated activations, updated per layer) ----
    working_model = copy.deepcopy(true_model)
    zero_out_model_weights(working_model)
    print(f"Zerod Model = {calculate_accuracy(working_model, test_loader)}")
    sys.stdout.flush()

    # Per-filter outputs at these paths:
    #   weights -> ./N{noise}/{exp_name}output/{numTraces}/{layer}/filter_{idx}.pkl
    #   corr    -> ./N{noise}/{exp_name}Corr/{numTraces}/{layer}/...
    # where exp_name = seed{seed}/{exper_name}/{leakage_model}/{dataset_name}.
    exp_name = f"seed{seed}/{args.exper_name}/{args.leakage_model}/{args.dataset_name}"
    output_dir = args.output_dir or f"./N{args.noise_level}/{exp_name}output/{numTraces}"
    corr_dir = f"./N{args.noise_level}/{exp_name}Corr/{numTraces}"
    scripts_dir = f"./scriptsOut/{exp_name}/{numTraces}"   # per-filter extractor chatter
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
    # ---- last-layer sign-flip fix (fc2 has no BN) ----
    correct_last_layer_signs(modifiedModel, modifiedModel.fc2, test_loader, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.model_save_path)), exist_ok=True)
    torch.save(modifiedModel.state_dict(), args.model_save_path)
    print(f"ModifiedModel (post-fix) = {calculate_accuracy(modifiedModel, test_loader)}")
    sys.stdout.flush()
    if hasattr(executor, "shutdown"):
        executor.shutdown()
