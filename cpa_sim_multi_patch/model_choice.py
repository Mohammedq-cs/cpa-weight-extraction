"""
Random-model variant entry point (single-node, in-process).

Runs the multi-patch CPA pipeline against one of
preTrained / randInit / randUniform network weights, with
uniform-noise (U(-1, 1), i.e. uniform pixels after Normalize(0.5, 0.5)) attack inputs -- built on the in-process pipeline
(run_inproc + ColumnProvider + multi-GPU pool), exactly like demoInproc.py:
  * NO baseline/sim memmaps on NFS; both models held in memory,
  * per-filter extraction runs in-process (inline or worker pool), not SLURM,
  * the fc2 sign-flip fix is applied at the end.
Only the CHOSEN MODEL (the true_model being attacked) and the random inputs
differ from demoInproc.py; the extraction math and orchestration are identical.

The working model is zeroed (its initial weights are overwritten layer by layer
and never affect extraction -- see inpDist.py for the same note).
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

from datasets import CIFAR10_ROOT
from utils import calculate_accuracy, save_model_weights
from modelUtils import zero_out_model_weights
from convNet import Net
from simulationInproc import run_inproc, InlineExecutor
from lastLayerFix import correct_last_layer_signs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATHS = {
    'preTrained': 'model_cifar.pt',
    'randInit': './model_rand_init.pth',
    'randUniform': './model_rand_uniform.pth',
}


def build_executor(num_gpus):
    if num_gpus and num_gpus > 1:
        from workerPool import PoolExecutor
        return PoolExecutor(num_gpus)
    return InlineExecutor()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="In-process single-node multi-patch CPA against a chosen model variant.")
    parser.add_argument("--model_save_path", required=True)
    parser.add_argument("--num_traces", type=int, required=True)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--leakage_model", type=str, default='HW', choices=['HW', 'HD'])
    parser.add_argument("--exper_name", type=str, default='accuracy_experiment')
    parser.add_argument("--model", type=str, default='preTrained',
                        choices=['preTrained', 'randUniform', 'randInit'])
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--layers_subset", type=str, default=None)
    args = parser.parse_args()

    seed = args.seed
    model_use = args.model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- frozen true model (the chosen variant) ----
    true_model = Net().to(device)
    true_model.load_state_dict(torch.load(MODEL_PATHS[model_use]))
    true_model = true_model.half().eval()
    print(f"Seed={seed}, model={model_use}")
    save_model_weights(true_model, './model_weights')

    cifar10_test = CIFAR10(root=CIFAR10_ROOT, train=False, transform=transform, download=True)
    test_loader = DataLoader(cifar10_test, batch_size=10_000, shuffle=False)
    print(f"OGModel = {calculate_accuracy(true_model, test_loader)}")
    sys.stdout.flush()

    # ---- attack inputs: pure noise ----
    numTraces = int(args.num_traces)
    input_tensor = torch.rand(numTraces, 3, 32, 32) * 2 - 1   # U(-1, 1)
    print(f"numTraces={numTraces}, noise={args.noise_level}, leak={args.leakage_model}, "
          f"model={model_use}, gpus={args.num_gpus}")
    sys.stdout.flush()

    # ---- zeroed working model (simulated activations, updated per layer) ----
    working_model = copy.deepcopy(true_model)
    zero_out_model_weights(working_model)
    print(f"Zerod Model = {calculate_accuracy(working_model, test_loader)}")
    sys.stdout.flush()

    # Path scheme mirrors demoInproc.py with the model variant as the leaf:
    #   exp_name = seed{seed}/{exper_name}/{leakage}/{model}
    exp_name = f"seed{seed}/{args.exper_name}/{args.leakage_model}/{model_use}"
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
