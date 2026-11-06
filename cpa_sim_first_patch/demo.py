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
from generateFilterHW import *
from convNet import *
import torchvision
import torchvision.transforms as transforms
import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_shuffle_mask(num_elemnts, seed: int = 8) -> np.ndarray:
    """
    Generates a reproducible permutation mask for axis‐0 of `tensor`.

    Args:
        tensor (np.ndarray): any array with shape (N, …)
        seed (int): RNG seed for reproducibility.

    Returns:
        mask (np.ndarray): shape (N,), a permutation of [0, 1, …, N-1].
    """
    rng = np.random.default_rng(seed)
    mask = rng.permutation(num_elemnts)
    return mask


if __name__ == '__main__':
    # Ensure that we use the 'spawn' start method to avoid CUDA issues
    parser = argparse.ArgumentParser(
        description="Run CPA."
    )
    parser.add_argument(
        "--model_save_path",
        required=True,
        help="Path to save your extracted Model."
    )

    parser.add_argument(
        "--num_traces",
        type=int,
        required=True
    )
    parser.add_argument(
        '--noise_level',
          type=float, default=0.0,
        help='Index of the filter.')
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

    args = parser.parse_args()
    traces_use = int(args.num_traces)
    model_save_path = args.model_save_path
    noise_level = args.noise_level
    leakage_model = args.leakage_model
    exper_name = args.exper_name
    dataset_name = args.dataset_name

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    seed = 8
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Ensure deterministic CUDA behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Using deterministic CuDNN: {torch.backends.cudnn.deterministic}")

    # Generator for DataLoader
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

    train_dataset = CIFAR5MDataset(
    root_dir=CIFAR5M_ROOT,
    parts=["part0.npz"],
    transform=transform
    )
    #50k
    train_loader = DataLoader(train_dataset, batch_size=2_000_000, shuffle=False, num_workers=40)

    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    exp_name = f'{exper_name}/{leakage_model}/{dataset_name}'
    input_tensor = images

    input_tensor = input_tensor[:traces_use, ]    
    numTraces = input_tensor.shape[0]
    print(f"\nnumTraces = {numTraces}, noiseLevel= {noise_level}, leakageModel= {leakage_model}\n")
    
    runFilterHW(model, input_tensor, numTraces, exp_name = exp_name)

    zero_out_model_weights(model)
    print(f"Zerod Model = {calculate_accuracy(model, cifar10_test_loader)}")

    accLayers, modifiedModel = runWeightsRetrievalSimulation(model, input_tensor, numTraces, noise_level=noise_level, leakage_model=leakage_model, exp_name=exp_name)
    #save modified model
    torch.save(modifiedModel.state_dict(), model_save_path)

    print(accLayers)
    print(f"ModifiedModel = {calculate_accuracy(modifiedModel, cifar10_test_loader)}")
