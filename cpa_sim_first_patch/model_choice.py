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
from torch.utils.data import DataLoader, Subset
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
        '--model',
        type=str,
        default='pretrained',
        help='model to use (default: pretrained). Options: preTrained, randUniform, randInit',
        choices=['preTrained', 'randUniform', 'randInit']
    )

    args = parser.parse_args()
    traces_use = int(args.num_traces)
    model_save_path = args.model_save_path
    noise_level = args.noise_level
    leakage_model = args.leakage_model
    exper_name = args.exper_name
    model_use = args.model
    
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
    if model_use == 'preTrained':
        model.load_state_dict(torch.load('model_cifar.pt'))
        
    elif model_use == 'randInit':
        model.load_state_dict(torch.load('./model_rand_init.pth'))
    
    elif model_use == 'randUniform':
        model.load_state_dict(torch.load('./model_rand_uniform.pth'))
    else:
        raise ValueError("No such option")
        
    
    model = model.half()
    model.eval()
    
    print(f"Seed={seed}")
    print(f"CuDNN enabled = {torch.backends.cudnn.enabled}")
    save_model_weights(model, './model_weights')
    
    cifar10_test_data = CIFAR10(root=CIFAR10_ROOT, train=False, transform=transform, download=True)
    cifar10_test_loader = DataLoader(
        cifar10_test_data,
        batch_size=10_000,
        shuffle=False
    )
    cifar10_train_data = CIFAR10(root=CIFAR10_ROOT, train=True, transform=transform, download=True)
    cifar10_train_loader = DataLoader(
        cifar10_train_data,
        batch_size=60_000,
        shuffle=False
    )    


    print(f"OGModel = {calculate_accuracy(model, cifar10_test_loader)}")

    # random inputs: uniform pixels in [0,1] after Normalize(0.5, 0.5) -> U(-1, 1)
    input_tensor = torch.rand(traces_use, 3, 32, 32) * 2 - 1
    
    exp_name = f'{exper_name}/{leakage_model}/{model_use}'
    

    input_tensor = input_tensor[:traces_use, ]    
    numTraces = input_tensor.shape[0]
    print(f"\nnumTraces = {numTraces}, noiseLevel= {noise_level}, leakageModel= {leakage_model}\n")
    
    runFilterHW(model, input_tensor, numTraces, exp_name = exp_name)

    add_noise_to_model_weights(model)
    print(f"Noised Model = {calculate_accuracy(model, cifar10_test_loader)}")

    accLayers, modifiedModel = runWeightsRetrievalSimulation(model, input_tensor, numTraces, noise_level=noise_level, leakage_model=leakage_model, exp_name=exp_name)
    #save modified model
    torch.save(modifiedModel.state_dict(), model_save_path)

    print(accLayers)
    print(f"ModifiedModel = {calculate_accuracy(modifiedModel, cifar10_test_loader)}")
