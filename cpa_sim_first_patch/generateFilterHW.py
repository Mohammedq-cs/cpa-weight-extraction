import numpy as np
import itertools
import time
import tqdm
import itertools
from utils import *
from attack import *
from modelUtils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from ipdb import set_trace
import os
import pickle
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_baseline_inputs(name, numTraces, realInps, base_dir="./baseLineInps"):
    """
    Save the baseline inputs to a pickle file.
    
    Args:
        name (str): Identifier for the file name.
        numTraces (int): Number of traces, used to form the directory name.
        realInps: Baseline inputs to be saved.
        base_dir (str): Base directory where the inputs are stored.
    """
    # Create the directory if it doesn't exist.
    inps_dir_path = os.path.join(base_dir, str(numTraces))
    os.makedirs(inps_dir_path, exist_ok=True)
    
    # Define the full path to the file.
    inps_save_path = os.path.join(inps_dir_path, f'{name}.pkl')
    
    # Save using pickle.
    with open(inps_save_path, 'wb') as f:
        pickle.dump(realInps, f)

def save_filter_hw_files(name, numTraces, filter_hw, base_dir="./filtersHW"):
    """
    Save each filter from the filterHW array as a separate pickle file.
    
    Args:
        name (str): Identifier used to create a subdirectory.
        numTraces (int): Number of traces used to form the directory name.
        filter_hw (numpy.ndarray): Array containing filter data.
        base_dir (str): Base directory where filter files are stored.
    """
    # Create the base directory for filter files.
    filters_dir = os.path.join(base_dir, str(numTraces), name)
    os.makedirs(filters_dir, exist_ok=True)
    
    # Save each filter as an individual pickle file.
    for filter_idx in range(filter_hw.shape[0]):
        filter_data = filter_hw[filter_idx]
        file_path = os.path.join(filters_dir, f'filter_{filter_idx}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(filter_data, f)

def calc_layerInp_in_batches(model, inputTensor, layer, batch_size=10_000):
    """
    Processes the input tensor in two stages:
      1. First 10,000 samples in batches of 500
      2. Remaining samples in batches of 10,000
    Collects intermediate activations from the specified layer and returns the combined result.

    Args:
        model (torch.nn.Module): The PyTorch model.
        inputTensor (torch.Tensor): The large input tensor.
        layer (any): The key or identifier for the layer in the intermediateActivations dict.
    Returns:
        torch.Tensor: The concatenated layer inputs from the specified layer.
    """
    layer_inputs = []
    model.eval()
    total_samples = inputTensor.size(0)
    first_n = min(total_samples, 10_000)

    # Process first 10k samples in batches of 500
    #print(f"Processing first {first_n} samples in batches of 500 for layer {layer}")
    for batch in torch.split(inputTensor[:first_n], 500, dim=0):
        with torch.no_grad():
            batch = batch.to(device).half()
            _ = model(batch)
        layer_input, _ = intermediateActivations[layer]
        layer_inputs.append(layer_input.cpu())
        del _

    # Process remaining samples in batches of 10k
    if total_samples > first_n:
        #print(f"Processing remaining {total_samples - first_n} samples in batches of 10000 for layer {layer}")
        for batch in torch.split(inputTensor[first_n:], batch_size, dim=0):
            with torch.no_grad():
                batch = batch.to(device).half()
                _ = model(batch)
            layer_input, _ = intermediateActivations[layer]
            layer_inputs.append(layer_input.cpu())
            del _

    # Concatenate and return all collected activations
    return torch.cat(layer_inputs, dim=0)

def extractFilterHW(
    model: nn.Module,
    input_dataloader: torch.utils.data.DataLoader,
    numTraces: int,
    exp_name: str
) -> list:
    """
    Evaluates the model by processing each layer and calculating accuracy.

    Args:
        model (nn.Module): The neural network model.
        inputTensor (torch.Tensor): Input tensor for the model.
        numTraces (int): Number of traces.
        noiseLevel (int): Noise level parameter.
        numProcesses (int): Number of parallel processes.
    """


    #add here loading the last state
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layerType = "Convolutional Layer" if isinstance(layer, nn.Conv2d) else "Fully Connected Layer"
            print(f"{layerType}: {name}")
            if isinstance(layer, nn.Conv2d):
                print(f"Number of Filters (Out Channels): {layer.out_channels}")
                print(f"Filter Shape (Including Input Channels): {layer.weight.data.shape}")
                numFilters = layer.out_channels  # For Conv2d layers, use out_channels as the element count
            else:
                print(f"Weights: {layer.weight.data.shape}")
                numFilters = layer.weight.data.shape[0]  # For Linear layers, use weight's first dimension

            base_dir = f"./baseLineInps/{exp_name}"
            inps_dir_path = os.path.join(base_dir, str(numTraces))
            inps_save_path = os.path.join(inps_dir_path, f'{name}.pkl')

            # If the file already exists, do nothing
            if os.path.isfile(inps_save_path):
                print(f"Baseline inputs for layer {name} already exist. Skipping computation.")
                continue
    
            layerInput = calc_layerInp_in_batches(model, input_dataloader, layer)

            if isinstance(layer, nn.Conv2d):
                    in_ch  = layer.in_channels
                    kH, kW = layer.kernel_size
                    numWeights = in_ch * kH * kW

                    realInps = (
                        layerInput[:, :in_ch, :kH, :kW]
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                        .reshape((-1, numWeights))
                    )

            elif isinstance(layer, nn.Linear):
                    numWeights = layer.in_features

                    realInps = (
                        layerInput[:, :numWeights]
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )
            save_baseline_inputs(name, numTraces, realInps, base_dir=base_dir)

            sys.stdout.flush()

def runFilterHW(model, inputTensor, numTraces, exp_name):
    hooks = setupHooks(model)

    extractFilterHW(model, inputTensor, numTraces, exp_name)

    removeHooks(hooks)
    