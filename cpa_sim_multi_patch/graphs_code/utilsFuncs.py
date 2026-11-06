import pandas as pd
import numpy as np
import cupy as cp
import pickle
import os

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import cupy as cp
import os

import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings
import sys
sys.path.append('../')
from convNet import *
from cpaCupy import *

def loadModel(path):
    model = Net().to(device).half().eval()
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def load_filters_corr_stats(folder_path, numFilters):
    """
    Loads 3D numpy arrays from all .pkl files in the specified folder and stacks them into a 4D numpy array.
    
    Parameters:
        folder_path (str): The path to the folder containing .pkl files.
    
    Returns:
        np.ndarray: A 4D numpy array where the first dimension indexes the individual filters.
    """
    filters = []
    
    # Iterate over all files in the folder
    for i in range(numFilters):
        file_path = os.path.join(folder_path, f"filter_{i}.pkl")
        with open(file_path, "rb") as file:
            # Load the 3D numpy array from the pickle file
            filter_array = pickle.load(file)
            filters.append(filter_array)
    
    # Stack all 3D arrays into one 4D numpy array along a new axis
    big_array = np.stack(filters, axis=0)
    return big_array

def layerWeightsFunc(name, model):
    for layer_name, layer in model.named_modules():
        if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)) and layer_name == name:
            numFilters = layer.weight.data.shape[0]
            layerW = (layer.weight.data.flatten().cpu().numpy().astype(np.float16).reshape(numFilters, -1))
    return layerW

def compare_hfm_kernel(conv_inputs, layer_name, model_a, model_b,
                       output_channel, input_idx, zero_val=0.0):
    """
    Compare HFM kernel outputs for two models on a given conv-layer channel and input index.

    Parameters
    ----------
    conv_inputs : array‐like
        Input activations, shape (N, C_in, ...). We’ll index [:, input_idx].
    layer_name : str
        Name of the convolutional layer (e.g. 'conv2').
    model_a, model_b : any
        Model objects accepted by layerWeightsFunc.
    output_channel : int
        Which output channel to pull weights from (e.g. 30).
    input_idx : int
        Which input feature‐map index to use (e.g. 147).
    zero_val : float, optional
        The zero‐point to pass to hfm_kernel (default 0.0).

    Returns
    -------
    bool
        True if the two hfm_kernel outputs are exactly equal.
    """
    # gather inputs and weights
    inp = cp.asarray(conv_inputs[:, input_idx])
    w_a = cp.float16(layerWeightsFunc(layer_name, model_a)[output_channel, input_idx])
    w_b = cp.float16(layerWeightsFunc(layer_name, model_b)[output_channel, input_idx])
    # compute HFM outputs
    out_a = hfm_kernel(inp, w_a, cp.float16(zero_val))
    out_b = hfm_kernel(inp, w_b, cp.float16(zero_val))
    # compare
    return np.array_equal(out_a, out_b)  # Return both for debugging purposes

def compute_prevSum_HW(inputs, weights, leakage_model):
  prevSums = []
  tracesHW = []
  prevSum = cp.float16(0.0)
  for i, weight in enumerate(weights):
    intermediate_sum = hfm_kernel(cp.asarray(inputs[:, i]), cp.float16(weight), prevSum)
    sumHW = popc_kernel(intermediate_sum) if leakage_model == 'HW' else hd_kernel(prevSum, intermediate_sum)
    prevSums.append(intermediate_sum.get())
    tracesHW.append(sumHW.get())
    prevSum = intermediate_sum
  prevSums = (np.array(prevSums).T)
  tracesHW = (np.array(tracesHW).T)
  return prevSums, tracesHW

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

def getLayers():
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'fc1', 'fc2']
    return layers

import torch
import torch.nn as nn
import numpy as np

# Hook function to store the intermediate outputs
intermediateActivations = {}

def hookFn(module, input, output):
    """This function will be called during forward pass to capture input and output."""
    intermediateActivations[module] = (input[0].detach().cpu(), output.detach().cpu())

def setupHooks(model: nn.Module) -> list:
    """
    Registers forward hooks for all convolutional and linear layers in the model.

    Args:
        model (nn.Module): The neural network model.

    Returns:
        list: A list of hook handles.
    """
    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            hook = layer.register_forward_hook(hookFn)
            hooks.append(hook)
    return hooks

def removeHooks(hooks: list) -> None:
    """
    Removes all registered hooks.

    Args:
        hooks (list): List of hook handles.
    """
    for hook in hooks:
        hook.remove()

def calc_layer_Input_output_in_batches(model, inputTensor, layer, batch_size=10_000):
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
    layer_outputs = []
    model.eval()
    total_samples = inputTensor.size(0)
    first_n = min(total_samples, 10_000)

    # Process first 10k samples in batches of 500
    #print(f"Processing first {first_n} samples in batches of 500 for layer {layer}")
    for batch in torch.split(inputTensor[:first_n], 500, dim=0):
        with torch.no_grad():
            batch = batch.to(device).half()
            _ = model(batch)
        layer_input, layer_output = intermediateActivations[layer]
        layer_inputs.append(layer_input.cpu())
        layer_outputs.append(layer_output.cpu())
        del _

    # Process remaining samples in batches of 10k
    if total_samples > first_n:
        #print(f"Processing remaining {total_samples - first_n} samples in batches of 10000 for layer {layer}")
        for batch in torch.split(inputTensor[first_n:], batch_size, dim=0):
            with torch.no_grad():
                batch = batch.to(device).half()
                _ = model(batch)
            layer_input, layer_output = intermediateActivations[layer]
            layer_inputs.append(layer_input.cpu())
            layer_outputs.append(layer_output.cpu())
            del _

    # Concatenate and return all collected activations
    return np.array(torch.cat(layer_inputs, dim=0)), np.array(torch.cat(layer_outputs, dim=0))

# Function to calculate accuracy on the test dataset
def calculate_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for batch_isx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.half())
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

normalize = lambda h: (np.float64(h) - np.float64(h).mean(keepdims=True)) / np.float64(h).std(ddof=1)

comp_corr = lambda x, y: np.dot(normalize(x), normalize(y)) / (x.shape[0]-1)

import cupy as cp

def compute_hyp(candidate, inputs, prev_state=None, leakage_model='HW'):
    """
    Returns hypothesis vector for the chosen leakage model.
      - HW: popcount(current_state)
      - HD: popcount(xor(current_state, prev_state))

    candidate   : array-like, weights/params for your hfm_kernel
    inputs      : array-like, inputs for your hfm_kernel
    prev_state  : array-like or None; required for HD (previous state V_prev)
    leakage_model: 'HW' or 'HD'
    """
    # --- push to GPU (match your hfm_kernel dtypes) ---
    cand_gpu = cp.asarray(candidate, dtype=cp.float16)
    in_gpu   = cp.asarray(inputs,    dtype=cp.float16)
    prev_gpu = None if prev_state is None else cp.asarray(prev_state, dtype=cp.float16)

    # --- current intermediate state on GPU ---
    # If your hfm_kernel expects a scalar when there's no previous state:
    prev_arg = cp.float16(0.0) if prev_gpu is None else prev_gpu
    V_gpu = hfm_kernel(cand_gpu, in_gpu, prev_arg)

    # --- dispatch by leakage model ---
    lm = leakage_model.upper()
    if lm == 'HW':
        # Hamming Weight: popcount of current state
        hyp_gpu = popc_kernel(V_gpu)
    elif lm == 'HD':
        if prev_gpu is None:
            raise ValueError("HD leakage model requires prev_state (previous intermediate state).")
        # Hamming Distance: popcount of bit-difference between current and previous states
        # If your hd_kernel already returns the XOR result, keep popc_kernel here.
        # If hd_kernel already returns a population count, drop popc_kernel.
        hyp_gpu = (hd_kernel(V_gpu, prev_gpu))
    else:
        raise ValueError(f"Unknown leakage_model: {leakage_model!r}. Use 'HW' or 'HD'.")

    return hyp_gpu.get()

import torch
from typing import List, Dict, Optional

# Default CIFAR-10 names
CIFAR10_CLASSES = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

def per_class_accuracy(model: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       class_names: Optional[List[str]] = None,
                       device: Optional[torch.device] = None,
                       use_model_half: bool = True) -> Dict[str, float]:
    """
    Compute per-class accuracy for a multiclass classifier.

    - model: a PyTorch model (expects raw logits or class scores).
    - dataloader: DataLoader over the test set (labels are integer class indices).
    - class_names: optional list of class names (length = num_classes). Defaults to CIFAR10_CLASSES.
    - device: optional torch.device. Defaults to CUDA if available else CPU.
    - use_model_half: if True, calls model.half() and converts inputs to half (user requested inpout.half()).

    Returns:
      dict mapping class_name -> accuracy_percent (0-100).
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES
    num_classes = len(class_names)

    model = model.to(device)
    if use_model_half:
        # convert model params / buffers to float16
        model = model.half()

    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long, device=device)
    total = torch.zeros(num_classes, dtype=torch.long, device=device)

    with torch.no_grad():
        for imgs, labels in dataloader:
            # move to device
            imgs = imgs.to(device)
            labels = labels.to(device)

            # user requested: input.half()
            imgs = imgs.half()

            # forward
            outputs = model(imgs)   # shape (B, num_classes)
            preds = outputs.argmax(dim=1)

            # accumulate class-wise counts (vectorized per-batch)
            for cls in range(num_classes):
                mask = (labels == cls)
                if mask.any():
                    total[cls] += int(mask.sum().item())
                    correct[cls] += int((preds[mask] == labels[mask]).sum().item())

    # build result dict (percentages)
    acc_by_class = {}
    for i, name in enumerate(class_names):
        if total[i] > 0:
            acc = 100.0 * float(correct[i].item()) / float(total[i].item())
        else:
            acc = float('nan')   # class absent in loader
        acc_by_class[name] = acc

    return acc_by_class