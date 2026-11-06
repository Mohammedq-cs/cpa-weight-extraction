import numpy as np
import torch
import itertools
import time
import tqdm
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from cuda import cuda, cudart
import pickle
import torch.nn as nn
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def generate_weight_candidates(bit_length=16, upper_bound=5.0, lower_bound=-5.0):
    """
    Generates a list of weight candidates that are within the specified bounds.

    Parameters:
    ----------
    bit_length : int, optional
        The bit length for the binary representation of the candidates. Default is 16.
    
    upper_bound : float, optional
        The upper bound for the weight values. Default is 5.0.
    
    lower_bound : float, optional
        The lower bound for the weight values. Default is -5.0.

    Returns:
    -------
    weight_candidates : numpy.ndarray
        A numpy array of half-precision floating-point weight candidates within the specified bounds.
    """
    
    # Generate all possible binary combinations of the specified bit length
    candidates = list(itertools.product('01', repeat=bit_length))
    candidates = [''.join(comb) for comb in candidates]
    
    weight_candidates = []
    
    # Convert binary strings to half-precision floats and filter by the bounds
    for candidate in candidates:
        actual_weight_bytes = bytes(int(candidate, 2).to_bytes(2, byteorder='little'))
        half_weight = np.frombuffer(actual_weight_bytes, dtype=np.float16)
        if np.float16(upper_bound) >= half_weight >= np.float16(lower_bound):
            weight_candidates.append(half_weight)
    
    weight_candidates = np.array(weight_candidates, dtype=np.float16)
    return weight_candidates

def generate_system_noise_array(simulated_traces, noise_level):
    """
    Generate a combined array with random columns and add normal noise.

    Parameters:
    ----------
    simulated_traces : numpy.ndarray
        The simulated power traces. The function checks if simulated_traces[:, 1] has shape (n, 1).
        
    noise_level : float
        The standard deviation of the normal noise to be added.

    Returns:
    -------
    combined_array : numpy.ndarray
        The combined array with noise added.
    """
    
     # Check if simulated_traces is 1-dimensional
    if simulated_traces.ndim == 1:
        # If 1-dimensional, expand dims to make it (n, 1)
        trace_column = np.expand_dims(simulated_traces, axis=1)
    else:
        # If 2-dimensional, ensure the shape is (n, 1)
        if simulated_traces.shape[1] != 1:
            raise ValueError("simulated_traces must have a shape (n, 1) for this operation.")
        trace_column = simulated_traces

    # Generate random columns
    random_columns = np.random.randint(0, 17, size=(trace_column.shape[0], 3)).astype(np.float16)

    # Concatenate the original trace column with the random columns
    combined_array = np.hstack((trace_column, random_columns))

    # Add normal noise with the specified noise level
    noise = np.random.normal(0, noise_level, combined_array.shape).astype(np.float16)
    combined_array += noise

    # Shuffle the columns
    np.random.shuffle(combined_array.T)

    return combined_array

def check_gpu_memory():
    # Get free and total memory
    err, free_memory, total_memory = cudart.cudaMemGetInfo()

    # Convert memory from bytes to GB
    free_memory_GB = free_memory / (1024 * 1024 * 1024)
    total_memory_GB = total_memory / (1024 * 1024 * 1024)

    # Print memory in GB with two decimal places
    print(f"Free GPU Memory: {free_memory_GB:.2f} GB")
    print(f"Total GPU Memory: {total_memory_GB:.2f} GB")
    
def calcAccBitwise(layerWeights, extractedWeights):
    layerW = layerWeights.flatten().astype(np.float16)
    extractedWeights = extractedWeights.flatten().astype(np.float16)
    
    # View the 16-bit representation as unsigned 16-bit integers
    layerW_bits = layerW.view(np.uint16)
    extractedWeights_bits = extractedWeights.view(np.uint16)
    
    # Convert the 16-bit integer representation to binary strings
    layerW_binary = np.array([np.binary_repr(val, width=16) for val in layerW_bits])
    extractedWeights_binary = np.array([np.binary_repr(val, width=16) for val in extractedWeights_bits])
    
    # Count the exact matches in binary representation
    correct = np.sum(layerW_binary == extractedWeights_binary)

    return correct, layerW.size

def save_model_weights(model, save_directory):
    """
    Save the model weights to a file.
    """
      # create the folder if it doesn't exist
    for layer_name, layer in model.named_modules():
        if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
            layer_weights = layerWeightsFunc(layer_name, model)
            os.makedirs(f'{save_directory}/{layer_name}', exist_ok=True)
            for i in range(layer_weights.shape[0]):
                pickle.dump(layer_weights[i], open(f'{save_directory}/{layer_name}/filter_{i}.pkl', 'wb'))

def layerWeightsFunc(name, model):
    for layer_name, layer in model.named_modules():
        if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)) and layer_name == name:
            numFilters = layer.weight.data.shape[0]
            layerW = (layer.weight.data.flatten().cpu().numpy().astype(np.float16).reshape(numFilters, -1))
    return layerW

def print_top_k_correlations(corrRes, candidates, j, k=5, printF=True):
    """
    Finds, prints, and returns the top k weights sorted by:
      1. Highest max correlation (desc)
      2. Largest weight value to break ties (desc)
    """
    # 1. Compute max absolute correlation per weight
    max_corr = np.max(np.abs(corrRes), axis=1)
    # 2. Extract the primary and secondary keys
    #    - primary: descending corr → use -max_corr
    #    - secondary: descending weight → use -candidates[:,0]
    primary   = -max_corr
    secondary = -candidates[:, 0]

    # 3. Indirectly sort with lexsort: last key is primary
    sorted_inds = np.lexsort((secondary, primary))

    # 4. Select the top k indices (these are in descending order already)
    top_k_inds = sorted_inds[:k]

    # 5. Fetch the weights and correlations
    top_k_weights = candidates[top_k_inds]
    top_k_corrs   = max_corr[top_k_inds]

    if printF:
        print(f"{j}, Top {k} weights with their maximum correlation values:")
        for w, c in zip(top_k_weights, top_k_corrs):
            print(f"  Weight: {w[0]}, Max Correlation: {c}")

    return top_k_weights, top_k_corrs


def print_top_k_correlations_old(corrRes, candidates, j, k=5, printF=True):
    """
    Finds, prints, and returns the top k weights with the highest maximum correlation values.

    Args:
        corrRes (numpy.ndarray): 2D array of correlation results where each row corresponds to a weight.
        candidates (numpy.ndarray): 2D array of candidate weights corresponding to rows in corrRes.
        k (int): The number of top correlations to print and return.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - top_k_sorted_weights: The top k weights with the highest maximum correlation values.
            - top_k_sorted_corr_values: The corresponding correlation values.
    """
    # Calculate maximum correlation per weight
    max_corr_per_weight = np.max(corrRes, axis=1)

    # Find the indices of the top k weights with the highest maximum correlations
    top_k_indices = np.argpartition(max_corr_per_weight, -k)[-k:]

    # Extract the corresponding maximum correlation values and weights
    top_k_corr_values = max_corr_per_weight[top_k_indices]
    top_k_weights = candidates[top_k_indices]

    # Sort the top k weights based on their maximum correlation in descending order
    sorted_order = np.argsort(top_k_corr_values,)[::-1]
    top_k_sorted_weights = top_k_weights[sorted_order]
    top_k_sorted_corr_values = top_k_corr_values[sorted_order]

    if printF:
        # Print the top k weights with their corresponding maximum correlation values
        print(f"{j}, Top {k} weights with their maximum correlation values:")
        for idx in range(len(top_k_sorted_weights)):
            weight = top_k_sorted_weights[idx]
            corr = top_k_sorted_corr_values[idx]
            print(f"Weight: {weight[0]}, Max Correlation: {corr}")

    # Return the top k weights and their maximum correlation values
    return top_k_sorted_weights, top_k_sorted_corr_values

def print_difference_histogram(extractedWeights, layerWeights, bins=20):
    # Flatten and convert weights to float32, if needed
    extractedWeights = extractedWeights.flatten().astype(np.float32)
    layerWeights = layerWeights.flatten().astype(np.float32)
    
    # Calculate the difference
    difference = extractedWeights - layerWeights

    # Compute the histogram of the difference
    counts, bin_edges = np.histogram(difference, bins=bins)

    # Print counts and bins
    print("Counts in each bin:", counts)
    print("Bin edges:", bin_edges)

def print_index_histogram(layerWeights, extractedWeights):
    # Reshape layerWeights to (layerWeights.shape[0], -1)
    reshaped_layerWeights = layerWeights.reshape(layerWeights.shape[0], -1).astype(np.float32)
    
    # Reshape extractedWeights to the same shape as reshaped_layerWeights
    reshaped_extractedWeights = extractedWeights.reshape(reshaped_layerWeights.shape).astype(np.float32)
    
    # Find indices along dim 1 where the absolute difference > 0.5
    diff_indices = np.abs(reshaped_extractedWeights - reshaped_layerWeights) > 0.04
    
     # Convert to binary (1 where difference > 0.5, else 0)
    binary_array = diff_indices.astype(int)
    
    # Sum along dimension 0
    summed_array = np.sum(binary_array, axis=0)
    
    print("Summed array along dimension 0:", summed_array)
    print(len(summed_array))
    return summed_array

def inputWiseAcc(ogInps, modifiedModelInps, NumTraces, name):
    difference = np.array(modifiedModelInps).astype(np.float16) - np.array(ogInps).astype(np.float16)
    randomInps1 = modifiedModelInps.view(NumTraces, -1).numpy().astype(np.float16)
    randomInps2 = ogInps.view(NumTraces, -1).numpy().astype(np.float16)
    traces, inputSize = randomInps2.shape
    zero_count = np.sum(difference == 0)
    total_count = difference.size
    #plot_error_histogram(randomInps1 != randomInps2, f'Error Counts per Trace with input {inputSize} with {NumTraces} Traces {name}', f'allInpsHistogram{name}')

    return (zero_count / total_count)
    
