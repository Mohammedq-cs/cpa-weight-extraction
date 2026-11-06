import numpy as np
import time
from utils import *
from attack import *
from modelUtils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from ipdb import set_trace
import pickle
import os
from slurmManager import *
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model_inputs(name, numTraces, extractedModelInps, base_directory="./simInps"):
    """
    Save the extracted model inputs to a pickle file.
    
    Args:
        name (str): The name used for the filename.
        numTraces (int): Number of traces used to form the directory name.
        extractedModelInps (Any): Data to be saved.
        base_directory (str): Base directory path where inputs are saved.
    """
    # Define the directory and file paths
    directory = os.path.join(base_directory, str(numTraces))
    filename = f'{name}.pkl'
    file_path = os.path.join(directory, filename)
    
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    sys.stdout.flush()
    
    # Save using pickle
    with open(file_path, 'wb') as f:
        pickle.dump(extractedModelInps, f)

def submit_slurm_jobs(name, numTraces, numFilters, batch_size=64, num_parallel=8, noise_level=0, leakage_model='HW', exp_name='randomSampleFromCifarCinic', sub_folder=None):
    """
    Submit SLURM jobs in batches and print the time taken.
    
    Args:
        name (str): Job identifier used in the submission.
        numTraces (int): Number of traces used for job submission.
        numFilters (int): Total number of filters (jobs to submit).
        batch_size (int): Number of jobs to submit in each batch.
        num_parallel (int): Number of GPUs in parallel.
        
    Note:
        Assumes the existence of a submit_batch function.
    """
    startTime = time.time()
    for start in range(0, numFilters, batch_size):
        end = min(start + batch_size - 1, numFilters - 1)
        submit_batch(start, end, num_parallel, name, numTraces, noise_level, leakage_model, exp_name=exp_name, sub_folder=sub_folder)
    endTime = time.time()
    print(f"Time Taken = {((endTime - startTime)/60):.2f} mins\n")

def load_extracted_filters(name, numTraces, numFilters, base_directory="./output"):
    """
    Load extracted filters from pickle files and return them as a numpy array.
    
    Args:
        name (str): The name of the subdirectory containing filter files.
        numTraces (int): Number of traces, used as a directory name.
        numFilters (int): Number of filter files to load.
        base_directory (str): Base directory path where outputs are stored.
        
    Returns:
        numpy.ndarray: Array containing all loaded filter data.
    """
    extractedFilters = []
    filters_directory = os.path.join(base_directory, str(numTraces), name)
    for filter_idx in range(numFilters):
        file_path = os.path.join(filters_directory, f'filter_{filter_idx}.pkl')
        with open(file_path, 'rb') as f:
            filter_data = pickle.load(f)
        extractedFilters.append(filter_data)
    
    return np.array(extractedFilters)

def evaluate_extraction(layer_weights, extractedFilters):
    """
    Evaluate the extraction by comparing the original layer weights with the extracted filters.

    Args:
        layer: A layer object with a weight attribute (e.g., a PyTorch layer).
        extractedFilters (numpy.ndarray): Array containing the extracted filters.
        
    Returns:
        float: The calculated bitwise accuracy.
        
    Note:
        This function requires the existence of the following functions:
            - calcAccBitwise: to compute the bitwise accuracy and support.
            - print_difference_histogram: to visualize differences between actual weights and extracted filters.
    """    
    # Calculate bitwise accuracy components.
    layerAccBW, layerSuBW = calcAccBitwise(layer_weights, extractedFilters)
    
    # Print histogram of differences.
    print_difference_histogram(layer_weights, extractedFilters)
    
    # Calculate and print accuracy.
    accuracy = layerAccBW / layerSuBW if layerSuBW != 0 else 0
    print(f"Layer Accuracy Bit Wise = {accuracy:.4f}")
    
    return accuracy

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

def weightsRetrievalSimulation(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    numTraces: int,
    noise_level: int,
    leakage_model: str,
    exp_name: str,
    sub_folder: str
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
    startTimeGlobal = time.time()
    accRes = []
    model.eval()
    startTimeGlobal = time.time()

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
            base_directory_p= f'./N{noise_level}/{exp_name}output/{sub_folder}' if sub_folder else f'./N{noise_level}/{exp_name}output'
            filters_directory = os.path.join(base_directory_p, str(numTraces), name)
            
            if not os.path.isdir(filters_directory):
                layerInput = calc_layerInp_in_batches(model, data_loader, layer)
                if isinstance(layer, nn.Conv2d):
                
                    in_ch  = layer.in_channels
                    kH, kW = layer.kernel_size
                    numWeights = in_ch * kH * kW

                    extractedModelInps = (
                        layerInput[:, :in_ch, :kH, :kW]
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                        .reshape((-1, numWeights))
                    )

                elif isinstance(layer, nn.Linear):
                    numWeights = layer.in_features

                    extractedModelInps = (
                        layerInput[:, :numWeights]
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )
                save_model_inputs(name, numTraces, extractedModelInps, base_directory=f"./simInps/N{noise_level}/{exp_name}/{sub_folder}" if sub_folder else f"./simInps/N{noise_level}/{exp_name}")
                submit_slurm_jobs(name, numTraces, numFilters, batch_size=32, num_parallel=64, noise_level=noise_level, leakage_model=leakage_model, exp_name=exp_name, sub_folder=sub_folder)
                del extractedModelInps

            extractedFilters = load_extracted_filters(name, numTraces, numFilters, base_directory= f'./N{noise_level}/{exp_name}output/{sub_folder}' if sub_folder else f'./N{noise_level}/{exp_name}output')
            baseLineFilters = load_extracted_filters(name, 'model_weights', numFilters, base_directory= f'./')

            # evaluate the extraction
            accuracy = evaluate_extraction(baseLineFilters, extractedFilters)
            accRes.append(accuracy)
            
            updateLayerWeights(layer, extractedFilters)
            sys.stdout.flush()

    print(f"Total Time Taken = {((time.time() - startTimeGlobal)/60):.2f} mins")
    return accRes, model

def runWeightsRetrievalSimulation(model, data_loader, numTraces, noise_level = 0, leakage_model = 'HW', exp_name = None, sub_folder = None):
    hooks = setupHooks(model)

    accLayers = weightsRetrievalSimulation(model, data_loader, numTraces, noise_level, leakage_model=leakage_model, exp_name=exp_name, sub_folder=sub_folder)

    removeHooks(hooks)

    return accLayers
