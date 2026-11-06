import pickle
import numpy as np
from cpaCupy import *
from utils import *
import time
from ipdb import set_trace
from attack import *
import os
import argparse
import pickle
import pandas as pd

parser = argparse.ArgumentParser(description='Process simulation parameters.')

# Add arguments
parser.add_argument('--baseLineInputDir', type=str, required=True,
                        help='Path to the simulated traces file.')
parser.add_argument('--simulatedInputsDir', type=str, required=True,
                        help='Path to the input directory.')
parser.add_argument('--outputDir', type=str, required=True,
                        help='Path to the output directory.')
parser.add_argument('--corrStatSavePath', type=str, required=True,
                        help='Path to the output directory.')                        
parser.add_argument('--layerName', type=str, required=True,
                        help='Name of the layer.')
parser.add_argument('--filterIdx', type=int, required=True,
                        help='Index of the filter.')
parser.add_argument('--leakage_model', type=str, required=True,
                    choices=['HW', 'HD'])
parser.add_argument('--seed', type=int, default=8,
                    help='Index of the filter.')
parser.add_argument('--NoiseLevel', type=float, default=0.0,
                    help='Index of the filter.')  
                  
# Parse the arguments
args = parser.parse_args()

layerName = args.layerName
filter_idx = args.filterIdx
simulatedInpPath = args.simulatedInputsDir
savePath = args.outputDir
corrStatSavePath = args.corrStatSavePath
baseLineInpsPath = args.baseLineInputDir
seed = args.seed
noise_level = args.NoiseLevel
leakage_model = args.leakage_model

print(f"Using seed: {seed}, Noise Level: {noise_level}, Leakage Model: {leakage_model}")
np.random.seed(seed)

# Load the pickle file
with open(os.path.join(simulatedInpPath, f'{layerName}.pkl'), 'rb') as f:
    simulated_inputs = pickle.load(f)

# Load the pickle file
with open(os.path.join(baseLineInpsPath, f'{layerName}.pkl'), 'rb') as f:
    baseline_inputs = pickle.load(f)

baseline_inputs = baseline_inputs[:simulated_inputs.shape[0]]
# Calculate the Leakage Value based on correct values.
# Load the pickle file
with open(os.path.join('./model_weights', f'{layerName}/filter_{filter_idx}.pkl'), 'rb') as f:
    filter_weights = pickle.load(f)

# The extraction process starts from here

threshold = 0.2 if noise_level == 0.0 else 0.018
filterLeakage = compute_filter_hw_in_batches(filter_weights, baseline_inputs, batch_size=30_000, leakage_model=leakage_model)

numWeights = simulated_inputs.shape[1]

extractedWeights = []
correlationStatistics = []    
# Generate weight candidates (assumed to be part of the function's context)
weight_candidates = generate_weight_candidates()
prevSum = np.zeros_like(simulated_inputs[:, 0])

t1 = time.time()
for i in range(0, numWeights):
    #Generate Noise
    trace_HW_column = filterLeakage[:, i]
    simulatedTraces = generate_system_noise_array(trace_HW_column, noise_level).astype(np.float32)
    
    input_column = simulated_inputs[:, i]
    
    final_corr = CPA(simulatedTraces, input_column, weight_candidates, prevSum, leakage_model=leakage_model, batch_size= 2000, dtype=np.float32)
    top_k_sorted_weights, top_k_sorted_corr_values = print_top_k_correlations((final_corr), weight_candidates, i, k=100, printF=False)
    
    weight_candidates_filtered = np.array(top_k_sorted_weights)

    final_corr = CPA(simulatedTraces, input_column, weight_candidates_filtered, prevSum, leakage_model=leakage_model, batch_size= 2000, dtype=np.float64)
    top_k_sorted_weights, top_k_sorted_corr_values = print_top_k_correlations((final_corr), weight_candidates_filtered, i, k=100, printF=False)
    correlationStatistics.append(np.column_stack((top_k_sorted_weights, top_k_sorted_corr_values)))
    if leakage_model == 'HD' and top_k_sorted_corr_values[0] <= threshold:
        print(f"most likely correlated to small value {i}. Appending zero weight.")
        extractedWeights.append(np.float16(0.0))
        prevSum = hfm_kernel(cp.asarray(np.ascontiguousarray(simulated_inputs[:, i])), cp.float16(0.0), cp.asarray(prevSum)).get().astype(np.float16)
        prevSum = np.ascontiguousarray(prevSum.astype(np.float16))
        continue

    corr_max = np.max((final_corr), axis=1)

    # 2. Find all indices where this maximum is attained
    max_inds = np.where(corr_max == corr_max.max())[0]

    # 3. Among those tied indices, pick the one with the largest weight value
    #    (assume weight value is at column 0 of weight_candidates_filtered)
    best_idx = max_inds[np.argmax(weight_candidates_filtered[max_inds, 0])]

    # 4. Extract the weight
    extractedWeight = weight_candidates_filtered[best_idx, 0]

    extractedWeights.append(extractedWeight)
    prevSum = hfm_kernel(cp.asarray(np.ascontiguousarray(input_column)), cp.float16(extractedWeight), cp.asarray(prevSum)).get().astype(np.float16)
    prevSum = np.ascontiguousarray(prevSum.astype(np.float16))



extractedWeights = np.array(extractedWeights)
t2 = time.time()

print(f"time taken = {t2 - t1} Sec")

print(extractedWeights)
# Define the directory path separately
# Ensure the directory exists

os.makedirs(os.path.join(savePath, f'{layerName}'), exist_ok=True)

out_save_path = os.path.join(savePath, f'{layerName}/filter_{filter_idx}.pkl')

# Save using pickle
with open(out_save_path, 'wb') as f:
    pickle.dump(extractedWeights, f)

#save the Corr statistics
#wrong should be filter index
os.makedirs(os.path.join(corrStatSavePath, f'{layerName}'), exist_ok=True)

pickle.dump(np.array(correlationStatistics), open(os.path.join(corrStatSavePath, f'{layerName}/filter_{filter_idx}.pkl'), 'wb'))
