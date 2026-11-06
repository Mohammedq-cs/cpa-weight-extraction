import numpy as np
from cpaCupy import *

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

def getFilterHW(layer, simulationInps, numFilters):
  FilterHW = []
  for i in range(numFilters):
        layerW = layer.weight.data[i].flatten().cpu().numpy().astype(np.float16)
        partialSum, partialSumHW = compute_prevSum_HW(simulationInps, layerW)
        FilterHW.append(partialSumHW)
  return FilterHW

def compute_filter_hw_in_batches(filter_weights, realInps, batch_size, leakage_model):
    """
    Compute the filterHW array in batches from realInps.

    Args:
        filter_weights (numpy.ndarray): The weights of the filters.
        realInps (iterable): The full set of real inputs.
        batch_size (int): Size of the batch to process at a time.

    Returns:
        numpy.ndarray: The concatenated filterHW array computed over all batches.
    """
    filter_hw_batches = []
    total_inputs = len(realInps)
    
    for start in range(0, total_inputs, batch_size):
        end = start + batch_size
        batch_inputs = realInps[start:end]
        # Compute filterHW for the current batch.
        batch_filter_prev_sums, batch_filter_hw = compute_prevSum_HW(batch_inputs, filter_weights, leakage_model)
        # Ensure that the batch result is a NumPy array
        batch_filter_hw = np.array(batch_filter_hw)
        #print(f"Batch {start} to {end} has shape: {batch_filter_hw.shape}")
        filter_hw_batches.append(batch_filter_hw)
    
    # Concatenate all the batch results along the first axis.
    filter_hw = np.concatenate(filter_hw_batches, axis=0)
    print(f"FilterHW shape: {filter_hw.shape}")
    return filter_hw
        