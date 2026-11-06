import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd

from cpaCupy import *
from utils import *
from attack import *
from ipdb import set_trace


parser = argparse.ArgumentParser(description='Process simulation parameters.')

parser.add_argument('--baseLineInputDir', type=str, required=True,
                    help='Path to baseline inputs directory.')
parser.add_argument('--simulatedInputsDir', type=str, required=True,
                    help='Path to simulated inputs directory.')
parser.add_argument('--outputDir', type=str, required=True,
                    help='Path to extracted weights output directory.')
parser.add_argument('--corrStatSavePath', type=str, required=True,
                    help='Path to correlation statistics output directory.')
parser.add_argument('--layerName', type=str, required=True,
                    help='Name of the layer.')
parser.add_argument('--filterIdx', type=int, required=True,
                    help='Index of the filter.')
parser.add_argument('--leakage_model', type=str, required=True,
                    choices=['HW', 'HD'])
parser.add_argument('--seed', type=int, default=8,
                    help='Random seed.')
parser.add_argument('--NoiseLevel', type=float, default=0.0,
                    help='Noise level.')
parser.add_argument('--chunkSize', type=int, default=40,
                    help='Number of weights to process before checkpointing.')
parser.add_argument('--weightsDir', type=str, default='./model_weights',
                    help='Directory holding ground-truth per-filter weights.')

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
chunk_size = args.chunkSize
weightsDir = args.weightsDir

print(f"Using seed: {seed}, Noise Level: {noise_level}, Leakage Model: {leakage_model}, Chunk Size: {chunk_size}")
np.random.seed(seed)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_layer_inputs_with_patches(input_dir, layer_name):
    """
    Supports:
      1) {layer_name}.pkl
      2) {layer_name}_patch0.pkl, {layer_name}_patch1.pkl, ...
      3) {layer_name}.patch0.pkl, {layer_name}.patch1.pkl, ...

    Returns a list of arrays.
    """
    base_file = os.path.join(input_dir, f"{layer_name}.pkl")
    if os.path.exists(base_file):
        return [load_pickle(base_file)], "single"

    inputs_list = []
    patch_idx = 0

    while True:
        patch_file_us = os.path.join(input_dir, f"{layer_name}_patch{patch_idx}.pkl")
        patch_file_dot = os.path.join(input_dir, f"{layer_name}.patch{patch_idx}.pkl")

        if os.path.exists(patch_file_us):
            inputs_list.append(load_pickle(patch_file_us))
            patch_idx += 1
        elif os.path.exists(patch_file_dot):
            inputs_list.append(load_pickle(patch_file_dot))
            patch_idx += 1
        else:
            if patch_idx == 0:
                raise FileNotFoundError(
                    f"Could not find {base_file}, {layer_name}_patch0.pkl, or {layer_name}.patch0.pkl in {input_dir}"
                )
            break

    return inputs_list, "patched"


def merge_input_lists(simulated_inputs_list, baseline_inputs_list, layer_name):
    """
    Truncate each simulated/baseline patch pair to same trace count,
    then concatenate all patches into one big array.
    """
    if len(simulated_inputs_list) != len(baseline_inputs_list):
        raise ValueError(
            f"{layer_name}: number of simulated input files ({len(simulated_inputs_list)}) "
            f"does not match number of baseline input files ({len(baseline_inputs_list)})"
        )

    aligned_simulated = []
    aligned_baseline = []

    for sim_arr, base_arr in zip(simulated_inputs_list, baseline_inputs_list):
        n = min(sim_arr.shape[0], base_arr.shape[0])
        aligned_simulated.append(sim_arr[:n])
        aligned_baseline.append(base_arr[:n])

    simulated_inputs = np.concatenate(aligned_simulated, axis=0)
    baseline_inputs = np.concatenate(aligned_baseline, axis=0)

    return simulated_inputs, baseline_inputs


def flatten_filter_weights(filter_weights):
    return np.asarray(filter_weights).reshape(-1).astype(np.float16)


def checkpoint_path(output_dir, layer_name, filter_idx):
    return os.path.join(output_dir, layer_name, f'filter_{filter_idx}_checkpoint.pkl')


def final_weights_path(output_dir, layer_name, filter_idx):
    return os.path.join(output_dir, layer_name, f'filter_{filter_idx}.pkl')


def final_corr_path(corr_dir, layer_name, filter_idx):
    return os.path.join(corr_dir, layer_name, f'filter_{filter_idx}.pkl')


def save_checkpoint(path, next_weight_idx, extractedWeights, correlationStatistics,
                    attack_prevSum, baseline_prevSum, rng_state):
    """
    Save only the newest checkpoint.
    Writes to temp file first, removes previous checkpoint, then replaces.
    """
    state = {
        'next_weight_idx': next_weight_idx,
        'extractedWeights': np.array(extractedWeights, dtype=np.float16),
        'correlationStatistics': correlationStatistics,
        'attack_prevSum': attack_prevSum,
        'baseline_prevSum': baseline_prevSum,
        'rng_state': rng_state,
    }

    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(tmp_path, 'wb') as f:
        pickle.dump(state, f)

    if os.path.exists(path):
        os.remove(path)

    os.replace(tmp_path, path)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    return load_pickle(path)


def calc_hw_array(arr):
    """
    Hamming weight elementwise for float16 viewed as uint16.
    """
    arr_u16 = arr.view(np.uint16)
    return np.unpackbits(arr_u16.view(np.uint8)).reshape(arr_u16.shape + (16,)).sum(axis=-1).astype(np.float16)


def calc_hd_array(prev_arr, cur_arr):
    """
    Hamming distance elementwise for float16 viewed as uint16.
    """
    prev_u16 = prev_arr.view(np.uint16)
    cur_u16 = cur_arr.view(np.uint16)
    xor_u16 = np.bitwise_xor(prev_u16, cur_u16)
    return np.unpackbits(xor_u16.view(np.uint8)).reshape(xor_u16.shape + (16,)).sum(axis=-1).astype(np.float16)


def compute_filter_leakage_chunk_with_running_prefix(
    baseline_inputs_chunk,
    filter_weights_chunk,
    baseline_prevSum_start,
    leakage_model
):
    """
    Compute leakage for the current chunk while respecting the true prefix sum
    up to chunk_start. Also returns the updated true baseline prevSum after
    finishing the chunk, so the next chunk can continue from it.
    """
    num_traces, chunk_len = baseline_inputs_chunk.shape
    filterLeakage_chunk = np.zeros((num_traces, chunk_len), dtype=np.float16)

    running_prev = baseline_prevSum_start.copy()

    for local_idx in range(chunk_len):
        baseline_col = baseline_inputs_chunk[:, local_idx]
        true_w = np.float16(filter_weights_chunk[local_idx])

        cur_sum = hfm_kernel(
            cp.asarray(baseline_col),
            cp.float16(true_w),
            cp.asarray(running_prev)
        ).get().astype(np.float16)

        if leakage_model == 'HW':
            filterLeakage_chunk[:, local_idx] = calc_hw_array(cur_sum)
        elif leakage_model == 'HD':
            filterLeakage_chunk[:, local_idx] = calc_hd_array(running_prev, cur_sum)
        else:
            raise ValueError(f"Unsupported leakage_model: {leakage_model}")

        running_prev = cur_sum

    return filterLeakage_chunk, running_prev


# =========================
# Load and merge inputs
# =========================
simulated_inputs_list, sim_mode = load_layer_inputs_with_patches(simulatedInpPath, layerName)
baseline_inputs_list, base_mode = load_layer_inputs_with_patches(baseLineInpsPath, layerName)

simulated_inputs, baseline_inputs = merge_input_lists(
    simulated_inputs_list,
    baseline_inputs_list,
    layerName
)

print(f"Loaded simulated inputs mode: {sim_mode}, num patches: {len(simulated_inputs_list)}")
print(f"Loaded baseline inputs mode: {base_mode}, num patches: {len(baseline_inputs_list)}")
print(f"Merged simulated_inputs shape: {simulated_inputs.shape}")
print(f"Merged baseline_inputs shape: {baseline_inputs.shape}")

# =========================
# Load filter weights
# =========================
filter_weights = load_pickle(os.path.join(weightsDir, f'{layerName}/filter_{filter_idx}.pkl'))
filter_weights_flat = flatten_filter_weights(filter_weights)

total_cols = simulated_inputs.shape[1]

if baseline_inputs.shape[1] != total_cols:
    raise ValueError(
        f"baseline_inputs.shape[1] ({baseline_inputs.shape[1]}) != simulated_inputs.shape[1] ({total_cols})"
    )

per_filter_nw = filter_weights_flat.shape[0]

if total_cols % per_filter_nw != 0:
    raise ValueError(
        f"total input cols ({total_cols}) not divisible by per-filter weights ({per_filter_nw})"
    )

# groups == 1 for a normal conv / linear; == in_channels for a depthwise conv.
# For a grouped/depthwise conv the stored inputs hold every channel's kHxkW patch
# as [ch0(nw), ch1(nw), ...]; filter f reads only its own channel f.
groups = total_cols // per_filter_nw
if groups > 1:
    c0 = filter_idx * per_filter_nw
    simulated_inputs = simulated_inputs[:, c0:c0 + per_filter_nw]
    baseline_inputs = baseline_inputs[:, c0:c0 + per_filter_nw]

numWeights = per_filter_nw

threshold = 0.2 if noise_level == 0.0 else 0.018
weight_candidates = generate_weight_candidates()

# =========================
# Resume or initialize
# =========================
ckpt_path = checkpoint_path(savePath, layerName, filter_idx)
ckpt = load_checkpoint(ckpt_path)

if ckpt is not None:
    print(f"Loading checkpoint from {ckpt_path}")
    next_weight_idx = ckpt['next_weight_idx']
    extractedWeights = list(np.asarray(ckpt['extractedWeights']).astype(np.float16))
    correlationStatistics = ckpt['correlationStatistics']
    attack_prevSum = ckpt['attack_prevSum']
    baseline_prevSum = ckpt['baseline_prevSum']
    np.random.set_state(ckpt['rng_state'])
else:
    print("No checkpoint found. Starting from scratch.")
    next_weight_idx = 0
    extractedWeights = []
    correlationStatistics = []
    attack_prevSum = np.zeros(simulated_inputs.shape[0], dtype=np.float16)
    baseline_prevSum = np.zeros(baseline_inputs.shape[0], dtype=np.float16)

t1 = time.time()

# =========================
# Main loop: chunk_size weights at a time
# =========================
for chunk_start in range(next_weight_idx, numWeights, chunk_size):
    chunk_end = min(chunk_start + chunk_size, numWeights)
    print(f"\nProcessing weights {chunk_start} to {chunk_end - 1}")

    simulated_inputs_chunk = simulated_inputs[:, chunk_start:chunk_end]
    baseline_inputs_chunk = baseline_inputs[:, chunk_start:chunk_end]
    filter_weights_chunk = filter_weights_flat[chunk_start:chunk_end]

    # True leakage for this chunk, starting from the true prefix sum at chunk_start
    filterLeakage_chunk, baseline_prevSum_after_chunk = compute_filter_leakage_chunk_with_running_prefix(
        baseline_inputs_chunk=baseline_inputs_chunk,
        filter_weights_chunk=filter_weights_chunk,
        baseline_prevSum_start=baseline_prevSum,
        leakage_model=leakage_model
    )

    for i in range(chunk_start, chunk_end):
        local_col_idx = i - chunk_start

        trace_HW_column = filterLeakage_chunk[:, local_col_idx]
        simulatedTraces = generate_system_noise_array(trace_HW_column, noise_level).astype(np.float32)

        input_column = simulated_inputs_chunk[:, local_col_idx]

        final_corr = CPA(
            simulatedTraces,
            input_column,
            weight_candidates,
            attack_prevSum,
            leakage_model=leakage_model,
            batch_size=150,
            dtype=np.float32
        )

        top_k_sorted_weights, top_k_sorted_corr_values = print_top_k_correlations(
            final_corr,
            weight_candidates,
            i,
            k=100,
            printF=False
        )

        weight_candidates_filtered = np.array(top_k_sorted_weights)

        final_corr = CPA(
            simulatedTraces,
            input_column,
            weight_candidates_filtered,
            attack_prevSum,
            leakage_model=leakage_model,
            batch_size=150,
            dtype=np.float64
        )

        top_k_sorted_weights, top_k_sorted_corr_values = print_top_k_correlations(
            final_corr,
            weight_candidates_filtered,
            i,
            k=100,
            printF=False
        )

        correlationStatistics.append(np.column_stack((top_k_sorted_weights, top_k_sorted_corr_values)))

        if leakage_model == 'HD' and top_k_sorted_corr_values[0] <= threshold:
            print(f"weight {i}: most likely correlated to small value. Appending zero weight.")
            extractedWeight = np.float16(0.0)
            extractedWeights.append(extractedWeight)

            attack_prevSum = hfm_kernel(
                cp.asarray(input_column),
                cp.float16(extractedWeight),
                cp.asarray(attack_prevSum)
            ).get().astype(np.float16)
            continue

        corr_max = np.max(final_corr, axis=1)
        max_inds = np.where(corr_max == corr_max.max())[0]
        best_idx = max_inds[np.argmax(weight_candidates_filtered[max_inds, 0])]
        extractedWeight = np.float16(weight_candidates_filtered[best_idx, 0])

        extractedWeights.append(extractedWeight)

        attack_prevSum = hfm_kernel(
            cp.asarray(input_column),
            cp.float16(filter_weights_flat[i]),
            cp.asarray(attack_prevSum)
        ).get().astype(np.float16)

        #print(f"weight {i}: extracted {extractedWeight}")

    # Advance true baseline prefix sum to end of this chunk
    baseline_prevSum = baseline_prevSum_after_chunk

    save_checkpoint(
        path=ckpt_path,
        next_weight_idx=chunk_end,
        extractedWeights=extractedWeights,
        correlationStatistics=correlationStatistics,
        attack_prevSum=attack_prevSum,
        baseline_prevSum=baseline_prevSum,
        rng_state=np.random.get_state()
    )
    print(f"Checkpoint saved at weight {chunk_end}")

    del filterLeakage_chunk
    del simulated_inputs_chunk
    del baseline_inputs_chunk
    del filter_weights_chunk

# =========================
# Save final results
# =========================
extractedWeights = np.array(extractedWeights, dtype=np.float16)
t2 = time.time()

print(f"\ntime taken = {t2 - t1:.2f} Sec")
print(extractedWeights)

out_save_path = final_weights_path(savePath, layerName, filter_idx)
save_pickle(extractedWeights, out_save_path)

corr_save_path = final_corr_path(corrStatSavePath, layerName, filter_idx)
save_pickle(np.array(correlationStatistics, dtype=object), corr_save_path)

if os.path.exists(ckpt_path):
    os.remove(ckpt_path)
    print(f"Removed checkpoint: {ckpt_path}")

print(f"Saved extracted weights to: {out_save_path}")
print(f"Saved correlation stats to: {corr_save_path}")