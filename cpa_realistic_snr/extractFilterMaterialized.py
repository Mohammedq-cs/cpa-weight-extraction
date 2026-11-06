"""
Materialized multi-patch best-activation CPA extractor (EfficientNet).

Per-filter SLURM-array worker. Reads the materialized column-major patch memmaps
written by patchInputsMaterialized (only the union of each weight's best patch),
and recovers the filter's weights column by column with the shared two-phase CPA
core (cpaExtractCore). Identical math to
../cpa_sim_multi_patch/extractFilterMultiPatchOptimized.py; the only differences are:

  * inputs are loaded PER FILTER for a depthwise layer (`{name}.f{f}.patch*.npy`)
    and shared for a standard conv (`{name}.patch*.npy`), driven by the layer's
    `{name}.meta.npz` is_depthwise flag, and
  * the unique-count cache key is per-filter for depthwise.

Checkpoint/resume (extractedWeights + prevSum_list + IncrementalLeakage state +
RNG state) every --checkpointEvery weights, same as the source worker.
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np

from cpaCupy import *
from utils import generate_weight_candidates
from patchInputsMaterialized import (
    load_filter_inputs, align_inputs, load_meta,
    load_unique_count_matrix, save_unique_count_matrix,
)
from cpaExtractCore import (
    select_patch, compute_unique_count_matrix, init_zero_prevsum_list,
    update_prevsum_all_patches, cpa_extract_at, IncrementalLeakage,
)


def checkpoint_path(output_dir, layer_name, filter_idx):
    return os.path.join(output_dir, layer_name, f"filter_{filter_idx}_checkpoint.pkl")


def save_checkpoint(path, next_weight_idx, extractedWeights, correlationStatistics,
                    chosen_patch_indices, chosen_unique_counts, prevSum_list,
                    true_sum_list, leak_pos, rng_state):
    state = {
        "next_weight_idx": next_weight_idx,
        "extractedWeights": np.asarray(extractedWeights, dtype=np.float16),
        "correlationStatistics": np.asarray(correlationStatistics, dtype=object),
        "chosen_patch_indices": np.asarray(chosen_patch_indices, dtype=np.int32),
        "chosen_unique_counts": np.asarray(chosen_unique_counts, dtype=np.int32),
        "prevSum_list": [np.asarray(x, dtype=np.float16) for x in prevSum_list],
        "true_sum_list": [np.asarray(x, dtype=np.float16) for x in true_sum_list],
        "leak_pos": int(leak_pos),
        "rng_state": rng_state,
    }
    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    if os.path.exists(path):
        os.remove(path)
    os.replace(tmp_path, path)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_filter_weights_multi_patch(
    simulated_inputs, baseline_inputs, filter_weights, leakage_model,
    noise_level, threshold, ckpt_path, unique_count_matrix=None, checkpoint_every=40,
    prev_known=False,
):
    """Recover one filter's weights column by column (multi-patch best-activation).
    Generic over LayerPatchInputs, so depthwise (numWeights=9) and standard
    (numWeights=27) both work unchanged.

    prev_known (BS / oracle): advance the running prevSum with the TRUE previous
    weights instead of the recovered ones, so each weight's CPA sees the correct
    prefix sum (no intra-filter error propagation). The recovered weight is still
    the CPA winner; only the prefix used for the next weights is the oracle one."""
    weight_candidates = generate_weight_candidates()
    num_patches = simulated_inputs.num_patches
    numWeights = simulated_inputs.num_weights

    filter_weights_flat = np.asarray(filter_weights).reshape(-1).astype(np.float16)
    if filter_weights_flat.shape[0] != numWeights:
        raise ValueError(
            f"filter_weights has {filter_weights_flat.shape[0]} weights, "
            f"but inputs have {numWeights} columns")

    if (unique_count_matrix is None
            or np.asarray(unique_count_matrix).shape != (num_patches, numWeights)):
        print("Computing unique-count matrix...")
        sys.stdout.flush()
        unique_count_matrix = compute_unique_count_matrix(simulated_inputs)
    else:
        unique_count_matrix = np.asarray(unique_count_matrix, dtype=np.int32)
        print("Using precomputed unique-count matrix.")
        sys.stdout.flush()

    inc_leak = IncrementalLeakage(baseline_inputs, filter_weights_flat, leakage_model)

    ckpt = load_checkpoint(ckpt_path)
    if ckpt is not None:
        print(f"Loading checkpoint from: {ckpt_path}")
        sys.stdout.flush()
        next_weight_idx = int(ckpt["next_weight_idx"])
        extractedWeights = list(np.asarray(ckpt["extractedWeights"]).astype(np.float16))
        correlationStatistics = list(ckpt["correlationStatistics"])
        chosen_patch_indices = list(np.asarray(ckpt["chosen_patch_indices"]).astype(np.int32))
        chosen_unique_counts = list(np.asarray(ckpt["chosen_unique_counts"]).astype(np.int32))
        prevSum_list = [np.ascontiguousarray(np.asarray(x, dtype=np.float16))
                        for x in ckpt["prevSum_list"]]
        np.random.set_state(ckpt["rng_state"])
        if ckpt.get("true_sum_list") is not None and int(ckpt.get("leak_pos", -1)) == next_weight_idx:
            inc_leak.restore(ckpt["true_sum_list"], ckpt["leak_pos"])
        else:
            inc_leak.advance_through(next_weight_idx)
        print(f"Resuming from weight index {next_weight_idx}")
        sys.stdout.flush()
    else:
        print("No checkpoint found. Starting from scratch.")
        sys.stdout.flush()
        extractedWeights = []
        correlationStatistics = []
        chosen_patch_indices = []
        chosen_unique_counts = []
        prevSum_list = init_zero_prevsum_list(simulated_inputs)
        next_weight_idx = 0
        save_checkpoint(
            ckpt_path, next_weight_idx, extractedWeights, correlationStatistics,
            chosen_patch_indices, chosen_unique_counts, prevSum_list,
            inc_leak.true_sum_list, inc_leak.pos, np.random.get_state())

    for nm, lst in (("extractedWeights", extractedWeights),
                    ("correlationStatistics", correlationStatistics),
                    ("chosen_patch_indices", chosen_patch_indices),
                    ("chosen_unique_counts", chosen_unique_counts)):
        if len(lst) != next_weight_idx:
            raise ValueError(f"Checkpoint mismatch: len({nm})={len(lst)} != "
                             f"next_weight_idx={next_weight_idx}")
    if len(prevSum_list) != num_patches:
        raise ValueError(f"prevSum_list has {len(prevSum_list)} patches, data has {num_patches}")

    t1 = time.time()
    for i in range(next_weight_idx, numWeights):
        best_p = select_patch(unique_count_matrix, i)
        best_uc = int(unique_count_matrix[best_p, i])
        leak_col = inc_leak.step(i, best_p)

        chosen_patch_indices.append(best_p)
        chosen_unique_counts.append(best_uc)

        input_col = np.ascontiguousarray(simulated_inputs.column(best_p, i))

        degenerate = best_uc <= 1 or np.all(input_col == 0)
        if degenerate:
            extractedWeight = np.float16(0.0)
            extractedWeights.append(extractedWeight)
            correlationStatistics.append(np.zeros((100, 2), dtype=np.float64))
            adv_w = np.float16(filter_weights_flat[i]) if prev_known else extractedWeight
            update_prevsum_all_patches(prevSum_list, simulated_inputs, i, adv_w)
            save_checkpoint(
                ckpt_path, i + 1, extractedWeights, correlationStatistics,
                chosen_patch_indices, chosen_unique_counts, prevSum_list,
                inc_leak.true_sum_list, inc_leak.pos, np.random.get_state())
            continue

        top_corr, recovered_w, top_w, top_c = cpa_extract_at(
            i, input_col, leak_col, prevSum_list[best_p],
            weight_candidates, leakage_model, noise_level)
        correlationStatistics.append(np.column_stack((top_w, top_c)))

        if leakage_model == "HD" and top_corr <= threshold:
            extractedWeight = np.float16(0.0)
            extractedWeights.append(extractedWeight)
            adv_w = np.float16(filter_weights_flat[i]) if prev_known else extractedWeight
            update_prevsum_all_patches(prevSum_list, simulated_inputs, i, adv_w)
            save_checkpoint(
                ckpt_path, i + 1, extractedWeights, correlationStatistics,
                chosen_patch_indices, chosen_unique_counts, prevSum_list,
                inc_leak.true_sum_list, inc_leak.pos, np.random.get_state())
            continue

        extractedWeights.append(recovered_w)
        adv_w = np.float16(filter_weights_flat[i]) if prev_known else recovered_w
        update_prevsum_all_patches(prevSum_list, simulated_inputs, i, adv_w)

        if ((i + 1) % checkpoint_every) == 0:
            save_checkpoint(
                ckpt_path, i + 1, extractedWeights, correlationStatistics,
                chosen_patch_indices, chosen_unique_counts, prevSum_list,
                inc_leak.true_sum_list, inc_leak.pos, np.random.get_state())
            print(f"Checkpoint saved after weight {i}")
            sys.stdout.flush()

    print(f"time taken = {time.time() - t1:.2f} Sec")
    sys.stdout.flush()

    extractedWeights = np.asarray(extractedWeights, dtype=np.float16)
    correlationStatistics = np.asarray(correlationStatistics, dtype=object)
    chosenPatchStatistics = np.column_stack((
        np.asarray(chosen_patch_indices, dtype=np.int32),
        np.asarray(chosen_unique_counts, dtype=np.int32)))
    if len(extractedWeights) != numWeights:
        raise ValueError(f"Final extractedWeights {len(extractedWeights)} != numWeights {numWeights}")
    return extractedWeights, correlationStatistics, chosenPatchStatistics


def save_outputs(savePath, corrStatSavePath, layerName, filter_idx,
                 extractedWeights, correlationStatistics, chosenPatchStatistics):
    os.makedirs(os.path.join(savePath, f"{layerName}"), exist_ok=True)
    os.makedirs(os.path.join(corrStatSavePath, f"{layerName}"), exist_ok=True)
    with open(os.path.join(savePath, f"{layerName}/filter_{filter_idx}.pkl"), "wb") as f:
        pickle.dump(extractedWeights, f)
    with open(os.path.join(corrStatSavePath, f"{layerName}/filter_{filter_idx}.pkl"), "wb") as f:
        pickle.dump(correlationStatistics, f)
    with open(os.path.join(corrStatSavePath, f"{layerName}/filter_{filter_idx}_chosen_patches.pkl"), "wb") as f:
        pickle.dump(chosenPatchStatistics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Materialized multi-patch CPA extractor (depthwise-aware).")
    parser.add_argument("--baseLineInputDir", type=str, required=True)
    parser.add_argument("--simulatedInputsDir", type=str, required=True)
    parser.add_argument("--outputDir", type=str, required=True)
    parser.add_argument("--corrStatSavePath", type=str, required=True)
    parser.add_argument("--layerName", type=str, required=True)
    parser.add_argument("--filterIdx", type=int, required=True)
    parser.add_argument("--leakage_model", type=str, required=True, choices=["HW", "HD"])
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--NoiseLevel", type=float, default=0.0)
    parser.add_argument("--checkpointEvery", type=int, default=40)
    parser.add_argument("--weightsDir", type=str, default="./model_weights",
                        help="Dir holding ground-truth weights {layer}/filter_{idx}.pkl "
                             "(used to synthesize the TRUE leakage traces).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="HD-only: force weight to 0 if peak CPA correlation <= threshold. "
                             "Default 0.2 (noiseless) / 0.018 (noisy). Set very small (e.g. 0.0) "
                             "to keep the CPA winner for nearly all weights.")
    parser.add_argument("--prev_known", action="store_true",
                        help="BS/oracle: advance prevSum with TRUE previous weights (no "
                             "intra-filter error propagation). Recovered weight is still the CPA winner.")
    args = parser.parse_args()

    print(f"seed={args.seed} noise={args.NoiseLevel} leakage={args.leakage_model} "
          f"layer={args.layerName} filter={args.filterIdx}")
    sys.stdout.flush()
    np.random.seed(args.seed)

    meta = load_meta(args.simulatedInputsDir, args.layerName)
    depthwise = bool(meta["is_depthwise"])

    simulated_inputs = load_filter_inputs(args.simulatedInputsDir, args.layerName, args.filterIdx, depthwise)
    baseline_inputs = load_filter_inputs(args.baseLineInputDir, args.layerName, args.filterIdx, depthwise)
    simulated_inputs, baseline_inputs = align_inputs(simulated_inputs, baseline_inputs, args.layerName)
    print(f"{args.layerName} f{args.filterIdx} (depthwise={depthwise}): "
          f"{simulated_inputs.num_patches} patch(es), {simulated_inputs.num_weights} weights, "
          f"{simulated_inputs.num_traces} traces")
    sys.stdout.flush()

    with open(os.path.join(args.weightsDir, f"{args.layerName}/filter_{args.filterIdx}.pkl"), "rb") as f:
        filter_weights = pickle.load(f)
    filter_weights = np.asarray(filter_weights).reshape(-1).astype(np.float16)

    # HD-only zeroing threshold: a weight whose peak CPA correlation <= threshold
    # is forced to 0.0. --threshold overrides the default (0.2 noiseless / 0.018 noisy);
    # set it very small (e.g. 0.0) to trust the CPA winner for nearly all weights.
    threshold = args.threshold if args.threshold is not None else (
        0.2 if args.NoiseLevel == 0.0 else 0.018)
    print(f"HD zeroing threshold = {threshold}")

    ucm_key = f"{args.layerName}.f{args.filterIdx}" if depthwise else args.layerName
    n_traces = simulated_inputs.num_traces
    num_patches = simulated_inputs.num_patches
    numWeights = simulated_inputs.num_weights
    unique_count_matrix = load_unique_count_matrix(
        args.simulatedInputsDir, ucm_key, n_traces, num_patches, numWeights)
    if unique_count_matrix is None:
        print("Unique-count cache miss; computing and caching.")
        sys.stdout.flush()
        unique_count_matrix = compute_unique_count_matrix(simulated_inputs)
        save_unique_count_matrix(args.simulatedInputsDir, ucm_key, unique_count_matrix, n_traces)
    else:
        print("Loaded cached unique-count matrix.")
        sys.stdout.flush()

    ckpt_path = checkpoint_path(args.outputDir, args.layerName, args.filterIdx)

    extractedWeights, correlationStatistics, chosenPatchStatistics = extract_filter_weights_multi_patch(
        simulated_inputs=simulated_inputs,
        baseline_inputs=baseline_inputs,
        filter_weights=filter_weights,
        leakage_model=args.leakage_model,
        noise_level=args.NoiseLevel,
        threshold=threshold,
        ckpt_path=ckpt_path,
        unique_count_matrix=unique_count_matrix,
        checkpoint_every=args.checkpointEvery,
        prev_known=args.prev_known,
    )

    print(extractedWeights)
    sys.stdout.flush()
    save_outputs(args.outputDir, args.corrStatSavePath, args.layerName, args.filterIdx,
                 extractedWeights, correlationStatistics, chosenPatchStatistics)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"Removed checkpoint: {ckpt_path}")
    print("Done.")
    sys.stdout.flush()
