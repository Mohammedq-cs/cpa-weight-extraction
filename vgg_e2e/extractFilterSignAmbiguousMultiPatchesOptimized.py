"""
Sign-ambiguous, multi-patch CPA filter extractor (vgg_e2e, disk-backed).

Same algorithm as vgg_sign_mitigation's extractor of the same name -- recover a
filter's weights up to a single global +/- sign by starting CPA at weight 0 with
prevSum = 0 and no first-weight seed -- but rebuilt on the shared primitives in
`cpaExtractCore` and the column-major memmap inputs in `patchInputs`, so:

  * inputs are never fully loaded into RAM: each selected patch is a
    `(numWeights, num_traces)` memmap and CPA reads one column at a time;
  * the TRUE-filter leakage trace is computed incrementally (a running per-patch
    accumulator) instead of materialising a full (num_traces, numWeights) array;
  * the per-(patch, weight) unique-count matrix is computed once per layer and
    cached next to the simulated inputs so every per-filter job reuses it.

Extraction math (patch choice, two-phase CPA, candidate tie-break, degenerate /
low-corr handling) is identical to the original; only IO and leakage
materialisation changed. The per-weight patch chosen is the one with the most
distinct input values (`select_patch`); only patches that are some weight's
argmax were materialised by the baseline/simulation step.
"""

import os
import gc
import time
import pickle
import argparse
import numpy as np

from cpaCupy import *
from utils import generate_weight_candidates
from patchInputs import (
    load_layer_inputs,
    align_inputs,
    load_unique_count_matrix,
    save_unique_count_matrix,
)
from cpaExtractCore import (
    select_patch,
    compute_unique_count_matrix,
    init_zero_prevsum_list,
    update_prevsum_all_patches,
    cpa_extract_at,
    IncrementalLeakage,
)
import sys


# ============================================================
# Checkpoint IO
# ============================================================

def checkpoint_path(output_dir, layer_name, filter_idx):
    return os.path.join(output_dir, layer_name, f"filter_{filter_idx}_checkpoint.pkl")


def save_checkpoint(path, next_weight_idx, extractedWeights, correlationStatistics,
                    chosen_patch_indices, chosen_unique_counts, prevSum_list,
                    true_sum_list, leak_pos, rng_state):
    """Atomic checkpoint. Carries both the prevSum accumulators (recovered-weight
    running sums) and the incremental-leakage accumulators (true-weight running
    sums) + their forward position, so a resumed job continues without
    recomputing either."""
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


# ============================================================
# Main extraction with checkpoints
# ============================================================

def extract_filter_weights_multi_patch(
    simulated_inputs, baseline_inputs, filter_weights, leakage_model,
    noise_level, threshold, ckpt_path, unique_count_matrix=None, checkpoint_every=40,
):
    """Recover one filter's weights (sign-ambiguous) column by column.

    For each weight i: pick the highest-unique patch (`select_patch`), read its
    column from the simulated memmap, get the TRUE leakage column from the
    incremental accumulator, run two-phase CPA, append the best candidate, and
    advance every patch's prevSum. Degenerate (<=1 unique or all-zero) columns
    and HD low-correlation columns get a zero weight, matching the original.
    """
    weight_candidates = generate_weight_candidates()
    num_patches = simulated_inputs.num_patches
    numWeights = simulated_inputs.num_weights

    filter_weights_flat = np.asarray(filter_weights).reshape(-1).astype(np.float16)
    if filter_weights_flat.shape[0] != numWeights:
        raise ValueError(
            f"filter_weights has {filter_weights_flat.shape[0]} weights, "
            f"but inputs have {numWeights} columns"
        )

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

    # ---------------- resume or initialise ----------------
    ckpt = load_checkpoint(ckpt_path)
    # Self-heal a stale checkpoint left over from a smaller selection (e.g. after
    # the orchestrator augmented the patch set on disk): its per-patch state no
    # longer matches num_patches, so discard it and restart rather than crashing.
    if ckpt is not None and len(ckpt["prevSum_list"]) != num_patches:
        print(f"Stale checkpoint: prevSum_list has {len(ckpt['prevSum_list'])} patches "
              f"but data has {num_patches} (selection changed); restarting from scratch.")
        sys.stdout.flush()
        ckpt = None
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
        # Restore incremental-leakage accumulators if present; else rebuild them.
        if ckpt.get("true_sum_list") is not None and int(ckpt.get("leak_pos", -1)) == next_weight_idx:
            inc_leak.restore(ckpt["true_sum_list"], ckpt["leak_pos"])
        else:
            inc_leak.advance_through(next_weight_idx)
        print(f"Resuming from weight index {next_weight_idx}")
        sys.stdout.flush()
    else:
        # Sign-ambiguous start: prevSum = 0, no ground-truth first weight.
        print("No checkpoint found. Starting from scratch (sign-ambiguous).")
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
            inc_leak.true_sum_list, inc_leak.pos, np.random.get_state(),
        )

    # ---------------- sanity ----------------
    for nm, lst in (("extractedWeights", extractedWeights),
                    ("correlationStatistics", correlationStatistics),
                    ("chosen_patch_indices", chosen_patch_indices),
                    ("chosen_unique_counts", chosen_unique_counts)):
        if len(lst) != next_weight_idx:
            raise ValueError(f"Checkpoint mismatch: len({nm})={len(lst)} != "
                             f"next_weight_idx={next_weight_idx}")
    if len(prevSum_list) != num_patches:
        raise ValueError(f"prevSum_list has {len(prevSum_list)} patches, data has {num_patches}")

    # ---------------- main loop ----------------
    t1 = time.time()
    for i in range(next_weight_idx, numWeights):
        best_p = select_patch(unique_count_matrix, i)
        best_uc = int(unique_count_matrix[best_p, i])
        # Advance the TRUE-weight accumulators for ALL patches at this weight and
        # fetch the chosen patch's cumulative leakage column (must happen every
        # weight, even degenerate ones, to keep the cumulative sums correct).
        leak_col = inc_leak.step(i, best_p)

        chosen_patch_indices.append(best_p)
        chosen_unique_counts.append(best_uc)

        input_col = np.ascontiguousarray(simulated_inputs.column(best_p, i))

        degenerate = best_uc <= 1 or np.all(input_col == 0)
        if degenerate:
            if best_uc <= 1:
                print(f"input column {i}: all patches have <=1 unique value. Zero weight.")
            else:
                print(f"input column {i}: chosen patch {best_p} all zeros. Zero weight.")
            sys.stdout.flush()
            extractedWeight = np.float16(0.0)
            extractedWeights.append(extractedWeight)
            correlationStatistics.append(np.zeros((100, 2), dtype=np.float64))
            update_prevsum_all_patches(prevSum_list, simulated_inputs, i, extractedWeight)
            save_checkpoint(
                ckpt_path, i + 1, extractedWeights, correlationStatistics,
                chosen_patch_indices, chosen_unique_counts, prevSum_list,
                inc_leak.true_sum_list, inc_leak.pos, np.random.get_state(),
            )
            continue

        top_corr, recovered_w, top_w, top_c = cpa_extract_at(
            i, input_col, leak_col, prevSum_list[best_p],
            weight_candidates, leakage_model, noise_level,
        )
        correlationStatistics.append(np.column_stack((top_w, top_c)))

        if leakage_model == "HD" and top_corr <= threshold:
            print(f"input column {i}: low correlation on patch {best_p}. Zero weight.")
            sys.stdout.flush()
            extractedWeight = np.float16(0.0)
            extractedWeights.append(extractedWeight)
            update_prevsum_all_patches(prevSum_list, simulated_inputs, i, extractedWeight)
            save_checkpoint(
                ckpt_path, i + 1, extractedWeights, correlationStatistics,
                chosen_patch_indices, chosen_unique_counts, prevSum_list,
                inc_leak.true_sum_list, inc_leak.pos, np.random.get_state(),
            )
            continue

        extractedWeights.append(recovered_w)
        update_prevsum_all_patches(prevSum_list, simulated_inputs, i, recovered_w)

        if ((i + 1) % checkpoint_every) == 0:
            save_checkpoint(
                ckpt_path, i + 1, extractedWeights, correlationStatistics,
                chosen_patch_indices, chosen_unique_counts, prevSum_list,
                inc_leak.true_sum_list, inc_leak.pos, np.random.get_state(),
            )
            print(f"Checkpoint saved after weight {i}")
            sys.stdout.flush()

    print(f"time taken = {time.time() - t1:.2f} Sec")
    sys.stdout.flush()

    extractedWeights = np.asarray(extractedWeights, dtype=np.float16)
    correlationStatistics = np.asarray(correlationStatistics, dtype=object)
    chosenPatchStatistics = np.column_stack((
        np.asarray(chosen_patch_indices, dtype=np.int32),
        np.asarray(chosen_unique_counts, dtype=np.int32),
    ))
    if len(extractedWeights) != numWeights:
        raise ValueError(f"Final extractedWeights {len(extractedWeights)} != numWeights {numWeights}")
    return extractedWeights, correlationStatistics, chosenPatchStatistics


# ============================================================
# Save final outputs
# ============================================================

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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign-ambiguous multi-patch CPA extractor (disk-backed).")
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
    args = parser.parse_args()

    print(f"seed={args.seed} noise={args.NoiseLevel} leakage={args.leakage_model} "
          f"checkpointEvery={args.checkpointEvery}")
    sys.stdout.flush()
    np.random.seed(args.seed)

    simulated_inputs = load_layer_inputs(args.simulatedInputsDir, args.layerName)
    baseline_inputs = load_layer_inputs(args.baseLineInputDir, args.layerName)
    simulated_inputs, baseline_inputs = align_inputs(simulated_inputs, baseline_inputs, args.layerName)
    print(f"{args.layerName}: {simulated_inputs.num_patches} patch(es), "
          f"{simulated_inputs.num_weights} weights, {simulated_inputs.num_traces} traces")
    sys.stdout.flush()

    with open(os.path.join("./model_weights", f"{args.layerName}/filter_{args.filterIdx}.pkl"), "rb") as f:
        filter_weights = pickle.load(f)
    filter_weights = np.asarray(filter_weights).reshape(-1).astype(np.float16)

    threshold = 0.2 if args.NoiseLevel == 0.0 else 0.018

    # Per-layer unique-count matrix: load the cache or compute+cache it (atomic),
    # so all of this layer's filter jobs and resumes share one computation.
    n_traces = simulated_inputs.num_traces
    num_patches = simulated_inputs.num_patches
    numWeights = simulated_inputs.num_weights
    unique_count_matrix = load_unique_count_matrix(
        args.simulatedInputsDir, args.layerName, n_traces, num_patches, numWeights
    )
    if unique_count_matrix is None:
        print("Unique-count cache miss; computing and caching.")
        sys.stdout.flush()
        unique_count_matrix = compute_unique_count_matrix(simulated_inputs)
        save_unique_count_matrix(args.simulatedInputsDir, args.layerName, unique_count_matrix, n_traces)
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
