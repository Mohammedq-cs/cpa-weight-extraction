"""
Multi-patch best-filter-activation CPA filter extractor.

`extract_filter_weights_multi_patch` recovers one filter's weights and is called
in-process, per filter, by the orchestrator's executor (simulationInproc /
workerPool). For each weight i of the target filter:

  1. pick the patch with the most distinct simulated input values (`select_patch`),
  2. read its column from the input provider (a ColumnProvider serving columns
     from a RAM-resident activation),
  3. fetch the TRUE-weight cumulative leakage for that patch from the running
     `IncrementalLeakage` accumulator,
  4. run two-phase CPA (float32 prefilter -> float64 refine) and pick the
     recovered weight via the shared tie-break,
  5. advance every patch's prevSum so the next weight's CPA is correct.

Degenerate (<=1 unique) columns and HD low-correlation columns get zero weight,
matching ../cpa_sim_first_patch/extractFilterTwoPhases.py. A checkpoint
(extractedWeights + prevSum_list + IncrementalLeakage state + RNG state) is
written every `checkpoint_every` weights so an interrupted job resumes without
recomputing anything.

Inputs are ColumnProvider views (see columnProvider.py); the per-(patch, weight)
unique-count matrix is passed in by the orchestrator (`unique_count_matrix=`),
which computes and caches it once per layer.
"""

import os
import sys
import time
import pickle
import numpy as np

from cpaCupy import *
from utils import generate_weight_candidates
from cpaExtractCore import (
    select_patch,
    compute_unique_count_matrix,
    init_zero_prevsum_list,
    update_prevsum_all_patches,
    cpa_extract_at,
    IncrementalLeakage,
)


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
    """Recover one filter's weights column by column using the multi-patch
    best-activation pipeline. Returns (extractedWeights, correlationStatistics,
    chosenPatchStatistics)."""
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
