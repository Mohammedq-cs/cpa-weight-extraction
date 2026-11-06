"""
Shared CPA extraction core for vgg_e2e.

This module holds the weight-recovery primitives that the sign-ambiguous
extractor (extractFilterSignAmbiguousMultiPatchesOptimized.py) and the
sign-correction probes (signCorrection.py) previously each re-implemented:

  * per-weight patch selection         -> select_patch
  * per-column unique-value counting    -> compute_unique_count_matrix
  * the two-phase CPA at one weight     -> cpa_extract_at
  * prevSum running-sum maintenance     -> advance_prevsum / update_prevsum_all_patches
                                           / rebuild_prevsum_through
  * incremental TRUE-weight leakage     -> IncrementalLeakage

Nothing here decides experiment policy (thresholds, when to flip, etc.); it is
just the math, so both callers stay byte-for-byte consistent.

The input abstraction is `LayerPatchInputs` (see patchInputs.py): a list of
column-major memmaps where logical column `w` of patch `p` is the contiguous
row `mm[p][w]`. All helpers below take either a `LayerPatchInputs` or anything
exposing `.num_patches`, `.num_traces`, `.num_weights`, and `.column(p, w)`.
"""

import numpy as np

from cpaCupy import *           # cp, hfm_kernel, popc_kernel, hd_kernel, CPA
from utils import (
    generate_weight_candidates,
    generate_system_noise_array,
    print_top_k_correlations,
)
from uniqueCount import count_distinct, new_present_buffer


# ============================================================
# Unique-value counting (column-major aware; fast counter in uniqueCount.py)
# ============================================================

def compute_unique_count_matrix(inputs):
    """Per-(patch, weight) distinct-value matrix of shape (num_patches, numWeights).

    `inputs` is a LayerPatchInputs (or any object exposing num_patches/
    num_weights/column). Reads one logical column at a time, so it works
    directly on column-major memmaps without materialising any patch.
    """
    num_patches = inputs.num_patches
    num_weights = inputs.num_weights
    out = np.empty((num_patches, num_weights), dtype=np.int32)
    present = new_present_buffer()
    for p in range(num_patches):
        for j in range(num_weights):
            out[p, j] = count_distinct(inputs.column(p, j), present)
    return out


def update_unique_count_matrix_block(base_ucm, inputs, start, end):
    """Return a copy of `base_ucm` with only columns [start, end) recomputed
    from `inputs`.

    Used by sign correction: flipping one upstream filter f changes only the
    oracle's input CHANNEL f, i.e. only the kH*kW columns of block f for every
    patch -- every other column is byte-identical to before, so its distinct
    count is unchanged. Recomputing just the block is ~(numWeights/blockSize)x
    cheaper than a full compute_unique_count_matrix and gives an identical
    matrix.
    """
    out = np.array(base_ucm, dtype=np.int32, copy=True)
    present = new_present_buffer()
    for p in range(inputs.num_patches):
        for j in range(start, end):
            out[p, j] = count_distinct(inputs.column(p, j), present)
    return out


# ============================================================
# Per-weight patch selection
# ============================================================

def select_patch(unique_count_matrix, w_idx, alt_min_unique=40):
    """Pick the patch (spatial kernel application) to use for weight w_idx.

    Prefer the patch with the most unique values at w_idx (best CPA
    distinguishability). BUT if that patch has no informative *previous* column
    (all of columns [0, w_idx) degenerate), its prevSum was advanced only over
    zero-weight columns and carries no real running sum, so CPA against it
    collapses; switch to a patch that DID accumulate signal, provided the
    alternative still has >= alt_min_unique unique values now. This mirrors the
    Phase-1 extractor's prev_has_signal fallback so the walk/verify never land
    on a dead-prevSum patch at a multi-patch boundary weight.
    """
    current_counts = unique_count_matrix[:, w_idx]
    best = int(np.argmax(current_counts))
    if w_idx == 0:
        return best
    prev_has_signal = np.any(unique_count_matrix[:, :w_idx] > 1, axis=1)
    if not prev_has_signal[best]:
        cand = np.flatnonzero(prev_has_signal)
        if cand.size > 0:
            alt = int(cand[int(np.argmax(current_counts[cand]))])
            if int(current_counts[alt]) >= alt_min_unique:
                return alt
    return best


# ============================================================
# prevSum helpers (running Σ recovered_weight · sim_input_column)
# ============================================================

def advance_prevsum(prevSum, input_col, weight):
    """Single-weight HFM advance: prevSum += weight * input_col (float16)."""
    out = hfm_kernel(
        cp.asarray(np.ascontiguousarray(input_col)),
        cp.float16(weight),
        cp.asarray(np.ascontiguousarray(prevSum)),
    ).get().astype(np.float16)
    return np.ascontiguousarray(out)


def init_zero_prevsum_list(inputs):
    """One zero prevSum (length num_traces) per patch."""
    n = inputs.num_traces
    return [
        np.ascontiguousarray(np.zeros(n, dtype=np.float16))
        for _ in range(inputs.num_patches)
    ]


def stack_columns(inputs, w_idx):
    """All patches' logical column w_idx as one contiguous (num_patches, N) array
    (gathered from the per-patch read buffers). Lets the per-patch prevSum/leakage
    advances run as ONE batched GPU kernel instead of num_patches separate ones."""
    return np.ascontiguousarray(np.stack(
        [np.ascontiguousarray(inputs.column(p, w_idx)) for p in range(inputs.num_patches)]
    ))


def update_prevsum_all_patches(prevSum_list, inputs, w_idx, weight):
    """Advance every patch's prevSum with the recovered weight at column w_idx.

    Vectorised: one hfm kernel over the (num_patches, N) stack rather than one
    call per patch (the multi-patch oracle layers have up to ~200 patches, so the
    per-patch Python+launch overhead dominated otherwise)."""
    P = inputs.num_patches
    if P == 1:
        prevSum_list[0] = advance_prevsum(prevSum_list[0], inputs.column(0, w_idx), weight)
        return
    cols = stack_columns(inputs, w_idx)                      # (P, N)
    prev = np.ascontiguousarray(np.stack([prevSum_list[p] for p in range(P)]))  # (P, N)
    out = hfm_kernel(cp.asarray(cols), cp.float16(weight), cp.asarray(prev)).get().astype(np.float16)
    for p in range(P):
        prevSum_list[p] = np.ascontiguousarray(out[p])


def rebuild_prevsum_through(inputs, extracted_weights, end_idx):
    """Rebuild prevSum_list from zero using extracted_weights[0:end_idx]."""
    prevSum_list = init_zero_prevsum_list(inputs)
    for w_idx in range(end_idx):
        update_prevsum_all_patches(
            prevSum_list, inputs, w_idx, np.float16(extracted_weights[w_idx])
        )
    return prevSum_list


# ============================================================
# Incremental TRUE-weight leakage (replaces full leakage arrays)
# ============================================================

class IncrementalLeakage:
    """Cumulative leakage trace for the TRUE filter, stepped one weight at a
    time so the full (num_traces, numWeights) leakage array is never stored.

    attack.compute_prevSum_HW defines leakage[:, i] = HW(Σ_{j<=i} w_j·x_j) for
    HW, or HD(Σ_{j<i}, Σ_{j<=i}) for HD -- a strictly cumulative quantity. We
    therefore keep one running float16 accumulator (the TRUE partial sum over
    BASELINE inputs) per patch and advance ALL of them at every weight (each
    patch may become the CPA-chosen one at some later weight, and its leakage
    then needs its own cumulative sum). The accumulators are tiny
    (num_patches * num_traces float16); the returned per-weight column is
    byte-identical to indexing the precomputed leakage array.

    Strictly forward: `step(w_idx, chosen_p)` requires w_idx == next index.
    `true_sum_list` is exposed for checkpointing; pass it (and `pos`) back via
    `restore` so an interrupted SLURM job resumes without recomputing.
    """

    def __init__(self, baseline_inputs, true_filter_weights, leakage_model):
        self._inputs = baseline_inputs
        self._w = np.asarray(true_filter_weights).reshape(-1).astype(np.float16)
        self._leakage_model = leakage_model
        n = baseline_inputs.num_traces
        self.true_sum_list = [
            np.ascontiguousarray(np.zeros(n, dtype=np.float16))
            for _ in range(baseline_inputs.num_patches)
        ]
        self.pos = 0  # next weight index to advance into true_sum_list

    def restore(self, true_sum_list, pos):
        """Reload accumulator state from a checkpoint (forward position `pos`)."""
        self.true_sum_list = [
            np.ascontiguousarray(np.asarray(x, dtype=np.float16)) for x in true_sum_list
        ]
        self.pos = int(pos)

    def advance_through(self, end_idx):
        """Advance accumulators to include weights [0, end_idx) without returning
        any leakage column. Used to rebuild state on resume when the checkpoint
        did not carry true_sum_list."""
        while self.pos < end_idx:
            self._advance_one(self.pos, chosen_p=None)

    def _advance_one(self, w_idx, chosen_p):
        # Vectorised over patches: one hfm over the (P, N) stack, then the chosen
        # patch's HW/HD leakage from its row -- instead of one kernel per patch.
        P = self._inputs.num_patches
        wj = cp.float16(self._w[w_idx])
        cols = stack_columns(self._inputs, w_idx)                       # (P, N)
        prev_gpu = cp.asarray(np.ascontiguousarray(np.stack(self.true_sum_list)))  # (P, N)
        inter = hfm_kernel(cp.asarray(cols), wj, prev_gpu)              # (P, N) on GPU
        leak_chosen = None
        if chosen_p is not None:
            if self._leakage_model == "HW":
                leak = popc_kernel(inter[chosen_p])
            else:
                leak = hd_kernel(prev_gpu[chosen_p], inter[chosen_p])
            leak_chosen = np.ascontiguousarray(leak.get().astype(np.float16))
        inter_cpu = inter.get().astype(np.float16)
        self.true_sum_list = [np.ascontiguousarray(inter_cpu[p]) for p in range(P)]
        self.pos += 1
        return leak_chosen

    def step(self, w_idx, chosen_p):
        """Advance to include weight w_idx and return the TRUE leakage column for
        patch `chosen_p` at that weight (HW or HD, cumulative)."""
        if w_idx != self.pos:
            raise ValueError(
                f"IncrementalLeakage.step called for weight {w_idx} but next is {self.pos}"
            )
        return self._advance_one(w_idx, chosen_p)


def compute_leakage(baseline_inputs, true_weights, leakage_model):
    """Compute the FULL cumulative TRUE-filter leakage as in-RAM column-major
    `(numWeights, num_traces)` arrays, one per patch.

    Used by sign correction, where the oracle's leakage is fixed across upstream
    flips and is indexed at arbitrary (patch, weight) by the verify probes. This
    runs in the single orchestrator process (a big-RAM node), so it is kept in
    RAM rather than written to disk -- the on-disk memmap version was pure write
    overhead here (the disk-backed design only matters for the 64 concurrent
    per-filter extractor jobs, which use IncrementalLeakage instead). Vectorised
    over patches: one hfm + one popc/hd over the (num_patches, N) stack per weight.
    """
    w = np.asarray(true_weights).reshape(-1).astype(np.float16)
    K = baseline_inputs.num_patches
    nw = baseline_inputs.num_weights
    N = baseline_inputs.num_traces
    leak_arrays = [np.empty((nw, N), dtype=np.float16) for _ in range(K)]
    true_sum = np.zeros((K, N), dtype=np.float16)   # (K, N) running cumulative sums
    for j in range(nw):
        wj = cp.float16(w[j])
        cols = stack_columns(baseline_inputs, j)              # (K, N)
        prev_gpu = cp.asarray(np.ascontiguousarray(true_sum))
        inter = hfm_kernel(cp.asarray(cols), wj, prev_gpu)    # (K, N)
        leak = popc_kernel(inter) if leakage_model == "HW" else hd_kernel(prev_gpu, inter)
        leak_cpu = leak.get().astype(np.float16)              # (K, N)
        for p in range(K):
            leak_arrays[p][j] = leak_cpu[p]
        true_sum = inter.get().astype(np.float16)             # (K, N)
    return leak_arrays


# ============================================================
# Two-phase CPA at a single weight
# ============================================================

def cpa_extract_at(
    w_idx, input_col, leak_col, prevSum, weight_candidates,
    leakage_model, noise_level,
):
    """Two-phase CPA (float32 prefilter -> float64 refine) at one weight.

    Returns (top_corr, recovered_weight, top_w, top_c):
      * top_corr        -- peak (float64) correlation, top_c[0]
      * recovered_weight-- best candidate via the shared tie-break (max
                           correlation, then largest weight value)
      * top_w, top_c    -- the refined pass's sorted candidate weights and
                           correlations (for the extractor's correlation stats)
    Callers that only need the first two can discard the rest.
    """
    sim_traces = generate_system_noise_array(leak_col, noise_level).astype(np.float32)

    final_corr = CPA(
        sim_traces, input_col, weight_candidates, prevSum,
        leakage_model=leakage_model, batch_size=1000, dtype=np.float32,
    )
    top_w, _ = print_top_k_correlations(
        final_corr, weight_candidates, w_idx, k=100, printF=False,
    )
    filtered = np.array(top_w)

    final_corr = CPA(
        sim_traces, input_col, filtered, prevSum,
        leakage_model=leakage_model, batch_size=1000, dtype=np.float64,
    )
    top_w, top_c = print_top_k_correlations(
        final_corr, filtered, w_idx, k=100, printF=False,
    )
    return float(top_c[0]), _best_candidate(final_corr, filtered), top_w, top_c


def _best_candidate(final_corr, filtered):
    """Shared tie-break: highest max-correlation row, then largest weight value."""
    corr_max = np.max(final_corr, axis=1)
    max_inds = np.where(corr_max == corr_max.max())[0]
    best_idx = max_inds[np.argmax(filtered[max_inds, 0])]
    return np.float16(filtered[best_idx, 0])
