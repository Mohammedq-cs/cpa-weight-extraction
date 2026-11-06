"""
Shared layer-input extraction + per-weight patch selection + disk-backed memmap
IO for vgg_e2e.

This is the single place that materialises Conv2d/Linear layer inputs, so the
baseline generator (`generate_baseline_inputs`), the per-layer simulation step,
and the sign-correction oracle recompute all go through identical code -- one
copy of "unfold -> select -> save/load patches", no mirrored extraction logic
and no hardcoded patch dict.

Two design choices, both driven by the "low RAM regardless of trace count"
requirement:

1. Per-weight patch selection (cpa_resnet style). For each Conv2d layer we keep,
   for every weight index, the spatial kernel application (patch) whose input
   column has the most distinct float16 values -- the best CPA target. The
   per-layer set we materialise is the UNION over all weights of that argmax
   patch (`compute_selected_patches`). Padding/edge patches that are degenerate
   for a weight are never that weight's argmax, so they drop out; only patches
   that actually carry signal for some weight survive.

2. Column-major memmaps. Each selected patch is stored TRANSPOSED, as a
   `(numWeights, num_traces)` .npy memmap, so logical column `w` of a patch is
   the contiguous row `mm[w]`. The CPA loop reads columns straight from disk in
   small windows (see `COLUMN_BUFFER_WINDOW`): `column(p, w)` pulls a contiguous
   block of `window` rows for patch p into a tiny RAM buffer and serves the
   per-weight columns from it, so a forward walk over a block of weights costs
   one sequential read per patch instead of `window` separate reads. The buffer
   is `window * num_traces * num_patches` float16 -- a small fraction of the full
   patch list -- so RAM stays bounded; a low-unique patch's window is loaded only
   when it is actually touched (e.g. to keep its prevSum in sync).

Selection indices are persisted once (from the REAL model, during baseline
generation) so the simulation step and the oracle recompute reuse the exact
same spatial patches -- baseline and simulated patch lists then align 1:1.
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.format import open_memmap

from modelUtils import intermediateActivations, setupHooks, removeHooks
from uniqueCount import count_distinct, new_present_buffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# How many consecutive columns (weights) to pull into RAM per patch on a memmap
# read. Bigger = fewer, larger sequential reads (less per-call overhead) at the
# cost of more buffer RAM (window * num_traces * num_patches float16). 1 disables
# buffering (read straight from the memmap). 40 matches the extractor's
# checkpoint cadence and keeps the buffer a small fraction of the full patch list.
COLUMN_BUFFER_WINDOW = 40


# ============================================================
# In-process view over a layer's column-major patch memmaps
# ============================================================

class LayerPatchInputs:
    """A layer's inputs as a list of column-major `(numWeights, num_traces)`
    arrays (memmaps in normal use). Presents the logical `(num_traces,
    numWeights)` view via `column(p, w)` -> contiguous 1-D array.

    Reads are buffered per patch in windows of `window` columns: the first
    access into a window pulls `mm[w0:w0+window]` (a contiguous, sequential
    read) into RAM, and subsequent columns in that window are served from the
    buffer. Buffers are PER PATCH and independent, so this is efficient for both
    the weight-outer CPA loop (all patches at one weight) and the patch-outer
    unique-count loop (one patch, all weights), and tolerates sign-correction
    rewinds (an out-of-window column just reloads). `window=1` disables it.

    The CPA/sign-correction code only ever needs single columns, the trace
    count, the weight count, and the patch count, so that is the whole API.
    """

    def __init__(self, arrays, window=COLUMN_BUFFER_WINDOW):
        if len(arrays) == 0:
            raise ValueError("LayerPatchInputs needs at least one patch array")
        self._a = arrays
        self.num_patches = len(arrays)
        self.num_weights = arrays[0].shape[0]
        self.num_traces = arrays[0].shape[1]
        self._window = max(1, int(window))
        self._reset_buffers()

    def _reset_buffers(self):
        # Per-patch window buffer + the column index it starts at (-1 = empty).
        self._buf = [None] * self.num_patches
        self._buf_start = [-1] * self.num_patches

    def column(self, p, w):
        """Logical column w of patch p == contiguous row w of the stored array,
        served from the per-patch window buffer (loading it on a miss)."""
        if self._window <= 1:
            return self._a[p][w]
        w0 = (w // self._window) * self._window
        if self._buf_start[p] != w0:
            # One contiguous sequential read of up to `window` rows into RAM.
            self._buf[p] = np.ascontiguousarray(self._a[p][w0:w0 + self._window])
            self._buf_start[p] = w0
        return self._buf[p][w - w0]

    def truncate_traces(self, n):
        """Restrict to the first n traces (used to align sim/baseline lengths)."""
        if n < self.num_traces:
            self._a = [arr[:, :n] for arr in self._a]
            self.num_traces = n
            self._reset_buffers()  # buffered windows are now the wrong width


def align_inputs(sim, baseline, layer_name):
    """Sanity-check matching patch/weight counts and truncate both to the same
    trace count (mirror of the old align_input_lists, for the memmap views)."""
    if sim.num_patches != baseline.num_patches:
        raise ValueError(
            f"{layer_name}: simulated has {sim.num_patches} patches but baseline "
            f"has {baseline.num_patches}"
        )
    if sim.num_weights != baseline.num_weights:
        raise ValueError(
            f"{layer_name}: simulated has {sim.num_weights} weights but baseline "
            f"has {baseline.num_weights}"
        )
    n = min(sim.num_traces, baseline.num_traces)
    sim.truncate_traces(n)
    baseline.truncate_traces(n)
    return sim, baseline


# ============================================================
# Per-weight patch selection (computed from the REAL model)
# ============================================================

def _per_weight_best_patch(layer_inp, layer, selection_max_traces):
    """For each weight, the index of the patch (over ALL L spatial kernel
    applications) whose input column has the most distinct float16 values.

    Returns (best_patch_per_weight (numWeights,) int32, L). Uses up to
    `selection_max_traces` traces (the argmax of the unique count is stable well
    before all traces are seen, and capping bounds the transient unfold).
    """
    in_ch = layer.in_channels
    kH, kW = layer.kernel_size
    numWeights = in_ch * kH * kW

    n = min(layer_inp.shape[0], selection_max_traces)
    patches = F.unfold(
        layer_inp[:n],
        kernel_size=(kH, kW),
        dilation=layer.dilation,
        padding=layer.padding,
        stride=layer.stride,
    )  # (n, numWeights, L)
    L = patches.shape[2]
    pn = patches[:, :numWeights, :].cpu().numpy().astype(np.float16)

    unique_count = np.empty((L, numWeights), dtype=np.int32)
    present = new_present_buffer()
    for p in range(L):
        Xp = pn[:, :, p]
        for j in range(numWeights):
            unique_count[p, j] = count_distinct(Xp[:, j], present)

    return unique_count.argmax(axis=0).astype(np.int32), L


def compute_selected_patches(layer_inp, layer, selection_max_traces=20_000):
    """Sorted union over weights of each weight's most-unique patch (only patches
    that are some weight's argmax are kept)."""
    best_patch_per_weight, _ = _per_weight_best_patch(layer_inp, layer, selection_max_traces)
    return np.unique(best_patch_per_weight).astype(np.int32)


def flag_sim_selection_mismatch(layer_inp, layer, baseline_selection, layer_name,
                                selection_max_traces=20_000):
    """Flag weights whose SIMULATED most-unique patch is NOT in the saved
    (baseline-derived) selection.

    The selection is fixed from the real model, and only those patches are
    materialised for BOTH baseline and simulated inputs. So the extractor always
    uses a patch present in both -- but if extraction error shifts a weight's
    best patch in the simulated domain to one outside the saved set, that
    sim-only patch is unavailable and the extractor falls back to the best patch
    present in both. This prints a per-layer warning when that happens so the
    case is visible. `layer_inp` is the SIMULATED (current-model) activation.
    Returns the list of (weight, sim_best_patch) mismatches.
    """
    if not isinstance(layer, nn.Conv2d):
        return []
    best_patch_per_weight, L = _per_weight_best_patch(layer_inp, layer, selection_max_traces)
    saved = set(int(p) for p in np.asarray(baseline_selection).ravel())
    mismatches = [(int(w), int(best_patch_per_weight[w]))
                  for w in range(best_patch_per_weight.shape[0])
                  if int(best_patch_per_weight[w]) not in saved]
    if mismatches:
        print(
            f"[selection-mismatch] {layer_name}: {len(mismatches)}/"
            f"{best_patch_per_weight.shape[0]} weights have a simulated best-patch "
            f"OUTSIDE the saved selection ({len(saved)} of {L} patches saved); the "
            f"extractor will use the best patch present in BOTH sim+baseline. "
            f"examples (weight->sim_best_patch): {mismatches[:10]}"
        )
    else:
        print(
            f"[selection-mismatch] {layer_name}: OK -- every weight's simulated "
            f"best-patch is in the saved selection."
        )
    sys.stdout.flush()
    return mismatches


def augment_selection_for_sim_mismatch(
        name, num_traces, real_layer, layer_name,
        sim_layer_inp, real_layer_inp,
        baseline_base_dir, sim_base_dir,
        write_sim=False, selection_max_traces=20_000, batch=5_000):
    """Materialise (instead of falling back) the patches the SIMULATED setting
    needs but the saved (real-derived) selection lacks. Conv2d only.

    For each weight whose SIMULATED most-unique patch is OUTSIDE the saved
    selection, APPEND that spatial patch to the saved set -- in FILE ORDER (saved
    slots keep their indices; new spatial indices are appended, NEVER globally
    re-sorted, so existing memmaps stay valid). The appended BASELINE memmaps are
    written from `real_layer_inp` (real-model activation, recovery-independent);
    `{name}.selection.npy` is grown to the union; the stale sim unique-count cache
    is removed (the orchestrator recomputes it for the union). With `write_sim`,
    the appended SIMULATED memmaps are written too (the extraction path instead
    lets `save_layer_inputs` rewrite the full union sim; the oracle path needs no
    sim on disk -- it recomputes the oracle sim in RAM).

    No-op (returns the existing selection unchanged, ZERO writes) when there is no
    mismatch -- so a layer whose every sim best-patch is already saved is
    byte-identical to before. Idempotent/monotonic: a re-run finds the
    previously-added indices already saved and only appends genuinely-new ones.
    Returns the (possibly grown) selection (int32, FILE order)."""
    saved = load_selection(baseline_base_dir, num_traces, name)
    if not isinstance(real_layer, nn.Conv2d):
        return saved
    if saved is None:
        raise FileNotFoundError(
            f"{name}: missing patch selection in {baseline_base_dir}; "
            f"run baseline generation first.")

    best_patch_per_weight, L = _per_weight_best_patch(
        sim_layer_inp, real_layer, selection_max_traces)
    saved_set = set(int(p) for p in np.asarray(saved).ravel())
    needed = sorted(set(int(p) for p in best_patch_per_weight) - saved_set)
    if not needed:
        return saved

    union = np.concatenate([np.asarray(saved, dtype=np.int32),
                            np.asarray(needed, dtype=np.int32)])
    start = len(saved)
    # Appended BASELINE memmaps (real-model activation). Existing baseline slots
    # [0, start) are untouched.
    _append_conv_patch_memmaps(name, num_traces, real_layer_inp, real_layer,
                               needed, start, baseline_base_dir, batch)
    if write_sim:
        _append_conv_patch_memmaps(name, num_traces, sim_layer_inp, real_layer,
                                   needed, start, sim_base_dir, batch)
    # Selection lives next to the baseline inputs; the sim step and the oracle
    # recompute both reload it.
    save_selection(baseline_base_dir, num_traces, name, union)
    # The sim unique-count cache now has the wrong num_patches; drop it so a stale
    # copy is never loaded (the orchestrator recomputes it for the union).
    stale = unique_count_cache_path(os.path.join(sim_base_dir, str(num_traces)), name)
    if os.path.exists(stale):
        os.remove(stale)

    shown = needed[:10]
    print(
        f"[selection-augment] {layer_name}: +{len(needed)} patches "
        f"{shown}{'...' if len(needed) > 10 else ''} -> {len(union)} total "
        f"(was {len(saved)} of {L} spatial patches)"
    )
    sys.stdout.flush()
    return union


# ============================================================
# Selection persistence (lives next to the baseline inputs)
# ============================================================

def selection_path(sel_dir, num_traces, name):
    return os.path.join(sel_dir, str(num_traces), f"{name}.selection.npy")


def save_selection(sel_dir, num_traces, name, selection):
    path = selection_path(sel_dir, num_traces, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(selection, dtype=np.int32))


def load_selection(sel_dir, num_traces, name):
    path = selection_path(sel_dir, num_traces, name)
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.int32)


# ============================================================
# Materialise layer inputs as column-major memmaps
# ============================================================

def _conv_patch_paths(input_dir, name, num_selected):
    return [os.path.join(input_dir, f"{name}.patch{i}.npy") for i in range(num_selected)]


def _linear_path(input_dir, name):
    return os.path.join(input_dir, f"{name}.npy")


def _fill_conv_patch_memmaps(mmaps, spatial_indices, layer_inp, layer, numWeights,
                             batch):
    """Fill already-opened `(numWeights, N)` memmaps from `layer_inp`'s unfold,
    one memmap per spatial patch index (`mmaps[i]` <- spatial `spatial_indices[i]`),
    in trace-batches so the transient unfold stays bounded. Shared by the full
    writer and the append-only augmenter."""
    kH, kW = layer.kernel_size
    N = layer_inp.shape[0]
    for ts in range(0, N, batch):
        te = min(ts + batch, N)
        u = F.unfold(
            layer_inp[ts:te],
            kernel_size=(kH, kW),
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride,
        )  # (b, numWeights, L)
        u = u[:, :numWeights, :].cpu().numpy().astype(np.float16)
        for i, p_idx in enumerate(spatial_indices):
            # (b, numWeights) -> (numWeights, b) into the column-major memmap
            mmaps[i][:, ts:te] = u[:, :, int(p_idx)].T
    for m in mmaps:
        m.flush()


def save_conv_patches_memmap(name, num_traces, layer_inp, layer, selection,
                             base_dir, batch=5_000):
    """Write each selected spatial patch as a `(numWeights, N)` .npy memmap,
    filling it in trace-batches so the transient unfold stays bounded."""
    in_ch = layer.in_channels
    kH, kW = layer.kernel_size
    numWeights = in_ch * kH * kW
    N = layer_inp.shape[0]

    input_dir = os.path.join(base_dir, str(num_traces))
    os.makedirs(input_dir, exist_ok=True)
    paths = _conv_patch_paths(input_dir, name, len(selection))
    mmaps = [
        open_memmap(p, mode='w+', dtype=np.float16, shape=(numWeights, N))
        for p in paths
    ]
    _fill_conv_patch_memmaps(mmaps, selection, layer_inp, layer, numWeights, batch)
    del mmaps
    return paths


def _append_conv_patch_memmaps(name, num_traces, layer_inp, layer, spatial_indices,
                               start_file_idx, base_dir, batch=5_000):
    """Append new `(numWeights, N)` patch memmaps for `spatial_indices`, named
    `{name}.patch{start_file_idx+i}.npy`, from `layer_inp`'s unfold. Existing
    patch files are never reopened, so the on-disk slots [0, start_file_idx) stay
    byte-identical. Used by `augment_selection_for_sim_mismatch`."""
    in_ch = layer.in_channels
    kH, kW = layer.kernel_size
    numWeights = in_ch * kH * kW
    N = layer_inp.shape[0]

    input_dir = os.path.join(base_dir, str(num_traces))
    os.makedirs(input_dir, exist_ok=True)
    paths = [os.path.join(input_dir, f"{name}.patch{start_file_idx + i}.npy")
             for i in range(len(spatial_indices))]
    mmaps = [
        open_memmap(p, mode='w+', dtype=np.float16, shape=(numWeights, N))
        for p in paths
    ]
    _fill_conv_patch_memmaps(mmaps, spatial_indices, layer_inp, layer, numWeights, batch)
    del mmaps
    return paths


def save_linear_memmap(name, num_traces, layer_inp, layer, base_dir):
    """Write a Linear layer's input as a single `(in_features, N)` .npy memmap."""
    numWeights = layer.in_features
    N = layer_inp.shape[0]
    input_dir = os.path.join(base_dir, str(num_traces))
    os.makedirs(input_dir, exist_ok=True)
    path = _linear_path(input_dir, name)
    arr = layer_inp[:, :numWeights].cpu().numpy().astype(np.float16)  # (N, in_features)
    mm = open_memmap(path, mode='w+', dtype=np.float16, shape=(numWeights, N))
    mm[:] = arr.T
    mm.flush()
    del mm
    return [path]


def save_layer_inputs(name, num_traces, layer_inp, layer, selection, base_dir):
    """Dispatch a layer's inputs to the right memmap writer. `selection` is
    ignored for Linear layers. This is the shared entry point used by both
    baseline generation and the per-layer simulation step."""
    if isinstance(layer, nn.Conv2d):
        return save_conv_patches_memmap(name, num_traces, layer_inp, layer, selection, base_dir)
    if isinstance(layer, nn.Linear):
        return save_linear_memmap(name, num_traces, layer_inp, layer, base_dir)
    raise TypeError(f"{name}: unsupported layer type {type(layer)}")


def expected_input_paths(input_dir, name, layer, selection):
    """Paths save_layer_inputs would write, for cheap cache-skip checks."""
    if isinstance(layer, nn.Linear):
        return [_linear_path(input_dir, name)]
    return _conv_patch_paths(input_dir, name, len(selection))


def load_layer_inputs(input_dir, name):
    """Load a layer's inputs as a LayerPatchInputs of read-only memmaps. Handles
    both the Linear single-file (`{name}.npy`) and Conv multi-patch
    (`{name}.patch{i}.npy`) layouts."""
    linear = _linear_path(input_dir, name)
    if os.path.exists(linear):
        return LayerPatchInputs([np.load(linear, mmap_mode='r')])

    arrays = []
    i = 0
    while True:
        p = os.path.join(input_dir, f"{name}.patch{i}.npy")
        if not os.path.exists(p):
            break
        arrays.append(np.load(p, mmap_mode='r'))
        i += 1
    if not arrays:
        raise FileNotFoundError(
            f"No inputs for layer {name} in {input_dir} "
            f"(neither {name}.npy nor {name}.patch0.npy)"
        )
    return LayerPatchInputs(arrays)


# ============================================================
# Unique-count matrix cache (computed once per layer, shared by all filters)
# ============================================================
# The per-(patch, weight) distinct-value matrix depends only on a layer's
# SIMULATED inputs, which are identical for every filter of that layer. Caching
# it next to the simulated inputs lets every per-filter SLURM job (and every
# resume) load it instead of re-reading all columns to recompute it.

def unique_count_cache_path(input_dir, name):
    return os.path.join(input_dir, f"{name}.unique_counts.pkl")


def save_unique_count_matrix(input_dir, name, unique_count_matrix, n_traces):
    """Persist the matrix atomically so concurrent array jobs never read a
    partial file."""
    ucm = np.asarray(unique_count_matrix, dtype=np.int32)
    num_patches, numWeights = ucm.shape
    payload = {
        "unique_count_matrix": ucm,
        "n_traces": int(n_traces),
        "num_patches": int(num_patches),
        "numWeights": int(numWeights),
    }
    path = unique_count_cache_path(input_dir, name)
    tmp = path + f".tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def load_unique_count_matrix(input_dir, name, expected_n_traces,
                             expected_num_patches, expected_numWeights):
    """Load the cached matrix only if its shape/trace-count matches the caller's
    (post-alignment) inputs; otherwise None so the caller recomputes."""
    path = unique_count_cache_path(input_dir, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if (payload.get("n_traces") != int(expected_n_traces)
            or payload.get("num_patches") != int(expected_num_patches)
            or payload.get("numWeights") != int(expected_numWeights)):
        return None
    mat = np.asarray(payload["unique_count_matrix"], dtype=np.int32)
    if mat.shape != (int(expected_num_patches), int(expected_numWeights)):
        return None
    return mat


# ============================================================
# Forward-pass helpers (capture a layer's input via hooks)
# ============================================================

def capture_layer_input(model, input_tensor, layer, batch_size=10_000):
    """Run the model in its current state and return one layer's concatenated
    input activation (CPU tensor). Assumes hooks are already installed."""
    model.eval()
    parts = []
    with torch.no_grad():
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        for batch in torch.split(input_tensor, batch_size, dim=0):
            batch = batch.to(device, non_blocking=True).half()
            _ = model(batch)
            layer_inp, _ = intermediateActivations[layer]
            parts.append(layer_inp.detach().cpu().clone())
    return torch.cat(parts, dim=0)


def generate_baseline_inputs(model, input_tensor, num_traces, base_dir,
                             batch=10_000, selection_max_traces=20_000):
    """Forward-pass the REAL model and write every Conv2d/Linear layer's baseline
    inputs as column-major memmaps under `base_dir/{num_traces}/`.

    For each Conv2d layer the per-weight patch selection is computed from the
    real activations and persisted (`{name}.selection.npy`) so the simulation
    step and oracle recompute reuse the identical spatial patches; only the
    selected patches are materialised. One forward pass per layer keeps RAM at
    roughly a single layer's activation regardless of trace count. Layers whose
    output files already exist are skipped (resumable).

    Assumes hooks are already installed (setupHooks). `base_dir` is e.g.
    `./baseLineInps/{bsInpsPath}`.
    """
    layers = {
        name: layer
        for name, layer in model.named_modules()
        if isinstance(layer, (nn.Conv2d, nn.Linear))
    }
    input_dir = os.path.join(base_dir, str(num_traces))

    for name, layer in layers.items():
        if isinstance(layer, nn.Linear):
            if os.path.exists(_linear_path(input_dir, name)):
                print(f"{name}: cached, skipping")
                sys.stdout.flush()
                continue
        else:
            sel = load_selection(base_dir, num_traces, name)
            if sel is not None and all(
                os.path.exists(p) for p in _conv_patch_paths(input_dir, name, len(sel))
            ):
                print(f"{name}: cached ({len(sel)} patches), skipping")
                sys.stdout.flush()
                continue

        layer_inp = capture_layer_input(model, input_tensor, layer, batch_size=batch)
        if isinstance(layer, nn.Conv2d):
            selection = compute_selected_patches(layer_inp, layer, selection_max_traces)
            save_selection(base_dir, num_traces, name, selection)
            save_conv_patches_memmap(name, num_traces, layer_inp, layer, selection, base_dir)
            print(f"{name}: {len(selection)} selected patches "
                  f"(from {layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]} weights)")
        else:
            save_linear_memmap(name, num_traces, layer_inp, layer, base_dir)
            print(f"{name}: linear inputs saved")
        sys.stdout.flush()
        del layer_inp


def generate_baseline_inputs_hooked(model, input_tensor, num_traces, base_dir,
                                    batch=10_000, selection_max_traces=20_000):
    """generate_baseline_inputs with hook setup/teardown managed for you."""
    hooks = setupHooks(model)
    try:
        generate_baseline_inputs(
            model, input_tensor, num_traces, base_dir,
            batch=batch, selection_max_traces=selection_max_traces,
        )
    finally:
        removeHooks(hooks)


def recompute_oracle_inputs(model, input_tensor, target_layer, target_name,
                            selection, tmp_dir=None, batch=10_000):
    """Forward-pass the current model and return the oracle layer's input as an
    in-RAM LayerPatchInputs (column-major `(numWeights, N)` arrays, window=1).

    Used by sign correction after an upstream filter flip. This runs in the
    single orchestrator process (big-RAM node), so the recomputed oracle inputs
    are kept in RAM rather than written to disk -- the previous memmap version
    rewrote the entire ~10GB oracle-sim file on every flip, which was the main
    sign-correction bottleneck. (`tmp_dir` is accepted for back-compat but no
    longer used; the disk-backed path matters only for the per-filter extractor
    jobs, whose inputs are written once by save_layer_inputs.)"""
    layer_inp = capture_layer_input(model, input_tensor, target_layer, batch_size=batch)

    if isinstance(target_layer, nn.Linear):
        numWeights = target_layer.in_features
        arr = layer_inp[:, :numWeights].cpu().numpy().astype(np.float16)   # (N, in_features)
        return LayerPatchInputs([np.ascontiguousarray(arr.T)], window=1)

    in_ch = target_layer.in_channels
    kH, kW = target_layer.kernel_size
    numWeights = in_ch * kH * kW
    N = layer_inp.shape[0]
    arrays = [np.empty((numWeights, N), dtype=np.float16) for _ in range(len(selection))]
    for ts in range(0, N, batch):
        te = min(ts + batch, N)
        u = F.unfold(
            layer_inp[ts:te], kernel_size=(kH, kW),
            dilation=target_layer.dilation, padding=target_layer.padding,
            stride=target_layer.stride,
        )
        u = u[:, :numWeights, :].cpu().numpy().astype(np.float16)
        for i, p_idx in enumerate(selection):
            arrays[i][:, ts:te] = u[:, :, int(p_idx)].T
    return LayerPatchInputs(arrays, window=1)
