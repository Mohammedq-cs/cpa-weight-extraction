"""
Layer-by-layer first-weight-known weight retrieval for cpa_resnet.

Derived from vgg_e2e/simulation.py with the sign-correction layer removed: the
per-filter extractor here is seeded with the TRUE first weight of each filter
(see extractFilterFirstWeightKnownMultiPatchesOptimized.py), so recovered
filters are NOT sign-ambiguous and no oracle correction step is needed. Each
layer is extracted, the recovered weights are loaded into the model, accuracy
is reported, and the next layer is processed against the already-substituted
upstream weights.

Single-responsibility helpers (mirrors the vgg_e2e structure so a future
sign-correction port has a recognisable seam):

    _collect_conv_linear_layers          -- enumerate the target layers
    _count_filters                       -- filters/units for a layer
    _fast_forward_load_layer             -- skip a layer, just load its weights
    _generate_sim_inputs_and_submit      -- write sim-input memmaps + fan out SLURM
    _load_recovered_and_groundtruth      -- read recovered + ground-truth filters

A `FirstWeightKnownRunContext` dataclass holds the run parameters and derives
all directory paths so the helpers stay short. Per-Conv2d patch selection
(highest unique values per weight) is materialised by patchInputs; there is no
hardcoded patch dict.
"""

import os
import glob
import time
import pickle
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import sys

from utils import calcAccBitwise, print_difference_histogram
from modelUtils import setupHooks, removeHooks, updateLayerWeights
from slurmManager import submit_batch
from patchInputs import (
    load_layer_inputs, save_layer_inputs, compute_selected_patches,
    capture_layer_input, save_unique_count_matrix,
)
from cpaExtractCore import compute_unique_count_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================================================
# Run context: static parameters + derived directory layout
# =========================================================

@dataclass
class FirstWeightKnownRunContext:
    """All parameters of one first-weight-known run, plus the directory layout
    that the per-layer steps read/write. Keeps the helper signatures short.

    `model` is the working/recovered model (drives SIMULATED activations + patch
    selection); `true_model` is the frozen ground-truth model (supplies the
    BASELINE activations for the measured leakage)."""
    model: nn.Module
    true_model: nn.Module
    data_loader: object
    numTraces: int
    noise_level: float
    leakage_model: str
    exp_name: str
    bsInpsPath: str
    # Delete a layer's transient sim + baseline patch memmaps once its weights are
    # recovered. These memmaps are per-layer scratch (only the recovered pickles
    # are kept); leaving them on disk made every layer's ~tens-of-GB memmaps
    # accumulate until the run hit `OSError: Disk quota exceeded`. Cleanup bounds
    # disk to roughly the in-flight layer(s). Set False (demo --keep_layer_memmaps)
    # to retain them for debugging.
    cleanup_memmaps: bool = True

    @property
    def baseline_base_dir(self):
        return f'./baseLineInps/{self.bsInpsPath}'

    @property
    def baseline_input_dir(self):
        return os.path.join(self.baseline_base_dir, str(self.numTraces))

    @property
    def sim_base_dir(self):
        return f'./simInps/N{self.noise_level}/{self.exp_name}'

    @property
    def output_base(self):
        return f'./N{self.noise_level}/{self.exp_name}output'

    def output_dir(self, name):
        return os.path.join(self.output_base, str(self.numTraces), name)


# =========================================================
# Small reusable helpers
# =========================================================

def submit_slurm_jobs(name, numTraces, numFilters, batch_size=64, num_parallel=8,
                      noise_level=0, leakage_model='HW', exp_name='', bsInpsPath='',
                      script_name="extractFilterFirstWeightKnownMultiPatchesOptimized"):
    """Submit per-filter SLURM jobs in batches."""
    startTime = time.time()
    for start in range(0, numFilters, batch_size):
        end = min(start + batch_size - 1, numFilters - 1)
        submit_batch(start, end, num_parallel, name, numTraces, noise_level,
                     leakage_model, exp_name=exp_name, bsInpsPath=bsInpsPath,
                     script_name=script_name)
    print(f"Time Taken = {((time.time() - startTime)/60):.2f} mins\n")


def load_extracted_filters(name, numTraces, numFilters, base_directory="./output"):
    """Stack the per-filter pickles for one layer into a numpy array."""
    filters_directory = os.path.join(base_directory, str(numTraces), name)
    out = []
    for filter_idx in range(numFilters):
        with open(os.path.join(filters_directory, f'filter_{filter_idx}.pkl'), 'rb') as f:
            out.append(pickle.load(f))
    return np.array(out)


def evaluate_extraction(layer_weights, extractedFilters):
    """Bitwise accuracy of recovered vs ground-truth filters (+ diff histogram)."""
    layerAccBW, layerSuBW = calcAccBitwise(layer_weights, extractedFilters)
    print_difference_histogram(layer_weights, extractedFilters)
    accuracy = layerAccBW / layerSuBW if layerSuBW != 0 else 0
    print(f"Layer Accuracy Bit Wise = {accuracy:.4f}")
    return accuracy


def _collect_conv_linear_layers(model):
    """Ordered {name: layer} of Conv2d/Linear layers + the name order."""
    layers = {
        name: layer
        for name, layer in model.named_modules()
        if isinstance(layer, (nn.Conv2d, nn.Linear))
    }
    return layers, list(layers.keys())


def _count_filters(layer):
    """Number of output filters/units (extraction targets) for a layer."""
    return layer.out_channels if isinstance(layer, nn.Conv2d) else layer.weight.data.shape[0]


# =========================================================
# Per-layer steps
# =========================================================

def _fast_forward_load_layer(ctx, name, layer, num_filters):
    """Testing fast-forward: load a layer's already-recovered weights into the
    model without re-extracting (so later layers' forward passes are correct)."""
    ff_filters = load_extracted_filters(name, ctx.numTraces, num_filters,
                                        base_directory=ctx.output_base)
    updateLayerWeights(layer, ff_filters)
    print(f"[fast-forward] {name}: loaded recovered weights, skipping")
    sys.stdout.flush()


def _generate_sim_inputs_and_submit(ctx, name, layer, t_layer, num_filters):
    """Write this layer's simulated AND baseline input memmaps and fan out the
    per-filter first-weight-known extraction SLURM jobs. No-op if the layer's
    outputs already exist.

    Patch selection is decided from the SIMULATED (working_model) activation --
    the attacker-side view -- over ALL traces on the GPU (compute_selected_patches
    defaults to the gpuUniqueCount path). The SAME selection is then used to
    materialise both the simulated memmaps (from working_model) and the baseline
    memmaps (from true_model / `t_layer`), so the two patch lists align 1:1 and
    the per-filter extractor reads a simulated input column and the baseline true
    leakage at the same patch. There is no baseline-derived selection anymore.
    """
    if os.path.isdir(ctx.output_dir(name)):
        return

    # --- SIMULATED activation -> selection (sim-driven) -> sim memmaps ---
    sim_A = capture_layer_input(ctx.model, ctx.data_loader, layer)
    selection = (compute_selected_patches(sim_A, layer)
                 if isinstance(layer, nn.Conv2d) else None)
    save_layer_inputs(name, ctx.numTraces, sim_A, layer, selection, ctx.sim_base_dir)
    if isinstance(layer, nn.Conv2d):
        print(f"{name}: {len(selection)} selected patches (sim-driven, all traces)")
        sys.stdout.flush()
    del sim_A

    # Per-layer unique-count matrix ONCE here (single process), cached next to the
    # simulated inputs so every per-filter SLURM job loads it instead of
    # recomputing (it depends only on the simulated inputs, shared across filters).
    sim_input_dir = os.path.join(ctx.sim_base_dir, str(ctx.numTraces))
    sim_inputs = load_layer_inputs(sim_input_dir, name)
    save_unique_count_matrix(
        sim_input_dir, name, compute_unique_count_matrix(sim_inputs), ctx.numTraces)
    del sim_inputs

    # --- BASELINE activation (true_model) -> SAME selection -> baseline memmaps ---
    base_A = capture_layer_input(ctx.true_model, ctx.data_loader, t_layer)
    save_layer_inputs(name, ctx.numTraces, base_A, t_layer, selection, ctx.baseline_base_dir)
    del base_A

    submit_slurm_jobs(
        name, ctx.numTraces, num_filters, batch_size=64, num_parallel=64,
        noise_level=ctx.noise_level, leakage_model=ctx.leakage_model,
        exp_name=ctx.exp_name, bsInpsPath=ctx.bsInpsPath,
        script_name="extractFilterFirstWeightKnownMultiPatchesOptimized")
    sys.stdout.flush()


def _wait_for_layer_outputs(ctx, name, num_filters, timeout_s=10800, poll_s=10):
    """Block until every per-filter output pickle for this layer exists.

    `submit_slurm_jobs` uses `sbatch --wait`, but that can return before the
    array's outputs are visible on the orchestrator's NFS view (and, in some
    setups, before the tasks even finish). The orchestrator reads the outputs
    immediately after, so poll until they all appear before loading -- this is a
    no-op when --wait behaved, and a safety net for the long run otherwise."""
    out_dir = ctx.output_dir(name)
    deadline = time.time() + timeout_s
    while True:
        missing = [i for i in range(num_filters)
                   if not os.path.exists(os.path.join(out_dir, f"filter_{i}.pkl"))]
        if not missing:
            return
        if time.time() > deadline:
            raise TimeoutError(
                f"{name}: {len(missing)}/{num_filters} filter outputs still "
                f"missing after {timeout_s}s (e.g. {missing[:10]}) in {out_dir}")
        time.sleep(poll_s)


def _load_recovered_and_groundtruth(ctx, name, num_filters):
    """Recovered filters (first-weight-known) and the ground-truth filters."""
    _wait_for_layer_outputs(ctx, name, num_filters)
    recovered = load_extracted_filters(name, ctx.numTraces, num_filters,
                                       base_directory=ctx.output_base)
    groundtruth = load_extracted_filters(name, 'model_weights', num_filters,
                                         base_directory='./')
    return recovered, groundtruth


def _cleanup_layer_memmaps(ctx, name):
    """Delete this layer's transient sim + baseline patch memmaps (and the sim
    unique-count cache) after its weights are recovered and propagated.

    Files are matched by the `{name}.` prefix so only this layer is touched
    (e.g. `layer1.1.conv2.*` never matches `layer1.0.conv1.*`). The per-filter
    SLURM jobs have already read these files by the time we get here; only the
    recovered pickles under output_base are kept. Best-effort: a missing file is
    not an error. The prewarm monitor keys its STOP off these files disappearing,
    so this is also the signal that the layer is done."""
    sim_input_dir = os.path.join(ctx.sim_base_dir, str(ctx.numTraces))
    removed = 0
    for d in (sim_input_dir, ctx.baseline_input_dir):
        for p in glob.glob(os.path.join(d, f"{name}.*")):
            try:
                os.remove(p)
                removed += 1
            except OSError as e:
                print(f"[cleanup] could not remove {p}: {e}")
    print(f"[cleanup] {name}: removed {removed} transient memmap file(s)")
    sys.stdout.flush()


# =========================================================
# Orchestrator
# =========================================================

def weightsRetrievalSimulationFirstWeightKnown(
    model: nn.Module, true_model: nn.Module, data_loader, numTraces: int,
    noise_level: int, leakage_model: str, exp_name: str, bsInpsPath: str,
    start_at_layer: str = None, cleanup_memmaps: bool = True,
) -> list:
    """Recover every conv/linear layer in order: extract (first-weight-known) ->
    load recovered -> evaluate -> propagate. Each step is one helper above; this
    function only sequences them.

    `model` is the working/recovered model; `true_model` is the frozen
    ground-truth model that supplies the baseline activations. Both are walked in
    the same named_modules() order so each working layer is paired with its
    true-model counterpart `t_layer`."""
    ctx = FirstWeightKnownRunContext(
        model=model, true_model=true_model, data_loader=data_loader,
        numTraces=numTraces, noise_level=noise_level, leakage_model=leakage_model,
        exp_name=exp_name, bsInpsPath=bsInpsPath, cleanup_memmaps=cleanup_memmaps)

    layers, layer_names_in_order = _collect_conv_linear_layers(model)
    true_layers, _ = _collect_conv_linear_layers(true_model)
    reached_start = start_at_layer is None
    accRes = []
    startTimeGlobal = time.time()

    for name, layer in layers.items():
        layerType = "Convolutional Layer" if isinstance(layer, nn.Conv2d) else "Fully Connected Layer"
        print(f"{layerType}: {name}")
        num_filters = _count_filters(layer)

        # Fast-forward: load earlier layers' weights without re-doing the work.
        if not reached_start:
            if name == start_at_layer:
                reached_start = True
            else:
                _fast_forward_load_layer(ctx, name, layer, num_filters)
                continue

        _generate_sim_inputs_and_submit(ctx, name, layer, true_layers[name], num_filters)

        recovered, groundtruth = _load_recovered_and_groundtruth(ctx, name, num_filters)
        accRes.append(evaluate_extraction(groundtruth, recovered))
        updateLayerWeights(layer, recovered)
        if ctx.cleanup_memmaps:
            _cleanup_layer_memmaps(ctx, name)
        sys.stdout.flush()

    print(f"Total Time Taken = {((time.time() - startTimeGlobal) / 60):.2f} mins")
    return accRes, model


def runWeightsRetrievalSimulationFirstWeightKnown(
    model, true_model, data_loader, numTraces,
    noise_level=0, leakage_model='HW', exp_name=None, bsInpsPath=None,
    start_at_layer=None, cleanup_memmaps=True,
):
    """Install activation hooks on BOTH models, run the orchestrator, remove
    hooks. `intermediateActivations` is keyed by module identity, so hooking the
    working and true models at once is safe -- each layer's two copies are
    distinct modules and capture_layer_input reads the one it forwarded."""
    hooks = setupHooks(model) + setupHooks(true_model)
    try:
        result = weightsRetrievalSimulationFirstWeightKnown(
            model, true_model, data_loader, numTraces,
            noise_level=noise_level, leakage_model=leakage_model,
            exp_name=exp_name, bsInpsPath=bsInpsPath,
            start_at_layer=start_at_layer, cleanup_memmaps=cleanup_memmaps)
    finally:
        removeHooks(hooks)
    return result
