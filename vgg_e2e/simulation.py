"""
Layer-by-layer sign-ambiguous weight retrieval for vgg_e2e.

Structure: one orchestrator (`weightsRetrievalSimulationSignAmbiguous`) that
walks the conv/linear layers and delegates each well-defined step to a small
single-responsibility helper:

    _collect_conv_linear_layers          -- enumerate the target layers
    _count_filters                       -- filters/units for a layer
    _fast_forward_load_layer             -- skip a layer, just load its weights
    _generate_sim_inputs_and_submit      -- write sim-input memmaps + fan out SLURM
    _load_recovered_and_groundtruth      -- read recovered + ground-truth filters
    _correct_signs_with_conv_oracle      -- Conv2d-oracle unconditional sweep
    _correct_signs_with_fc_oracle        -- Linear-oracle head-to-head sweep
    _correct_terminal_layer_signs        -- per-class-accuracy fix (last Linear)
    _sign_correct_layer                  -- dispatch to the right corrector above

A `SignAmbiguousRunContext` carries the static run parameters + derived paths so
the helpers stay short. Patch selection per weight = the highest-unique-value
patch (materialised by patchInputs); there is no hardcoded patch dict.
"""

import os
import time
import pickle
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import sys

from convNet import cifar10_vgg11_bn
from utils import calcAccBitwise, print_difference_histogram
from modelUtils import setupHooks, removeHooks, updateLayerWeights
from slurmManager import submit_batch
from patchInputs import (
    load_layer_inputs, load_selection, save_layer_inputs,
    capture_layer_input, save_unique_count_matrix, flag_sim_selection_mismatch,
    augment_selection_for_sim_mismatch,
)
from cpaExtractCore import compute_unique_count_matrix
from signCorrection import (
    correct_layer_signs,
    correct_layer_signs_via_fc,
    correct_last_layer_signs_by_class_accuracy,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================================================
# Run context: static parameters + derived directory layout
# =========================================================

@dataclass
class SignAmbiguousRunContext:
    """All parameters of one sign-ambiguous run, plus the directory layout that
    the per-layer steps read/write. Keeps the helper signatures short."""
    model: nn.Module
    data_loader: object
    numTraces: int
    noise_level: float
    leakage_model: str
    exp_name: str
    bsInpsPath: str
    g_idx: int = 0
    fc_num_probes: int = 8
    test_loader: object = None
    save_unique_counts: bool = False
    real_model_path: str = 'cifar10_vgg11_bn-eaeebf42.pt'
    # Lazily reloaded pretrained REAL model (+ its capture hooks), used only to
    # regenerate augmented baseline patch memmaps on demand. Private, throwaway.
    _real_model: nn.Module = field(default=None, repr=False, compare=False)
    _real_hooks: list = field(default=None, repr=False, compare=False)

    def real_model(self):
        """The pretrained REAL model (fp16, eval) with capture hooks installed,
        reloaded once per run. Its activations reproduce baseline generation
        byte-for-byte (same load-then-half order as demoSignAmbiguous), so
        appended baseline patches line up with the originals. ~39MB; negligible."""
        if self._real_model is None:
            m = cifar10_vgg11_bn().to(device)
            m.load_state_dict(torch.load(self.real_model_path, map_location=device))
            m = m.half().eval()
            self._real_model = m
            self._real_hooks = setupHooks(m)
        return self._real_model

    def real_layer(self, name):
        """The REAL model's module for `name` (distinct object from ctx.model's,
        so capture_layer_input keys it separately in intermediateActivations)."""
        return dict(self.real_model().named_modules())[name]

    def release_real_model(self):
        """Remove the real model's capture hooks at end of run (optional)."""
        if self._real_hooks is not None:
            removeHooks(self._real_hooks)
            self._real_hooks = None
        self._real_model = None

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

    @property
    def unique_counts_diag_dir(self):
        return os.path.join(f'./uniqueCounts/N{self.noise_level}/{self.exp_name}',
                            str(self.numTraces))


# =========================================================
# Small reusable helpers
# =========================================================

def submit_slurm_jobs(name, numTraces, numFilters, batch_size=64, num_parallel=8,
                      noise_level=0, leakage_model='HW', exp_name='', bsInpsPath='',
                      script_name="extractFilterSignAmbiguousMultiPatchesOptimized"):
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


def _find_oracle_layer_name(layer_names_in_order, layers, current_name):
    """Next Conv2d successor; else next Linear successor (lets the last conv be
    corrected via the FC oracle). None only for a Linear with no successor."""
    try:
        idx = layer_names_in_order.index(current_name)
    except ValueError:
        return None
    for next_name in layer_names_in_order[idx + 1:]:
        if isinstance(layers[next_name], nn.Conv2d):
            return next_name
    for next_name in layer_names_in_order[idx + 1:]:
        if isinstance(layers[next_name], nn.Linear):
            return next_name
    return None


# =========================================================
# Per-layer steps
# =========================================================

def _fast_forward_load_layer(ctx, name, layer, num_filters):
    """Testing fast-forward: load a layer's already-recovered weights into the
    model without re-extracting or re-correcting (so later layers' forward
    passes are correct)."""
    ff_filters = load_extracted_filters(name, ctx.numTraces, num_filters,
                                        base_directory=ctx.output_base)
    updateLayerWeights(layer, ff_filters)
    print(f"[fast-forward] {name}: loaded recovered weights, skipping")
    sys.stdout.flush()


def _save_unique_count_diag(ctx, name, selection, sim_ucm):
    """Diagnostic (gated by --save_unique_counts): per layer, dump the per-weight
    input distinct-value counts over the SAVED (union) patches, for the SIMULATED
    inputs (primary) and the BASELINE inputs (sanity). Both share the same
    selection, so the matrices are 1:1-aligned `(num_patches, numWeights)`; the
    per-weight max is each weight's best-patch count (the CPA target signal). No
    per-output-filter axis: the input column at (patch, weight) is identical for
    every output filter, so it would be redundant."""
    base_inputs = load_layer_inputs(ctx.baseline_input_dir, name)
    base_ucm = compute_unique_count_matrix(base_inputs)
    payload = {
        "layer": name,
        "selection": (np.asarray(selection, dtype=np.int32) if selection is not None
                      else None),
        "sim_unique_counts": np.asarray(sim_ucm, dtype=np.int32),
        "base_unique_counts": np.asarray(base_ucm, dtype=np.int32),
        "sim_per_weight_max": np.asarray(sim_ucm, dtype=np.int32).max(axis=0),
        "base_per_weight_max": np.asarray(base_ucm, dtype=np.int32).max(axis=0),
        "n_traces": int(ctx.numTraces),
    }
    out_dir = ctx.unique_counts_diag_dir
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.unique_counts_diag.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[unique-counts] {name}: saved per-weight sim+baseline distinct-value "
          f"counts -> {path}")
    sys.stdout.flush()


def _generate_sim_inputs_and_submit(ctx, name, layer, num_filters):
    """Write this layer's simulated-input memmaps (reusing the persisted patch
    selection) and fan out the per-filter sign-ambiguous extraction SLURM jobs.
    No-op if the layer's outputs already exist."""
    if os.path.isdir(ctx.output_dir(name)):
        return
    layer_inp = capture_layer_input(ctx.model, ctx.data_loader, layer)
    selection = (load_selection(ctx.baseline_base_dir, ctx.numTraces, name)
                 if isinstance(layer, nn.Conv2d) else None)
    if isinstance(layer, nn.Conv2d) and selection is None:
        raise FileNotFoundError(
            f"{name}: missing patch selection in {ctx.baseline_base_dir}; "
            f"run baseline generation first.")
    # If any weight's SIMULATED best patch is outside the saved (real-derived)
    # selection, materialise it now (append baseline patches from the real model +
    # grow {name}.selection.npy) instead of falling back. No-op when there is no
    # mismatch, so the validated path stays byte-identical. `selection` becomes the
    # union; save_layer_inputs below then writes the full union sim memmaps.
    if isinstance(layer, nn.Conv2d):
        real_inp = capture_layer_input(ctx.real_model(), ctx.data_loader, ctx.real_layer(name))
        selection = augment_selection_for_sim_mismatch(
            name, ctx.numTraces, ctx.real_layer(name), name,
            sim_layer_inp=layer_inp, real_layer_inp=real_inp,
            baseline_base_dir=ctx.baseline_base_dir, sim_base_dir=ctx.sim_base_dir,
            write_sim=False)
        del real_inp
    save_layer_inputs(name, ctx.numTraces, layer_inp, layer, selection, ctx.sim_base_dir)
    # Post-augmentation visibility check: should now print "OK" (every sim
    # best-patch is in the saved union).
    if isinstance(layer, nn.Conv2d):
        flag_sim_selection_mismatch(layer_inp, layer, selection, name)
    del layer_inp

    # Compute the per-layer unique-count matrix ONCE here (single process) and
    # cache it next to the simulated inputs, so every per-filter SLURM job loads
    # it instead of recomputing (it depends only on the simulated inputs, which
    # are shared across all filters of the layer).
    sim_input_dir = os.path.join(ctx.sim_base_dir, str(ctx.numTraces))
    sim_inputs = load_layer_inputs(sim_input_dir, name)
    sim_ucm = compute_unique_count_matrix(sim_inputs)
    save_unique_count_matrix(sim_input_dir, name, sim_ucm, ctx.numTraces)
    if ctx.save_unique_counts:
        _save_unique_count_diag(ctx, name, selection, sim_ucm)

    submit_slurm_jobs(
        name, ctx.numTraces, num_filters, batch_size=64, num_parallel=32,
        noise_level=ctx.noise_level, leakage_model=ctx.leakage_model,
        exp_name=ctx.exp_name, bsInpsPath=ctx.bsInpsPath,
        script_name="extractFilterSignAmbiguousMultiPatchesOptimized")
    sys.stdout.flush()


def _load_recovered_and_groundtruth(ctx, name, num_filters):
    """Recovered (ambiguous-sign) filters and the ground-truth filters."""
    recovered = load_extracted_filters(name, ctx.numTraces, num_filters,
                                       base_directory=ctx.output_base)
    groundtruth = load_extracted_filters(name, 'model_weights', num_filters,
                                         base_directory='./')
    return recovered, groundtruth


def _load_true_filter(oracle_name, idx):
    with open(f'./model_weights/{oracle_name}/filter_{idx}.pkl', 'rb') as f:
        return pickle.load(f)


def _correct_signs_with_conv_oracle(ctx, layer, layer_name, oracle_layer, oracle_name,
                                    ambiguous_filters):
    """Sign-correct `layer` using the downstream Conv2d oracle (unconditional
    head-to-head sweep). The oracle's inputs are recomputed inside via a partial
    forward from layer_i's cached input, so we no longer materialise them here."""
    # Augment the ORACLE's selection/baseline BEFORE loading it, so the oracle
    # uses each weight's true simulated-best patch (and oracle_selection stays
    # FIXED across the per-flip recomputes). The corrector recomputes the oracle
    # sim in RAM over this selection, so no oracle sim memmap is needed.
    oracle_sim_inp = capture_layer_input(ctx.model, ctx.data_loader, oracle_layer)
    real_oracle_inp = capture_layer_input(
        ctx.real_model(), ctx.data_loader, ctx.real_layer(oracle_name))
    augment_selection_for_sim_mismatch(
        oracle_name, ctx.numTraces, ctx.real_layer(oracle_name), oracle_name,
        sim_layer_inp=oracle_sim_inp, real_layer_inp=real_oracle_inp,
        baseline_base_dir=ctx.baseline_base_dir, sim_base_dir=ctx.sim_base_dir,
        write_sim=False)
    del oracle_sim_inp, real_oracle_inp

    baseline_inputs = load_layer_inputs(ctx.baseline_input_dir, oracle_name)
    oracle_selection = load_selection(ctx.baseline_base_dir, ctx.numTraces, oracle_name)
    true_g_weights = _load_true_filter(oracle_name, ctx.g_idx)
    return correct_layer_signs(
        model=ctx.model, layer_i=layer, layer_i_name=layer_name,
        layer_i_plus_1=oracle_layer, layer_i_plus_1_name=oracle_name,
        ambiguous_layer_i_filters=ambiguous_filters,
        layer_i_plus_1_baseline_inputs=baseline_inputs,
        oracle_selection=oracle_selection,
        true_g_weights=true_g_weights,
        input_tensor=ctx.data_loader,
        leakage_model=ctx.leakage_model, noise_level=ctx.noise_level,
        g_idx=ctx.g_idx,
        layer_i_output_dir=ctx.output_dir(layer_name))


def _correct_signs_with_fc_oracle(ctx, layer, layer_name, oracle_layer, oracle_name,
                                  ambiguous_filters):
    """Sign-correct `layer` using the downstream Linear oracle (head-to-head sweep)."""
    baseline_inputs = load_layer_inputs(ctx.baseline_input_dir, oracle_name)
    g_indices = list(range(min(ctx.fc_num_probes, oracle_layer.out_features)))
    true_fc_weights_by_g = {g: _load_true_filter(oracle_name, g) for g in g_indices}
    return correct_layer_signs_via_fc(
        model=ctx.model, layer_i=layer, layer_i_name=layer_name,
        fc_layer=oracle_layer, fc_layer_name=oracle_name,
        ambiguous_layer_i_filters=ambiguous_filters,
        fc_baseline_inputs=baseline_inputs,
        true_fc_weights_by_g=true_fc_weights_by_g,
        input_tensor=ctx.data_loader,
        leakage_model=ctx.leakage_model, noise_level=ctx.noise_level,
        g_indices=g_indices,
        layer_i_output_dir=ctx.output_dir(layer_name))


def _correct_terminal_layer_signs(ctx, layer, layer_name, last_filters):
    """Sign-correct the terminal Linear (no oracle) via per-class accuracy."""
    return correct_last_layer_signs_by_class_accuracy(
        model=ctx.model, last_layer=layer, last_layer_filters=last_filters,
        test_loader=ctx.test_loader, device=device,
        layer_i_output_dir=ctx.output_dir(layer_name))


def _sign_correct_layer(ctx, name, layer, recovered_filters, layers,
                        layer_names_in_order):
    """Dispatch sign correction for one layer to the right corrector and return
    the post-correction filters (reshaped to the layer's weight shape)."""
    oracle_name = _find_oracle_layer_name(layer_names_in_order, layers, name)

    if oracle_name is not None:
        oracle_layer = layers[oracle_name]
        print(f"[sign-correct] {name}: oracle = {oracle_name}")
        sys.stdout.flush()
        ambiguous_filters = recovered_filters.reshape(layer.weight.data.shape)
        if isinstance(oracle_layer, nn.Conv2d):
            result = _correct_signs_with_conv_oracle(
                ctx, layer, name, oracle_layer, oracle_name, ambiguous_filters)
        else:
            result = _correct_signs_with_fc_oracle(
                ctx, layer, name, oracle_layer, oracle_name, ambiguous_filters)
        print(f"[sign-correct] {name}: flipped = {result['flipped_filters']}, "
              f"reverted = {result['reverted_filters']}")
        sys.stdout.flush()
        return ambiguous_filters

    if isinstance(layer, nn.Linear) and ctx.test_loader is not None:
        last_filters = recovered_filters.reshape(layer.weight.data.shape)
        result = _correct_terminal_layer_signs(ctx, layer, name, last_filters)
        print(f"[sign-correct] {name}: last-layer flipped classes = "
              f"{result['flipped_classes']} (acc {result['acc_before']:.2f} "
              f"-> {result['acc_after']:.2f})")
        sys.stdout.flush()
        return last_filters

    print(f"[sign-correct] {name}: no oracle successor, skipping")
    sys.stdout.flush()
    return recovered_filters


# =========================================================
# Orchestrator
# =========================================================

def weightsRetrievalSimulationSignAmbiguous(
    model: nn.Module, data_loader, numTraces: int, noise_level: int,
    leakage_model: str, exp_name: str, bsInpsPath: str,
    g_idx: int = 0, fc_num_probes: int = 8,
    start_at_layer: str = None, test_loader=None, save_unique_counts: bool = False,
) -> list:
    """Recover every conv/linear layer in order: extract (sign-ambiguous) ->
    load -> sign-correct against the downstream oracle -> evaluate -> propagate.
    Each step is one helper above; this function only sequences them."""
    ctx = SignAmbiguousRunContext(
        model=model, data_loader=data_loader, numTraces=numTraces,
        noise_level=noise_level, leakage_model=leakage_model, exp_name=exp_name,
        bsInpsPath=bsInpsPath, g_idx=g_idx, fc_num_probes=fc_num_probes,
        test_loader=test_loader, save_unique_counts=save_unique_counts)

    layers, layer_names_in_order = _collect_conv_linear_layers(model)
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

        _generate_sim_inputs_and_submit(ctx, name, layer, num_filters)

        recovered, groundtruth = _load_recovered_and_groundtruth(ctx, name, num_filters)
        # Load ambiguous-sign filters so sign correction can forward-pass / read weights.
        updateLayerWeights(layer, recovered)

        corrected = _sign_correct_layer(
            ctx, name, layer, recovered, layers, layer_names_in_order)

        accRes.append(evaluate_extraction(groundtruth, corrected))
        updateLayerWeights(layer, corrected)
        sys.stdout.flush()

    print(f"Total Time Taken = {((time.time() - startTimeGlobal) / 60):.2f} mins")
    return accRes, model


def runWeightsRetrievalSimulationSignAmbiguous(
    model, data_loader, numTraces,
    noise_level=0, leakage_model='HW', exp_name=None, bsInpsPath=None,
    g_idx=0, fc_num_probes=8,
    start_at_layer=None, test_loader=None, save_unique_counts=False,
):
    """Install activation hooks, run the orchestrator, remove hooks."""
    hooks = setupHooks(model)
    try:
        result = weightsRetrievalSimulationSignAmbiguous(
            model, data_loader, numTraces,
            noise_level=noise_level, leakage_model=leakage_model,
            exp_name=exp_name, bsInpsPath=bsInpsPath,
            g_idx=g_idx, fc_num_probes=fc_num_probes,
            start_at_layer=start_at_layer, test_loader=test_loader,
            save_unique_counts=save_unique_counts)
    finally:
        removeHooks(hooks)
    return result
