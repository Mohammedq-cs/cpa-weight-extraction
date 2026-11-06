"""
In-process, single-node orchestrator. Same per-layer logic as the
multi-patch simulation.py:weightsRetrievalSimulation, but:

  * activations are captured from BOTH the frozen true_model (baseline) and the
    working_model (simulated) and wrapped in `ColumnProvider` (on-the-fly columns,
    no NFS memmaps),
  * the unchanged extractor `extract_filter_weights_multi_patch` is called per
    filter (via a pluggable executor: inline single-GPU, or the multi-GPU pool),
  * recovered weights are assembled in memory and propagated with
    `updateLayerWeights` so the next layer's simulated activation is realistic.

Nothing in the extraction math changes -- only the inputs (providers) and how the
layers are wired together.
"""

import os
import sys
import time
import pickle
import contextlib

import numpy as np
import torch
import torch.nn as nn

from modelUtils import setupHooks, removeHooks, updateLayerWeights
from patchInputs import capture_layer_input, compute_selected_patches
from cpaExtractCore import compute_unique_count_matrix
from extractFilterMultiPatchOptimized import (
    extract_filter_weights_multi_patch, checkpoint_path, save_outputs,
)
from columnProvider import ColumnProvider
from utils import calcAccBitwise, print_difference_histogram

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@contextlib.contextmanager
def filter_log(scripts_dir, name, idx):
    """Redirect the extractor's per-filter stdout to
    scripts_dir/{name}/filter_{idx}.out so the main job log stays clean
    (just per-layer histogram + accuracy + ModifiedModel)."""
    if not scripts_dir:
        yield
        return
    d = os.path.join(scripts_dir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"filter_{idx}.out"), "w") as f, contextlib.redirect_stdout(f):
        yield


def count_filters(layer):
    return layer.out_channels if isinstance(layer, nn.Conv2d) else layer.weight.data.shape[0]


def load_saved_filters(output_dir, name, num_filters):
    """If all per-filter recovered weights for this layer are on disk, load and
    stack them into (num_filters, numWeights); else None (re-extract)."""
    d = os.path.join(output_dir, name)
    paths = [os.path.join(d, f"filter_{f}.pkl") for f in range(num_filters)]
    if not all(os.path.exists(p) for p in paths):
        return None
    out = []
    for p in paths:
        with open(p, "rb") as fh:
            out.append(np.asarray(pickle.load(fh), dtype=np.float16).reshape(-1))
    return np.asarray(out, dtype=np.float16)


def load_filter_if_present(layer_dir, idx):
    """Per-filter resume: return the recovered weights for filter idx if a
    complete filter_{idx}.pkl is on disk, else None (extract it). A corrupt/half-
    written pickle (job killed mid-save) is treated as absent so it is re-extracted.
    Lets a layer that was interrupted partway (e.g. fc1 killed at the time limit)
    resume from the filters already done instead of redoing the whole layer."""
    p = os.path.join(layer_dir, f"filter_{idx}.pkl")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as fh:
            return np.asarray(pickle.load(fh), dtype=np.float16).reshape(-1)
    except Exception:
        return None


def load_or_compute_ucm(output_dir, name, layer, sim_A):
    """Selection + UCM cache: load {selection, ucm} from disk if present, else
    compute (compute_selected_patches + compute_unique_count_matrix) and save, so
    a re-run/resume never repeats the expensive single-threaded scan. Returns
    (selection, ucm, sim_prov) -- the SIM provider built once (transposed layout)."""
    cache = os.path.join(output_dir, f"{name}.ucm.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            d = pickle.load(fh)
        selection, ucm = d["selection"], d["ucm"]
        return selection, ucm, ColumnProvider(sim_A, layer, selection=selection)
    selection = (compute_selected_patches(sim_A, layer)
                 if isinstance(layer, nn.Conv2d) else None)
    sim_prov = ColumnProvider(sim_A, layer, selection=selection)
    ucm = compute_unique_count_matrix(sim_prov)
    with open(cache, "wb") as fh:
        pickle.dump({"selection": selection, "ucm": ucm}, fh)
    return selection, ucm, sim_prov


def true_filter_weights(true_layer):
    """(num_filters, numWeights) float16 -- the ground-truth weights the extractor
    correlates against (identical to the ./model_weights pickles)."""
    nf = count_filters(true_layer)
    return (true_layer.weight.data.detach().cpu().numpy()
            .astype(np.float16).reshape(nf, -1))


def evaluate_layer(true_W, extracted_W):
    acc, su = calcAccBitwise(true_W, extracted_W)
    print_difference_histogram(true_W, extracted_W)   # per-layer bitwise diff histogram
    return acc / su if su else 0.0


def threshold_for(noise_level):
    return 0.2 if float(noise_level) == 0.0 else 0.018


# ============================================================
# Inline executor: run a layer's filters sequentially in this process
# (single GPU). Used for verification / num_gpus<=1.
# ============================================================

class InlineExecutor:
    def run_layer(self, name, num_filters, sim_prov, base_prov, ucm, true_W,
                  leakage_model, noise_level, threshold, output_dir, seed,
                  checkpoint_every=80, corr_dir=None, scripts_dir=None):
        out = [None] * num_filters
        layer_dir = os.path.join(output_dir, name)
        os.makedirs(layer_dir, exist_ok=True)
        for idx in range(num_filters):
            # per-filter resume: skip filters already saved to disk
            saved = load_filter_if_present(layer_dir, idx)
            if saved is not None:
                out[idx] = saved
                continue
            # Match the original per-filter SLURM job: each filter seeds np.random
            # with the same seed (the noise realisation is identical across filters).
            np.random.seed(seed)
            ckpt = checkpoint_path(output_dir, name, idx)
            with filter_log(scripts_dir, name, idx):   # extractor chatter -> scriptsOut
                w, corr, chosen = extract_filter_weights_multi_patch(
                    simulated_inputs=sim_prov, baseline_inputs=base_prov,
                    filter_weights=true_W[idx], leakage_model=leakage_model,
                    noise_level=noise_level, threshold=threshold, ckpt_path=ckpt,
                    unique_count_matrix=ucm, checkpoint_every=checkpoint_every,
                )
            out[idx] = np.asarray(w, dtype=np.float16)
            # persist per-filter recovered weights (+ corr stats + chosen patches)
            save_outputs(output_dir, corr_dir or output_dir, name, idx, w, corr, chosen)
            if os.path.exists(ckpt):
                os.remove(ckpt)
        return out


# ============================================================
# Orchestrator
# ============================================================

def run_inproc(true_model, working_model, input_tensor, numTraces,
               noise_level, leakage_model, output_dir, seed=8,
               executor=None, layers_subset=None, checkpoint_every=80,
               quiet=False, corr_dir=None, scripts_dir=None):
    """Recover every Conv2d/Linear layer in order. Returns (accRes, working_model).

    true_model: frozen, original weights (baseline activations + true weights).
    working_model: zeroed/noised, updated per layer (simulated activations).
    """
    if executor is None:
        executor = InlineExecutor()
    threshold = threshold_for(noise_level)

    h_true = setupHooks(true_model)
    h_work = setupHooks(working_model)
    accRes = []
    t0 = time.time()
    try:
        true_layers = dict(true_model.named_modules())
        for name, layer in working_model.named_modules():
            if not isinstance(layer, (nn.Conv2d, nn.Linear)):
                continue
            if layers_subset is not None and name not in layers_subset:
                continue
            t_layer = true_layers[name]
            num_filters = count_filters(layer)

            # RESUME: if this layer's per-filter weights are already on disk
            # (complete), load them, apply, and skip re-extraction.
            saved = load_saved_filters(output_dir, name, num_filters)
            if saved is not None:
                acc = evaluate_layer(true_filter_weights(t_layer), saved)
                accRes.append(acc)
                updateLayerWeights(layer, saved)
                if not quiet:
                    print(f"{name}: loaded from disk (resumed), acc = {acc:.4f}")
                    sys.stdout.flush()
                continue

            if not quiet:
                kind = "Conv" if isinstance(layer, nn.Conv2d) else "Linear"
                print(f"{kind} layer: {name}  ({num_filters} filters, "
                      f"{tuple(layer.weight.data.shape)})")
                sys.stdout.flush()

            # Capture BOTH activations and wrap as on-the-fly providers, restricted
            # to the SELECTION = the per-weight best-patch union from the SIMULATED
            # activation (the attacker only sees the reconstructed model). The
            # baseline provider exposes those SAME patches so the measured leakage
            # is read at the patches the sim selected. The extractor only ever
            # touches these relevant patches.
            tu = time.time()
            sim_A = capture_layer_input(working_model, input_tensor, layer)
            base_A = capture_layer_input(true_model, input_tensor, t_layer)
            # selection + UCM: load from cache if present, else compute once + save.
            selection, ucm, sim_prov = load_or_compute_ucm(output_dir, name, layer, sim_A)
            base_prov = ColumnProvider(base_A, t_layer, selection=selection)
            true_W = true_filter_weights(t_layer)

            extracted = executor.run_layer(
                name, num_filters, sim_prov, base_prov, ucm, true_W,
                leakage_model, noise_level, threshold, output_dir, seed,
                checkpoint_every=checkpoint_every, corr_dir=corr_dir,
                scripts_dir=scripts_dir,
            )
            extracted_W = np.asarray(extracted, dtype=np.float16)

            acc = evaluate_layer(true_W, extracted_W)
            accRes.append(acc)
            if not quiet:
                print(f"{name}: layer acc = {acc:.4f}  "
                      f"({time.time() - tu:.1f}s, patches={sim_prov.num_patches})")
                sys.stdout.flush()

            updateLayerWeights(layer, extracted_W)
            del sim_A, base_A, sim_prov, base_prov, ucm
    finally:
        removeHooks(h_true)
        removeHooks(h_work)

    if not quiet:
        print(f"Total extraction time = {(time.time() - t0) / 60:.2f} min")
    return accRes, working_model
