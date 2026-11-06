"""
Sign correction for the sign-ambiguous extractor (vgg_e2e, disk-backed).

Same algorithm and policy as vgg_sign_mitigation/signCorrection.py -- recover
the global +/- sign of each filter in layer i using the next layer as a CPA
oracle (reactive head-to-head walk for a Conv2d oracle; unconditional
head-to-head sweep for a Linear oracle; per-class-accuracy flip for the terminal
Linear) -- but rebuilt on the shared primitives in `cpaExtractCore` and the
column-major memmap inputs in `patchInputs`. Concretely:

  * oracle inputs are LayerPatchInputs over column-major memmaps; nothing is
    held fully in RAM;
  * the oracle's TRUE leakage (fixed across upstream flips) is materialised once
    to a memmap and read lazily by the verify probes;
  * after an upstream flip, the oracle's simulated inputs are recomputed to a
    memmap (ping-pong between two temp slots for the Conv2d path so a committed
    candidate is never overwritten);
  * select_patch / two-phase CPA / prevSum maintenance come from cpaExtractCore,
    so the extractor and the probes stay byte-for-byte consistent.

Threat model is unchanged: true downstream weights enter ONLY the device-side
leakage measurement (true weights over real baseline inputs); the flip/verify
logic sees only simulated inputs + RECOVERED weights.
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cpaCupy import *
from utils import generate_weight_candidates, calculate_accuracy
from cpaExtractCore import (
    select_patch,
    compute_unique_count_matrix,
    update_unique_count_matrix_block,
    advance_prevsum,
    init_zero_prevsum_list,
    update_prevsum_all_patches,
    cpa_extract_at,
    compute_leakage,
)
from patchInputs import LayerPatchInputs, capture_layer_input

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Partial forward: oracle input from a CACHED layer_i input
# ============================================================
# Flipping a filter of layer_i changes layer_i's OUTPUT, never its input, so
# layer_i's input is fixed throughout layer_i's sign-correction. We cache it once
# (one full forward) and then recompute the oracle's input by running ONLY the
# sub-network from layer_i to the oracle -- skipping every layer before layer_i.
# For a deep layer_i this turns a full-network forward per flip into a 3-4 module
# forward. Both `features` and `classifier` are nn.Sequential and the layer names
# ARE the indices ("features.18" == features[18]), so the sub-network is a slice.

def _module_index(name):
    grp, idx = name.split('.')
    return grp, int(idx)


def _oracle_input_tensor(model, layer_i_name, oracle_name, x_batch):
    """Run the sub-network from layer_i to the oracle on one batch of layer_i's
    input, returning the oracle layer's input tensor for that batch."""
    gi, ii = _module_index(layer_i_name)
    go, io = _module_index(oracle_name)
    if gi == "features" and go == "features":
        return model.features[ii:io](x_batch)
    if gi == "features" and go == "classifier":
        y = model.features[ii:](x_batch)
        y = torch.flatten(y, 1)
        return model.classifier[:io](y)
    if gi == "classifier" and go == "classifier":
        return model.classifier[ii:io](x_batch)
    raise ValueError(f"unsupported layer_i={layer_i_name} oracle={oracle_name}")


def partial_oracle_inputs(model, layer_i_name, layer_i_input, oracle_name,
                          oracle_layer, selection, batch=10_000):
    """Oracle layer's input as an in-RAM LayerPatchInputs, computed by running
    only the layer_i->oracle sub-network on the cached `layer_i_input` (CPU
    tensor). Conv oracle -> unfolded patches; Linear oracle -> flat columns."""
    model.eval()
    parts = []
    with torch.no_grad():
        for b in torch.split(layer_i_input, batch, dim=0):
            y = _oracle_input_tensor(model, layer_i_name, oracle_name,
                                     b.to(device).half())
            parts.append(y.detach().cpu())
    out = torch.cat(parts, dim=0)

    if isinstance(oracle_layer, nn.Linear):
        nw = oracle_layer.in_features
        arr = out[:, :nw].numpy().astype(np.float16)             # (N, in_features)
        return LayerPatchInputs([np.ascontiguousarray(arr.T)], window=1)

    in_ch = oracle_layer.in_channels
    kH, kW = oracle_layer.kernel_size
    nw = in_ch * kH * kW
    N = out.shape[0]
    arrays = [np.empty((nw, N), dtype=np.float16) for _ in range(len(selection))]
    for ts in range(0, N, batch):
        te = min(ts + batch, N)
        u = F.unfold(out[ts:te], kernel_size=(kH, kW), dilation=oracle_layer.dilation,
                     padding=oracle_layer.padding, stride=oracle_layer.stride)
        u = u[:, :nw, :].numpy().astype(np.float16)
        for i, p_idx in enumerate(selection):
            arrays[i][:, ts:te] = u[:, :, int(p_idx)].T
    return LayerPatchInputs(arrays, window=1)


def partial_oracle_block(model, layer_i_name, layer_i_input, oracle_name,
                         oracle_layer, selection, f, batch=10_000):
    """The CONV oracle input's block for upstream filter f -- i.e. oracle input
    CHANNEL f's kH*kW columns at each selected patch -- under the current model.

    Flipping layer_i filter f changes ONLY oracle input channel f, so only this
    block of the oracle input changes. Unfolding just channel f (one input
    channel) is ~in_channels x cheaper than rebuilding all patches, and is done
    on the GPU. Returns a list of (block_size, N) arrays, one per selected patch
    (block_size = kH*kW)."""
    kH, kW = oracle_layer.kernel_size
    model.eval()
    chunks = [[] for _ in selection]
    with torch.no_grad():
        for b in torch.split(layer_i_input, batch, dim=0):
            y = _oracle_input_tensor(model, layer_i_name, oracle_name,
                                     b.to(device).half())          # (b, C, H, W)
            u = F.unfold(y[:, f:f + 1, :, :], kernel_size=(kH, kW),
                         dilation=oracle_layer.dilation,
                         padding=oracle_layer.padding, stride=oracle_layer.stride)
            u = u.cpu().numpy().astype(np.float16)                 # (b, block_size, L)
            for i, p_idx in enumerate(selection):
                chunks[i].append(u[:, :, int(p_idx)].T)            # (block_size, b)
    return [np.ascontiguousarray(np.concatenate(c, axis=1)) for c in chunks]  # (block_size, N) each


class _BlockPatchedInputs:
    """A view of `base` (a LayerPatchInputs) with logical columns [start, end)
    replaced by per-patch `block_arrays[p]` of shape (end-start, N). Lets the
    head-to-head probe evaluate the flipped sign without copying the whole oracle
    sim -- only channel f's block actually changed."""

    def __init__(self, base, start, end, block_arrays):
        self._base = base
        self._start = start
        self._end = end
        self._block = block_arrays
        self.num_patches = base.num_patches
        self.num_weights = base.num_weights
        self.num_traces = base.num_traces

    def column(self, p, w):
        if self._start <= w < self._end:
            return self._block[p][w - self._start]
        return self._base.column(p, w)


# ============================================================
# Per-filter pickle write-back
# ============================================================

def _save_filter_pickle(output_dir, filter_idx, filter_weights):
    """Overwrite a single filter's per-filter pickle so tools that reload the
    extractor's output see the sign-corrected weights (flat float16). No-op if
    output_dir is None."""
    if output_dir is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    flat = np.asarray(filter_weights).reshape(-1).astype(np.float16)
    with open(os.path.join(output_dir, f"filter_{filter_idx}.pkl"), "wb") as f:
        pickle.dump(flat, f)


class _AsyncFilterWriter:
    """Write committed-flip per-filter pickles on a background thread so the
    sweep isn't stalled by per-file disk/NFS latency. A snapshot of each filter
    is taken at submit time, so the in-place model/numpy mutations afterwards are
    safe. Call `wait()` at the END of the sweep to block until every pickle has
    landed (and to surface any write error)."""

    def __init__(self, max_workers=4):
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futs = []

    def save(self, output_dir, filter_idx, filter_weights):
        if output_dir is None:
            return
        snapshot = np.asarray(filter_weights).reshape(-1).astype(np.float16).copy()
        self._futs.append(
            self._pool.submit(_save_filter_pickle, output_dir, filter_idx, snapshot)
        )

    def wait(self):
        for fu in self._futs:
            fu.result()                 # block + re-raise any write exception
        self._pool.shutdown(wait=True)
        self._futs = []


def upstream_filter_for_weight(w_idx, kH, kW):
    return w_idx // (kH * kW)


# ============================================================
# Reactive single-filter extractor with in-place sign correction
# (Conv2d oracle)
# ============================================================

def extract_g_reactive_with_sign_correction(
    g_idx,
    true_g_weights,
    baseline_inputs,
    oracle_selection,
    ambiguous_layer_i_filters,
    layer_i,
    layer_i_name,
    layer_i_plus_1,
    layer_i_plus_1_name,
    model,
    input_tensor,
    leakage_model,
    noise_level,
    layer_i_output_dir=None,
    conf_floor=0.1,
):
    """Extract downstream filter g sequentially. UNCONDITIONALLY for every
    upstream filter f: flip f, recompute the oracle's simulated inputs (partial
    forward from the cached layer_i input), run the RELATIVE head-to-head verify
    (prevSum from RECOVERED g weights), and commit only if the flipped sign wins
    a strict majority of the block's confident taps; else revert. Mutates
    ambiguous_layer_i_filters and layer_i.weight.data in place; overwrites
    `{layer_i_output_dir}/filter_{f}.pkl` on each commit.
    """
    kH, kW = layer_i_plus_1.kernel_size
    block_size = kH * kW
    numWeights = layer_i_plus_1.in_channels * kH * kW
    num_upstream = numWeights // block_size   # = layer_i out_channels = oracle in_channels

    # Oracle leakage from the TRUE g filter over the REAL baseline inputs. It
    # does NOT change when an upstream filter is flipped, so compute it once
    # (kept in RAM -- this is the single orchestrator process).
    leakage_inputs = LayerPatchInputs(
        compute_leakage(baseline_inputs, true_g_weights, leakage_model), window=1
    )

    # Cache layer_i's input ONCE (one full forward). Flipping layer_i's filters
    # changes its OUTPUT, not its input, so the oracle's input can be recomputed
    # by running only the layer_i->oracle sub-network on this cached tensor.
    layer_i_input = capture_layer_input(model, input_tensor, layer_i)
    working_sim = partial_oracle_inputs(
        model, layer_i_name, layer_i_input, layer_i_plus_1_name,
        layer_i_plus_1, oracle_selection,
    )
    unique_count_matrix = compute_unique_count_matrix(working_sim)

    weight_candidates = generate_weight_candidates()
    extracted = []
    prevSum_list = init_zero_prevsum_list(working_sim)

    flipped = set()
    reverted = set()
    writer = _AsyncFilterWriter()   # async per-flip pickle writes; joined at end

    # UNCONDITIONAL sweep over every upstream filter f (no low-corr trigger): at
    # each f's kH*kW block, flip f, recompute the oracle input (partial forward),
    # and head-to-head verify cur vs flip; commit iff the flipped sign wins a
    # strict majority of the block's confident taps. Then extract g's block-f
    # weights under the decided sign to extend prevSum for later blocks. Checking
    # EVERY filter (not just low-corr ones) catches the false negatives a trigger
    # would miss. The probe reuses the running prevSum_list (= prevSum through the
    # block start) as its prefix, so there is no per-flip prevSum rebuild.
    for f in range(num_upstream):
        start = f * block_size
        end = start + block_size

        # --- decide f's sign via unconditional head-to-head ---
        # Flip f, then recompute ONLY oracle channel f's block columns (cheap,
        # GPU) and splice them over the working sim via a view -- no full rebuild.
        ambiguous_layer_i_filters[f] = -ambiguous_layer_i_filters[f]
        with torch.no_grad():
            layer_i.weight.data[f] = -layer_i.weight.data[f]
        flip_block = partial_oracle_block(
            model, layer_i_name, layer_i_input, layer_i_plus_1_name,
            layer_i_plus_1, oracle_selection, f,
        )
        cand_sim = _BlockPatchedInputs(working_sim, start, end, flip_block)
        cand_ucm = update_unique_count_matrix_block(
            unique_count_matrix, cand_sim, start, end,
        )
        decision = _probe_conv_block_relative(
            extracted=extracted, leakage_inputs=leakage_inputs,
            cur_sim=working_sim, cur_ucm=unique_count_matrix,
            flip_sim=cand_sim, flip_ucm=cand_ucm,
            weight_candidates=weight_candidates, leakage_model=leakage_model,
            noise_level=noise_level, start=start, end=end, conf_floor=conf_floor,
            base_prev=prevSum_list,   # prevSum through `start` (no rebuild)
        )
        commit = decision["counted"] > 0 and decision["wins_flip"] > decision["wins_cur"]
        if commit:
            flipped.add(f)
            writer.save(layer_i_output_dir, f, ambiguous_layer_i_filters[f])
            # Apply the flipped block into the working sim in place (only block f
            # changed); model + numpy already carry the flip.
            for p in range(working_sim.num_patches):
                working_sim._a[p][start:end] = flip_block[p]
            unique_count_matrix = cand_ucm
            print(
                f"[sign-correct] g={g_idx} COMMIT flip f={f}: flipped wins "
                f"{decision['wins_flip']}/{decision['counted']} confident taps "
                f"(mean margin {decision['mean_margin']:+.4f})"
            )
            sys.stdout.flush()
        else:
            ambiguous_layer_i_filters[f] = -ambiguous_layer_i_filters[f]
            with torch.no_grad():
                layer_i.weight.data[f] = -layer_i.weight.data[f]
            reverted.add(f)

        # --- extract g's block-f weights (decided sign) to extend prevSum ---
        for w_idx in range(start, end):
            best_p = select_patch(unique_count_matrix, w_idx)
            best_uc = int(unique_count_matrix[best_p, w_idx])
            input_col = np.ascontiguousarray(working_sim.column(best_p, w_idx))
            if best_uc <= 1 or np.all(input_col == 0):
                extracted.append(np.float16(0.0))
            else:
                _, recovered_w, _, _ = cpa_extract_at(
                    w_idx, input_col, leakage_inputs.column(best_p, w_idx),
                    prevSum_list[best_p], weight_candidates, leakage_model, noise_level,
                )
                extracted.append(recovered_w)
            update_prevsum_all_patches(prevSum_list, working_sim, w_idx, extracted[-1])

        if (f + 1) % 64 == 0:
            print(
                f"[sign-correct] g={g_idx} ...{f + 1}/{num_upstream} filters swept, "
                f"{len(flipped)} flips so far", flush=True,
            )

    writer.wait()   # ensure every committed flip's pickle has landed on disk

    return {
        "g_extracted_weights": np.array(extracted, dtype=np.float16),
        "flipped_filters": sorted(flipped),
        "reverted_filters": sorted(reverted),
    }


# ============================================================
# Public orchestrator (Conv2d oracle)
# ============================================================

def correct_layer_signs(
    model,
    layer_i,
    layer_i_name,
    layer_i_plus_1,
    layer_i_plus_1_name,
    ambiguous_layer_i_filters,
    layer_i_plus_1_baseline_inputs,
    oracle_selection,
    true_g_weights,
    input_tensor,
    leakage_model,
    noise_level,
    g_idx=0,
    layer_i_output_dir=None,
    conf_floor=0.1,
):
    """Correct layer i's filter signs by reactively extracting downstream filter
    g of the Conv2d layer i+1. The oracle's simulated inputs are recomputed by a
    partial forward from layer_i's cached input. Mutates ambiguous_layer_i_filters
    and layer_i.weight.data in place."""
    result = extract_g_reactive_with_sign_correction(
        g_idx=g_idx,
        true_g_weights=true_g_weights,
        baseline_inputs=layer_i_plus_1_baseline_inputs,
        oracle_selection=oracle_selection,
        ambiguous_layer_i_filters=ambiguous_layer_i_filters,
        layer_i=layer_i,
        layer_i_name=layer_i_name,
        layer_i_plus_1=layer_i_plus_1,
        layer_i_plus_1_name=layer_i_plus_1_name,
        model=model,
        input_tensor=input_tensor,
        leakage_model=leakage_model,
        noise_level=noise_level,
        layer_i_output_dir=layer_i_output_dir,
        conf_floor=conf_floor,
    )
    return {
        "corrected_layer_i_filters": ambiguous_layer_i_filters,
        "g_extracted_weights": result["g_extracted_weights"],
        "flipped_filters": result["flipped_filters"],
        "reverted_filters": result["reverted_filters"],
    }


# ============================================================
# FC oracle: unconditional head-to-head sweep
# ============================================================

def correct_layer_signs_via_fc(
    model,
    layer_i,
    layer_i_name,
    fc_layer,
    fc_layer_name,
    ambiguous_layer_i_filters,
    fc_baseline_inputs,
    true_fc_weights_by_g,
    input_tensor,
    leakage_model,
    noise_level,
    g_indices,
    layer_i_output_dir=None,
    conf_floor=0.1,
    margin_floor=None,
):
    """Correct layer_i's signs using a downstream Linear oracle via an
    UNCONDITIONAL head-to-head sweep. For EVERY upstream filter f: flip it,
    recompute the oracle input, compare current vs flipped at f's block across
    the K probe units g (prevSum from RECOVERED oracle weights, greedy-forward),
    and commit iff the flipped sign wins a strict majority of the confident
    (g, weight) votes. Mutates ambiguous_layer_i_filters and layer_i.weight.data
    in place. Threat model: true_fc_weights_by_g enters ONLY the device
    measurement (true weights over baseline inputs).
    """
    out_units = ambiguous_layer_i_filters.shape[0]
    in_features = fc_layer.in_features
    if in_features % out_units != 0:
        raise ValueError(
            f"FC in_features ({in_features}) must be a multiple of upstream "
            f"units ({out_units}) for the block mapping to be valid"
        )
    block_size = in_features // out_units
    K = len(g_indices)

    print(
        f"[fc-sign-correct] layer={fc_layer_name}: K={K}, block_size={block_size}, "
        f"conf_floor={conf_floor} -- UNCONDITIONAL head-to-head sweep over {out_units} filters"
    )
    sys.stdout.flush()

    # Device measurement: true oracle weights over REAL baseline inputs, once per
    # probe g, kept in RAM (FC has one "patch"; leak[w] = column w).
    baseline_leakages = {}
    for g in g_indices:
        baseline_leakages[g] = compute_leakage(
            fc_baseline_inputs, true_fc_weights_by_g[g], leakage_model
        )[0]  # (in_features, N) array

    weight_candidates = generate_weight_candidates()
    # Cache layer_i's input once; recompute the FC oracle input via a partial
    # forward from it (skips all layers before layer_i).
    layer_i_input = capture_layer_input(model, input_tensor, layer_i)
    cur_sim = partial_oracle_inputs(           # FIXED initial oracle input (single patch)
        model, layer_i_name, layer_i_input, fc_layer_name, fc_layer, None,
    )
    N = cur_sim.num_traces

    prev = [np.ascontiguousarray(np.zeros(N, dtype=np.float16)) for _ in g_indices]

    to_flip = []
    kept_with_signal = 0
    writer = _AsyncFilterWriter()   # async per-flip pickle writes; joined at end
    for f in range(out_units):
        start = f * block_size
        end = start + block_size

        ambiguous_layer_i_filters[f] = -ambiguous_layer_i_filters[f]
        with torch.no_grad():
            layer_i.weight.data[f] = -layer_i.weight.data[f]
        flip_sim = partial_oracle_inputs(
            model, layer_i_name, layer_i_input, fc_layer_name, fc_layer, None,
        )
        ambiguous_layer_i_filters[f] = -ambiguous_layer_i_filters[f]
        with torch.no_grad():
            layer_i.weight.data[f] = -layer_i.weight.data[f]

        counted = wins_flip = wins_cur = 0
        margin = 0.0
        cprev_by_gi = [None] * K
        fprev_by_gi = [None] * K
        for gi, g in enumerate(g_indices):
            leak = baseline_leakages[g]
            cprev = prev[gi]
            fprev = prev[gi].copy()
            for w in range(start, end):
                cur_col = np.ascontiguousarray(cur_sim.column(0, w))
                flip_col = np.ascontiguousarray(flip_sim.column(0, w))
                if np.all(cur_col == 0):
                    cc, cw = None, np.float16(0.0)
                else:
                    cc, cw, _, _ = cpa_extract_at(
                        w, cur_col, np.ascontiguousarray(leak[w]), cprev,
                        weight_candidates, leakage_model, noise_level,
                    )
                if np.all(flip_col == 0):
                    fcorr, fw = None, np.float16(0.0)
                else:
                    fcorr, fw, _, _ = cpa_extract_at(
                        w, flip_col, np.ascontiguousarray(leak[w]), fprev,
                        weight_candidates, leakage_model, noise_level,
                    )
                if cc is not None and fcorr is not None and max(cc, fcorr) >= conf_floor:
                    counted += 1
                    margin += fcorr - cc
                    if fcorr > cc:
                        wins_flip += 1
                    else:
                        wins_cur += 1
                cprev = advance_prevsum(cprev, cur_col, cw)
                fprev = advance_prevsum(fprev, flip_col, fw)
            cprev_by_gi[gi] = cprev
            fprev_by_gi[gi] = fprev

        del flip_sim  # release before next f overwrites the flip temp

        mm = (margin / counted) if counted else 0.0
        commit = counted > 0 and wins_flip > wins_cur
        if commit and margin_floor is not None:
            commit = mm > margin_floor

        # Greedy-forward carry: advance each probe's prevSum with the CHOSEN sign's
        # recovered weight so later filters are judged against corrected earlier ones.
        for gi in range(K):
            prev[gi] = fprev_by_gi[gi] if commit else cprev_by_gi[gi]

        if commit:
            to_flip.append(f)
            writer.save(layer_i_output_dir, f, -ambiguous_layer_i_filters[f])
            print(
                f"[fc-sign-correct] f={f}: COMMIT flip (flipped wins "
                f"{wins_flip}/{counted} confident g's, mean margin {mm:+.4f})"
            )
            sys.stdout.flush()
        elif counted > 0:
            kept_with_signal += 1
        if (f + 1) % 64 == 0:
            print(f"[fc-sign-correct] ...{f + 1}/{out_units} swept, {len(to_flip)} commits so far",
                  flush=True)

    writer.wait()   # ensure every committed flip's pickle has landed on disk

    committed_set = set(to_flip)
    for f in to_flip:
        ambiguous_layer_i_filters[f] = -ambiguous_layer_i_filters[f]
        with torch.no_grad():
            layer_i.weight.data[f] = -layer_i.weight.data[f]
    reverted = [f for f in range(out_units) if f not in committed_set]

    print(
        f"[fc-sign-correct] layer={fc_layer_name}: done. committed {len(to_flip)} "
        f"flips: {sorted(to_flip)} ; kept {len(reverted)} ({kept_with_signal} with signal)"
    )
    sys.stdout.flush()

    return {
        "corrected_layer_i_filters": ambiguous_layer_i_filters,
        "flipped_filters": sorted(to_flip),
        "reverted_filters": sorted(reverted),
    }


# ============================================================
# Terminal Linear layer: per-class-accuracy sign fix
# ============================================================

def correct_last_layer_signs_by_class_accuracy(
    model,
    last_layer,
    last_layer_filters,
    test_loader,
    device,
    layer_i_output_dir=None,
    acc_threshold=10.0,
):
    """Sign-correct the FINAL classification layer (no downstream oracle). Each
    output unit is a class; a flipped row tanks that class's per-class accuracy.
    Flip rows scoring below `acc_threshold`%, keep only if overall test accuracy
    improves. Uses LABELED test data as the functional signal."""
    model.eval()
    n_classes = last_layer.weight.data.shape[0]

    correct = np.zeros(n_classes, dtype=np.int64)
    total = np.zeros(n_classes, dtype=np.int64)
    with torch.no_grad():
        for inputs, labels in test_loader:
            preds = model(inputs.to(device).half()).max(1)[1].cpu().numpy()
            labels = labels.cpu().numpy()
            for c in range(n_classes):
                m = labels == c
                total[c] += int(m.sum())
                correct[c] += int((preds[m] == c).sum())
    per_class = 100.0 * correct / np.clip(total, 1, None)
    low = [c for c in range(n_classes) if per_class[c] < acc_threshold]

    acc_before = calculate_accuracy(model, test_loader)
    print(
        f"[last-layer-fix] per-class acc = {np.round(per_class, 1).tolist()}; "
        f"classes < {acc_threshold}% -> {low}; acc_before = {acc_before:.2f}"
    )
    sys.stdout.flush()

    if not low:
        return {"flipped_classes": [], "acc_before": acc_before,
                "acc_after": acc_before, "applied": False}

    with torch.no_grad():
        for c in low:
            last_layer.weight.data[c] *= -1
    acc_after = calculate_accuracy(model, test_loader)

    if acc_after > acc_before:
        for c in low:
            last_layer_filters[c] = -last_layer_filters[c]
            _save_filter_pickle(layer_i_output_dir, c, last_layer_filters[c])
        print(f"[last-layer-fix] KEPT flips {low}: acc {acc_before:.2f} -> {acc_after:.2f}")
        sys.stdout.flush()
        return {"flipped_classes": low, "acc_before": acc_before,
                "acc_after": acc_after, "applied": True}

    with torch.no_grad():
        for c in low:
            last_layer.weight.data[c] *= -1
    print(f"[last-layer-fix] flip of {low} did not help "
          f"({acc_before:.2f} -> {acc_after:.2f}); REVERTED, no change")
    sys.stdout.flush()
    return {"flipped_classes": [], "acc_before": acc_before,
            "acc_after": acc_before, "applied": False}


# ============================================================
# Post-flip head-to-head verify probe (Conv2d oracle)
# ============================================================

def _probe_conv_block_relative(
    extracted, leakage_inputs,
    cur_sim, cur_ucm,
    flip_sim, flip_ucm,
    weight_candidates, leakage_model, noise_level,
    start, end, conf_floor=0.1, base_prev=None,
):
    """Head-to-head per-weight verify for one upstream filter's block. Re-extract
    the block [start, end) under BOTH the current sign (cur_sim) and the flipped
    sign (flip_sim), prevSum through `start` built from RECOVERED g weights.
    A tap votes only if both arms are non-degenerate and max(cur, flip) >=
    conf_floor. Returns vote counts + mean margin; the caller commits iff the
    flipped sign wins a strict majority of the confident taps.

    `base_prev` (prevSum through `start`, one array per patch) can be passed in
    to skip the prefix rebuild -- the unconditional sweep passes its running
    prevSum_list, which is exactly prevSum through the block start. If None, it
    is rebuilt here from `extracted[0:start]` (standalone use)."""
    num_patches = cur_sim.num_patches
    if base_prev is None:
        base_prev = init_zero_prevsum_list(cur_sim)
        for w_idx in range(start):
            update_prevsum_all_patches(base_prev, cur_sim, w_idx, np.float16(extracted[w_idx]))
    cur_prev = [a.copy() for a in base_prev]
    flip_prev = [a.copy() for a in base_prev]

    counted = wins_flip = wins_cur = 0
    margin_sum = 0.0
    for w_idx in range(start, end):
        cp_ = select_patch(cur_ucm, w_idx)
        cur_col = np.ascontiguousarray(cur_sim.column(cp_, w_idx))
        if int(cur_ucm[cp_, w_idx]) <= 1 or np.all(cur_col == 0):
            cur_corr, cur_w = None, np.float16(0.0)
        else:
            cur_corr, cur_w, _, _ = cpa_extract_at(
                w_idx, cur_col, leakage_inputs.column(cp_, w_idx),
                cur_prev[cp_], weight_candidates, leakage_model, noise_level,
            )
        fp_ = select_patch(flip_ucm, w_idx)
        flip_col = np.ascontiguousarray(flip_sim.column(fp_, w_idx))
        if int(flip_ucm[fp_, w_idx]) <= 1 or np.all(flip_col == 0):
            flip_corr, flip_w = None, np.float16(0.0)
        else:
            flip_corr, flip_w, _, _ = cpa_extract_at(
                w_idx, flip_col, leakage_inputs.column(fp_, w_idx),
                flip_prev[fp_], weight_candidates, leakage_model, noise_level,
            )
        if cur_corr is not None and flip_corr is not None and max(cur_corr, flip_corr) >= conf_floor:
            counted += 1
            margin_sum += flip_corr - cur_corr
            if flip_corr > cur_corr:
                wins_flip += 1
            else:
                wins_cur += 1
        update_prevsum_all_patches(cur_prev, cur_sim, w_idx, cur_w)
        update_prevsum_all_patches(flip_prev, flip_sim, w_idx, flip_w)
    return {
        "counted": counted,
        "wins_flip": wins_flip,
        "wins_cur": wins_cur,
        "mean_margin": (margin_sum / counted) if counted else 0.0,
    }
