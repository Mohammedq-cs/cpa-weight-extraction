# preprocess_collision.py
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# You already have these (kernels + helper funcs) in utilsFuncs.py
# Must provide: loadModel, layerWeightsFunc, compute_prevSum_HW,
#              hfm_kernel, popc_kernel, hd_kernel, and (likely) cp (cupy)
from utilsFuncs import *  # noqa


# -----------------------------
# Config
# -----------------------------
@dataclass
class CollisionConfig:
    model_path: str = "../model_cifar.pt"
    baseline_inputs_root: str = "../baseLineInps/model_extract/HD/cifar6m/10000"
    chkpt_dir: str = "./chkpts"

    # layer -> num_filters
    layer_numFilters: List[Tuple[str, int]] = None

    # which artifacts are required for "graphs notebook"
    required_pickles: List[str] = None

    def __post_init__(self):
        if self.layer_numFilters is None:
            self.layer_numFilters = [
                ("conv1", 32), ("conv2", 32), ("conv3", 64), ("conv4", 64),
                ("conv5", 128), ("conv6", 128), ("fc1", 128), ("fc2", 10)
            ]
        if self.required_pickles is None:
            self.required_pickles = [
                "multi_means.pkl",
                "means_intermediate_sum.pkl",
                "means_hw.pkl",
                "means_hd.pkl",
            ]


# -----------------------------
# Small IO helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def pkl_path(cfg: CollisionConfig, name: str) -> str:
    return os.path.join(cfg.chkpt_dir, name)

def save_pkl(cfg: CollisionConfig, name: str, obj: Any) -> None:
    ensure_dir(cfg.chkpt_dir)
    with open(pkl_path(cfg, name), "wb") as f:
        pickle.dump(obj, f)

def load_pkl(cfg: CollisionConfig, name: str) -> Any:
    with open(pkl_path(cfg, name), "rb") as f:
        return pickle.load(f)

def exists_all(cfg: CollisionConfig, names: List[str]) -> bool:
    return all(os.path.exists(pkl_path(cfg, n)) for n in names)


# -----------------------------
# Core math helpers (modular)
# -----------------------------
def compute_incorrect_prev_sums(
    inputs: np.ndarray,
    secondary_weights: np.ndarray,
    baseline_prevSums_per_filter: np.ndarray,
) -> np.ndarray:
    """
    Computes prev-sums for each filter using 'secondary_weights', but with chaining
    baseline prev sums for the recursive part (same as your original code).

    Shapes (typical):
      inputs: (N, W)
      secondary_weights: (F, W)
      baseline_prevSums_per_filter: (F, N, W)  [or compatible with your compute_prevSum_HW]
    Returns:
      (F, N, W)
    """
    incorrectPrevSum = []
    F = secondary_weights.shape[0]
    W = inputs.shape[1]

    for i in range(F):
        res = [
            hfm_kernel(
                cp.asarray(inputs[:, 0]),
                cp.float16(secondary_weights[i][0]),
                cp.float16(0),
            ).get()
        ]
        for j in range(1, W):
            res.append(
                hfm_kernel(
                    cp.asarray(inputs[:, j]),
                    cp.float16(secondary_weights[i][j]),
                    cp.asarray(baseline_prevSums_per_filter[i, :, j - 1]),
                ).get()
            )
        incorrectPrevSum.append(np.array(res).T)  # (N, W)

    return np.array(incorrectPrevSum)  # (F, N, W)


def max_means(diffs_plus: List[np.ndarray], diffs_minus: List[np.ndarray]) -> List[np.ndarray]:
    """
    Your logic:
      For each layer, for each weight index:
        - Take per-filter mean collision (axis=1 after mean(axis=1) etc)
        - Take max(plus, minus) runner-up
        - Average across filters
    Returns:
      List of arrays (one per layer), each array length = weight_count.
    """
    out = []
    for plus_layer, minus_layer in zip(diffs_plus, diffs_minus):
        highest_rate_per_weight = np.fmax(
            plus_layer.mean(axis=1),   # (F, W)
            minus_layer.mean(axis=1),  # (F, W)
        )
        out.append(highest_rate_per_weight.mean(axis=0))  # (W,)
    return out


def prevsum_to_hw(prevSums_layer: np.ndarray) -> np.ndarray:
    """
    Convert prev-sums values into HW values per element using popc_kernel.
    Input shape: (F, N, W)
    Output shape: (F, N, W) as ints (or whatever your kernel returns).
    """
    F = prevSums_layer.shape[0]
    hw = []
    for i in range(F):
        hw.append(popc_kernel(cp.asarray(prevSums_layer[i]).astype(cp.float16)).get())
    return np.array(hw)


def prevsum_to_hd(prevSums_layer: np.ndarray) -> np.ndarray:
    """
    Compute HD between shifted(prev) and current(prev) per element.
    Input shape: (F, N, W)
    Output shape: (F, N, W)
    """
    shifted = np.zeros_like(prevSums_layer, dtype=np.float32)
    shifted[..., 1:] = prevSums_layer[..., :-1]
    F = prevSums_layer.shape[0]
    hd = []
    for i in range(F):
        hd.append(
            hd_kernel(
                cp.asarray(shifted[i]).astype(cp.float16),
                cp.asarray(prevSums_layer[i]).astype(cp.float16),
            ).get()
        )
    return np.array(hd)


# -----------------------------
# Per-layer processing
# -----------------------------
def load_baseline_inputs(cfg: CollisionConfig, layer: str) -> np.ndarray:
    return pd.read_pickle(os.path.join(cfg.baseline_inputs_root, f"{layer}.pkl"))

def compute_baseline_prevSums(inputs: np.ndarray, correct_weights: np.ndarray, num_filters: int) -> np.ndarray:
    """
    baselinePrevsums: (F, N, W)
    """
    baselinePrevsums = []
    for i in range(num_filters):
        # compute_prevSum_HW returns (prevSums, something_else?) per your usage [0]
        baselinePrevsums.append(compute_prevSum_HW(inputs, correct_weights[i, :], leakage_model="HW")[0])
    return np.array(baselinePrevsums)

def compute_multi_values(inputs: np.ndarray, weights: np.ndarray, num_filters: int) -> np.ndarray:
    """
    This matches your "multipication value" computation:
      hfm_kernel(inputs_matrix, weights_vector, 0) per filter
    Output: (F, N) or (F, N, ?) depending on your kernel.
    Keeping exactly your call.
    """
    vals = []
    inputs_f16 = cp.asarray(inputs).astype(cp.float16)
    for i in range(num_filters):
        vals.append(
            hfm_kernel(
                inputs_f16,
                cp.asarray(weights[i, :]).astype(cp.float16),
                cp.float16(0),
            ).get()
        )
    return np.array(vals)

def process_one_layer(cfg: CollisionConfig, model, layer: str, num_filters: int) -> Dict[str, Any]:
    print(f"[preprocess] Layer {layer} ...")

    inputs = load_baseline_inputs(cfg, layer)
    correct = layerWeightsFunc(layer, model)

    plus_eps = np.nextafter(correct, np.float16(np.inf), dtype=np.float16)
    minus_eps = np.nextafter(correct, np.float16(-np.inf), dtype=np.float16)

    baseline_prev = compute_baseline_prevSums(inputs, correct, num_filters)  # (F, N, W)

    plus_prev = compute_incorrect_prev_sums(inputs, plus_eps, baseline_prev)
    minus_prev = compute_incorrect_prev_sums(inputs, minus_eps, baseline_prev)

    baseline_multi = compute_multi_values(inputs, correct, num_filters)
    plus_multi = compute_multi_values(inputs, plus_eps, num_filters)
    minus_multi = compute_multi_values(inputs, minus_eps, num_filters)

    return {
        "prevSums": baseline_prev,
        "pls_candidate": plus_prev,
        "minus_candidate": minus_prev,
        "baseline_multi_candidate": baseline_multi,
        "pls_multi_candidate": plus_multi,
        "minus_multi_candidate": minus_multi,
    }


# -----------------------------
# Full pipeline
# -----------------------------
def run_full_preprocessing(cfg: CollisionConfig, force: bool = False) -> None:
    """
    Produces:
      - prev_sums_ckpt.pkl, pls_candidate_chkpt.pkl, minus_candidate_chkpt.pkl
      - baseline_multi_candidate.pkl, pls_multi_candidate.pkl, minus_multi_candidate.pkl
      - means_intermediate_sum.pkl, multi_means.pkl
      - means_hw.pkl, means_hd.pkl
    """
    ensure_dir(cfg.chkpt_dir)

    if (not force) and exists_all(cfg, cfg.required_pickles):
        print("[preprocess] Required pickles already exist. Skipping preprocessing.")
        return

    model = loadModel(cfg.model_path)

    prevSums_all = []
    pls_all = []
    minus_all = []
    base_multi_all = []
    pls_multi_all = []
    minus_multi_all = []

    # ---- Layer loop ----
    for layer, nf in cfg.layer_numFilters:
        out = process_one_layer(cfg, model, layer, nf)

        prevSums_all.append(out["prevSums"])
        pls_all.append(out["pls_candidate"])
        minus_all.append(out["minus_candidate"])
        base_multi_all.append(out["baseline_multi_candidate"])
        pls_multi_all.append(out["pls_multi_candidate"])
        minus_multi_all.append(out["minus_multi_candidate"])

        # (optional) save incremental checkpoints to survive long runs
        save_pkl(cfg, "prev_sums_ckpt.pkl", prevSums_all)
        save_pkl(cfg, "pls_candidate_chkpt.pkl", pls_all)
        save_pkl(cfg, "minus_candidate_chkpt.pkl", minus_all)
        save_pkl(cfg, "baseline_multi_candidate.pkl", base_multi_all)
        save_pkl(cfg, "pls_multi_candidate.pkl", pls_multi_all)
        save_pkl(cfg, "minus_multi_candidate.pkl", minus_multi_all)

    # ---- Value collision rates ----
    plus_eq_val = [np.array(prevSums_all[i]) == np.array(pls_all[i]) for i in range(len(prevSums_all))]
    minus_eq_val = [np.array(prevSums_all[i]) == np.array(minus_all[i]) for i in range(len(prevSums_all))]
    
    save_pkl(cfg, "eq_prevsum_value_plus.pkl",  plus_eq_val)
    save_pkl(cfg, "eq_prevsum_value_minus.pkl", minus_eq_val)
    
    means_val = max_means(plus_eq_val, minus_eq_val)
    save_pkl(cfg, "means_intermediate_sum.pkl", means_val)

    # ---- Multiplication collision rates ----
    plus_eq_multi = [np.array(base_multi_all[i]) == np.array(pls_multi_all[i]) for i in range(len(base_multi_all))]
    minus_eq_multi = [np.array(base_multi_all[i]) == np.array(minus_multi_all[i]) for i in range(len(base_multi_all))]
    save_pkl(cfg, "eq_mult_value_plus.pkl",  plus_eq_multi)
    save_pkl(cfg, "eq_mult_value_minus.pkl", minus_eq_multi)
    means_multi = max_means(plus_eq_multi, minus_eq_multi)
    save_pkl(cfg, "multi_means.pkl", means_multi)

    # ---- HW collision rates (on prev sums) ----
    prev_hw_all = [prevsum_to_hw(np.array(mat)) for mat in prevSums_all]
    pls_hw_all = [prevsum_to_hw(np.array(mat)) for mat in pls_all]
    minus_hw_all = [prevsum_to_hw(np.array(mat)) for mat in minus_all]

    plus_eq_hw = [prev_hw_all[i] == pls_hw_all[i] for i in range(len(prev_hw_all))]
    minus_eq_hw = [prev_hw_all[i] == minus_hw_all[i] for i in range(len(prev_hw_all))]
    save_pkl(cfg, "eq_prevsum_hw_plus.pkl",  plus_eq_hw)
    save_pkl(cfg, "eq_prevsum_hw_minus.pkl", minus_eq_hw)
    means_hw = max_means(plus_eq_hw, minus_eq_hw)
    save_pkl(cfg, "means_hw.pkl", means_hw)

    # ---- HD collision rates (on prev sums) ----
    prev_hd_all = [prevsum_to_hd(np.array(mat)) for mat in prevSums_all]

    # compute HD for plus/minus candidates using *their* matrices, but shifted baseline (same convention)
    def cand_to_hd(prev_base: np.ndarray, cand: np.ndarray) -> np.ndarray:
        shifted = np.zeros_like(prev_base, dtype=np.float32)
        shifted[..., 1:] = prev_base[..., :-1]
        F = cand.shape[0]
        hd_list = []
        for i in range(F):
            hd_list.append(
                hd_kernel(
                    cp.asarray(shifted[i]).astype(cp.float16),
                    cp.asarray(cand[i]).astype(cp.float16),
                ).get()
            )
        return np.array(hd_list)

    pls_hd_all = [cand_to_hd(prevSums_all[i], pls_all[i]) for i in range(len(prevSums_all))]
    minus_hd_all = [cand_to_hd(prevSums_all[i], minus_all[i]) for i in range(len(prevSums_all))]

    plus_eq_hd = [prev_hd_all[i] == pls_hd_all[i] for i in range(len(prev_hd_all))]
    minus_eq_hd = [prev_hd_all[i] == minus_hd_all[i] for i in range(len(prev_hd_all))]
    save_pkl(cfg, "eq_prevsum_hd_plus.pkl",  plus_eq_hd)
    save_pkl(cfg, "eq_prevsum_hd_minus.pkl", minus_eq_hd)
    means_hd = max_means(plus_eq_hd, minus_eq_hd)
    save_pkl(cfg, "means_hd.pkl", means_hd)

    print("[preprocess] Done. Saved:")
    for n in [
        "means_intermediate_sum.pkl", "multi_means.pkl",
        "means_hw.pkl", "means_hd.pkl",
        "prev_sums_ckpt.pkl", "pls_candidate_chkpt.pkl", "minus_candidate_chkpt.pkl",
        "baseline_multi_candidate.pkl", "pls_multi_candidate.pkl", "minus_multi_candidate.pkl",
    ]:
        print("  -", pkl_path(cfg, n))


def ensure_required_pickles(cfg: CollisionConfig, force: bool = False) -> None:
    """
    This is the function your notebook should call.
    If the 4 required pickles exist -> do nothing.
    Else -> run preprocessing.
    """
    if force or (not exists_all(cfg, cfg.required_pickles)):
        run_full_preprocessing(cfg, force=force)
    else:
        print("[preprocess] Found required pickles. Notebook can plot immediately.")

def main():
    """
    Runs preprocessing only when this file is executed directly:
        python preprocess_collision.py

    By default:
      - If required pickles exist -> do nothing
      - Else -> run preprocessing
    """
    cfg = CollisionConfig(
        chkpt_dir="./chkpts",
        model_path="../model_cifar.pt",
        baseline_inputs_root="../baseLineInps/model_extract/HD/cifar6m/10000",
    )
    ensure_required_pickles(cfg, force=False)


if __name__ == "__main__":
    main()