"""
Compute bitwise extraction accuracy for every run under a tree, in one pass.

Replaces the one-off eval_bitacc_*.py scripts (which each cover a single sigma/case,
and one of which -- eval_bitacc_L1_N850.py -- points at a stale non-adaptive tree).
Also removes the need to hand-paste numbers into seq_noise_acc.ipynb.

Walks <root> for directories named after a target layer that contain filter_<i>.pkl,
parses (sigma, variant, leakage, dataset) out of the path, scores each against the
matching ground-truth weights, and prints a table / writes a CSV.

Usage:
    python eval_sweep.py                          # scan ./ , print table
    python eval_sweep.py --root ./run --csv out.csv
    python eval_sweep.py --layer features.0.0     # restrict to one layer
"""

import os
import re
import csv
import glob
import pickle
import argparse
import numpy as np

from utils import calcAccBitwise

LAYERS = ["features.0.0", "features.1.0.block.0.0", "conv1", "conv2"]

# dataset tag in the path -> ground-truth weights dir
GT_BY_DATASET = {
    "imageNet": "model_weights_imageNet",
    "randInps": "model_weights_random",
    "cifar":    "model_weights",
}

FILTER_RE = re.compile(r"^filter_(\d+)\.pkl$")
SIGMA_RE = re.compile(r"^N?(\d+(?:\.\d+)?)$")


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def as_array(obj):
    """Extracted filters are plain arrays; checkpoints are dicts. Skip the latter."""
    if isinstance(obj, dict):
        for k in ("weights", "extractedWeights", "filter_weights", "w"):
            if k in obj:
                return np.asarray(obj[k]).reshape(-1)
        return None
    return np.asarray(obj).reshape(-1)


def filter_files(layer_dir):
    out = []
    for p in glob.glob(os.path.join(layer_dir, "filter_*.pkl")):
        m = FILTER_RE.match(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return [p for _, p in sorted(out)]


def parse_run(rel_parts):
    """Pull (sigma, variant, leakage, dataset, traces) out of the path components."""
    sigma = variant = leakage = dataset = traces = None
    for part in rel_parts:
        m = SIGMA_RE.match(part)
        if m and sigma is None:
            sigma = float(m.group(1))
            continue
        if part in ("HW", "HD"):
            leakage = part
            continue
        if part.isdigit():
            traces = int(part)
            continue
        low = part.lower()
        if low.endswith("ouputs") or low.endswith("outputs"):  # note: repo misspells "Ouputs"
            for tag in GT_BY_DATASET:
                if part.startswith(tag) or tag.lower() in low:
                    dataset = tag
                    break
            continue
        if variant is None and part not in ("12240000",):
            variant = part
    return sigma, variant, leakage, dataset, traces


def score(layer_dir, gt_dir, layer):
    gt_files = filter_files(os.path.join(gt_dir, layer))
    ex_files = filter_files(layer_dir)
    if not gt_files or not ex_files:
        return None
    gt, ex = [], []
    for i in range(min(len(gt_files), len(ex_files))):
        g = as_array(load(gt_files[i]))
        e = as_array(load(ex_files[i]))
        if g is None or e is None:
            continue
        n = min(g.size, e.size)
        gt.append(g[:n])
        ex.append(e[:n])
    if not gt:
        return None
    gt = np.concatenate(gt).astype(np.float16)
    ex = np.concatenate(ex).astype(np.float16)
    correct, total = calcAccBitwise(gt, ex)
    return int(correct), int(total), len(ex_files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--layer", default=None, help="restrict to a single layer")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--include-corr", action="store_true",
                    help="also score *Corr* dirs (correlation stats, normally meaningless)")
    args = ap.parse_args()

    layers = [args.layer] if args.layer else LAYERS
    rows = []

    for dirpath, dirnames, _ in os.walk(args.root):
        base = os.path.basename(dirpath)
        if base not in layers:
            continue
        if not args.include_corr and "Corr" in dirpath:
            continue
        if not filter_files(dirpath):
            continue

        rel = os.path.relpath(dirpath, args.root)
        parts = rel.split(os.sep)
        if parts[0].startswith("model_weights"):
            continue  # ground truth itself

        sigma, variant, leakage, dataset, traces = parse_run(parts[:-1])
        gt_name = GT_BY_DATASET.get(dataset)
        if gt_name is None:
            continue
        gt_dir = os.path.join(args.root, gt_name)
        if not os.path.isdir(gt_dir):
            continue

        r = score(dirpath, gt_dir, base)
        if r is None:
            continue
        correct, total, nfilt = r
        rows.append({
            "sigma": sigma, "variant": variant, "leakage": leakage,
            "dataset": dataset, "traces": traces, "layer": base,
            "correct": correct, "total": total, "acc": correct / total,
            "nfilters": nfilt, "path": rel,
        })

    rows.sort(key=lambda r: (r["layer"], r["variant"] or "", r["sigma"] or 0))

    hdr = f"{'acc':>8}  {'correct':>11}  {'sig':>6}  {'variant':<12} {'lk':<3} {'dataset':<9} {'layer':<24} path"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['acc']:8.4f}  {r['correct']:5d}/{r['total']:<5d}  "
              f"{r['sigma'] if r['sigma'] is not None else -1:6.0f}  "
              f"{(r['variant'] or '?'):<12} {(r['leakage'] or '?'):<3} "
              f"{(r['dataset'] or '?'):<9} {r['layer']:<24} {r['path']}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                               ["sigma", "variant", "leakage", "dataset", "traces",
                                "layer", "correct", "total", "acc", "nfilters", "path"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
