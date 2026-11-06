"""
Score the Figure 19 sigma sweep FROM THE REPRODUCED DATA and save it as a CSV
(tools/data/fig19_data.csv); make_plots.py draws the figures from that CSV.

Every number is computed from the filter_*.pkl files (the original seq_noise_acc.ipynb
hard-coded both accuracy arrays as literals, so the figures could drift from the
extraction outputs without anyone noticing).

Refuses to write a sigma point that does not have all 32 filters -- a partial average
looks plausible and is wrong (the CIFAR run scored 0.3751 on 31/32 vs 0.3831 on 32/32).

Usage:
    python make_plot_data.py --root run          # -> tools/data/fig19_data.csv
"""

import os
import re
import csv
import glob
import pickle
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIGMAS = [0, 100, 200, 300, 400, 500, 600, 700, 850]
NFILT = 32
COLUMNS = ["sigma", "seq_correct", "seq_total", "seq_acc", "seq_nfilters",
           "gt_correct", "gt_total", "gt_acc", "gt_nfilters"]


def acc(out_dir, gt_dir, layer, expect=NFILT):
    """Bitwise accuracy over a layer. Returns (acc, nfilters, correct, total)."""
    d = os.path.join(out_dir, layer)
    fs = [p for p in glob.glob(os.path.join(d, "filter_*.pkl"))
          if re.match(r"filter_\d+\.pkl$", os.path.basename(p))]
    c = t = 0
    for p in fs:
        i = int(re.search(r"filter_(\d+)", p).group(1))
        e = np.asarray(pickle.load(open(p, "rb"))).astype(np.float16).ravel()
        g = np.asarray(pickle.load(open(os.path.join(gt_dir, layer, f"filter_{i}.pkl"), "rb"))
                       ).astype(np.float16).ravel()
        n = min(e.size, g.size)
        c += int((e[:n].view(np.uint16) == g[:n].view(np.uint16)).sum())
        t += n
    return (c / t if t else 0.0), len(fs), c, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--csv", default=os.path.join(HERE, "data", "fig19_data.csv"))
    args = ap.parse_args()

    gt = os.path.join(args.root, "model_weights_imageNet")
    L1 = "features.0.0"
    rows, incomplete = [], []

    for s in SIGMAS:
        a_s, n_s, c_s, t_s = acc(
            f"{args.root}/N{s}.0/seed8_mp/HW/imageNet12m_mpOuputs/12240000", gt, L1)
        a_b, n_b, c_b, t_b = acc(
            f"{args.root}/N{s}.0/BS/HW/imageNet12m_mpOuputs/12240000", gt, L1)
        if n_s != NFILT or n_b != NFILT:
            incomplete.append((s, n_s, n_b))
        rows.append((s, c_s, t_s, a_s, n_s, c_b, t_b, a_b, n_b))

    if incomplete:
        for s, n_s, n_b in incomplete:
            print(f"REFUSING TO WRITE: sigma={s} has seq={n_s}/32 BS={n_b}/32 filters")
        raise SystemExit("incomplete sigma points -- rerun the missing filters first")

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows([[repr(v) if isinstance(v, float) else v for v in r] for r in rows])
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
