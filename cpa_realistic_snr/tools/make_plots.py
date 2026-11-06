"""
Draw the paper's Figure 19 plots from tools/data/fig19_data.csv, plus a diff table
against the published values.

The CSV is produced from the reproduced extraction outputs by make_plot_data.py
(which refuses incomplete sigma points), so the figures never drift from the data.

Usage:
    python make_plot_data.py --root run          # score the sweep -> tools/data/fig19_data.csv
    python make_plots.py --outdir run/plots      # -> fig19_diff_table.md + *.pdf
"""

import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
SIGMAS = [0, 100, 200, 300, 400, 500, 600, 700, 850]

# published reference values, verified bit-for-bit from the pre-existing outputs
REF_SEQ = [1.0000, .9965, .9479, .9225, .8542, .7569, .5590, .4583, .3391]
REF_BS = [1.0000, .9965, .9780, .9537, .9144, .8866, .8507, .8171, .7280]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "data", "fig19_data.csv"),
                    help="written by make_plot_data.py")
    ap.add_argument("--outdir", default="./plots")
    args = ap.parse_args()

    with open(args.csv) as f:
        data = list(csv.DictReader(f))
    if [int(r["sigma"]) for r in data] != SIGMAS:
        raise SystemExit(f"{args.csv}: expected sigma points {SIGMAS} -- rerun make_plot_data.py")
    seq = [float(r["seq_acc"]) for r in data]
    bs = [float(r["gt_acc"]) for r in data]
    rows = [(s, int(r["seq_correct"]), int(r["seq_total"]), a_s, rs,
             int(r["gt_correct"]), int(r["gt_total"]), a_b, rb)
            for s, r, a_s, a_b, rs, rb in zip(SIGMAS, data, seq, bs, REF_SEQ, REF_BS)]

    os.makedirs(args.outdir, exist_ok=True)

    # ---- diff table -------------------------------------------------------
    lines = ["| sigma | sequential (repro) | reference | GT-oracle (repro) | reference | match |",
             "|---|---|---|---|---|---|"]
    allok = True
    for s, cs, ts, a_s, rs, cb, tb, a_b, rb in rows:
        ok = abs(a_s - rs) < 5e-5 and abs(a_b - rb) < 5e-5
        allok &= ok
        lines.append(f"| {s} | {cs}/{ts} = {a_s:.4f} | {rs:.4f} | "
                     f"{cb}/{tb} = {a_b:.4f} | {rb:.4f} | {'exact' if ok else 'DIFF'} |")
    lines.append("")
    lines.append(f"**{'All 18 results reproduce exactly.' if allok else 'MISMATCHES PRESENT.'}**")
    tbl = os.path.join(args.outdir, "fig19_diff_table.md")
    open(tbl, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {tbl}")

    # ---- figures (same style as the original notebook) --------------------
    sns.set(style="whitegrid")
    FONT, LW, MS = 18, 2.5, 10
    pal = sns.color_palette()
    x = np.array(SIGMAS)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, seq, marker="o", linestyle="-", linewidth=LW, markersize=MS,
            color=pal[0], label="Sequential")
    ax.set_xlabel(r"$\sigma$", fontsize=FONT)
    ax.set_ylabel("Bitwise Accuracy", fontsize=FONT)
    ax.tick_params(axis="both", labelsize=FONT - 4)
    ax.set_ylim(0, 1.02); ax.set_xticks(x)
    plt.tight_layout()
    p1 = os.path.join(args.outdir, "seq_noise_acc.pdf")
    plt.savefig(p1, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, bs, marker="s", linestyle="--", linewidth=LW, markersize=MS,
            color=pal[1], label="GT")
    ax.plot(x, seq, marker="o", linestyle="-", linewidth=LW, markersize=MS,
            color=pal[0], label="Sequential")
    ax.set_xlabel(r"$\sigma$", fontsize=FONT)
    ax.set_ylabel("Bitwise Accuracy", fontsize=FONT)
    ax.tick_params(axis="both", labelsize=FONT - 4)
    ax.set_ylim(0, 1.02); ax.set_xticks(x)
    ax.legend(loc="best", fontsize=FONT - 4, framealpha=0.95)
    plt.tight_layout()
    p2 = os.path.join(args.outdir, "seq_vs_gt_noise_acc.pdf")
    plt.savefig(p2, bbox_inches="tight"); plt.close()

    print(f"wrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()
