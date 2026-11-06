"""
Diff a from-scratch re-run against (a) the paper's published values and (b) the
surviving reference tree, so "did it reproduce?" is one command.

PAPER   = the number printed in section 8.1 of the paper
REFERENCE = recomputed bit-for-bit from the original outputs
            (these matched the paper exactly, which is what pinned each result
             to a specific directory in the first place)
REPRO   = scored live from this re-run's outputs

Usage:
    python compare_to_paper.py                     # default roots
    python compare_to_paper.py --repro . --tol 0.005
"""

import os
import sys
import argparse

# the extraction modules live one level up (this file is in tools/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval_sweep import score  # reuse the scorer, don't reimplement it

# Optional: a previous run tree to diff against ("reference"). Not required --
# the PAPER column below is what a fresh reproduction should be judged on.
EFF_REF = os.environ.get("EFF_REF", "")
CIF_REF = os.environ.get("CIF_REF", "")

SIGMAS = [0, 100, 200, 300, 400, 500, 600, 700, 850]

# paper value, reference value, and how to locate the result in each tree.
# rel_repro is relative to the repro root; rel_ref to the reference tree.
SEQ_REF = [1.0000, .9965, .9479, .9225, .8542, .7569, .5590, .4583, .3391]
BS_REF = [1.0000, .9965, .9780, .9537, .9144, .8866, .8507, .8171, .7280]

# the reference sweep is split across int- and float-named dirs; try both
SEQ_REF_DIRS = ["N{s}/seed8_mp/HW/imageNet12m_mpOuputs/12240000",
                "N{s}.0/seed8_mp/HW/imageNet12m_mpOuputs/12240000"]
BS_REF_DIRS = ["N{s}.0/BS/HW/imageNet12m_mpOuputs/12240000",
               "N{s}/BS/HW/imageNet12m_mpOuputs/12240000"]


def first_existing(root, templates, sigma, layer):
    for t in templates:
        for s in (sigma, f"{sigma}.0" if "." not in t.split("{s}")[1][:2] else sigma):
            p = os.path.join(root, t.format(s=sigma))
            if os.path.isdir(os.path.join(p, layer)):
                return p
    return None


def build_cases():
    cases = []
    L1, L2, C1 = "features.0.0", "features.1.0.block.0.0", "conv1"

    for i, s in enumerate(SIGMAS):
        cases.append(dict(
            name=f"EffNet L1 seq  sigma={s}", layer=L1, gt="model_weights_imageNet",
            paper=SEQ_REF[i] if s == 850 else None, ref=SEQ_REF[i],
            repro=f"N{s}.0/seed8_mp/HW/imageNet12m_mpOuputs/12240000",
            ref_tmpl=SEQ_REF_DIRS, ref_root=EFF_REF, sigma=s))
        cases.append(dict(
            name=f"EffNet L1 BS   sigma={s}", layer=L1, gt="model_weights_imageNet",
            paper=None, ref=BS_REF[i],
            repro=f"N{s}.0/BS/HW/imageNet12m_mpOuputs/12240000",
            ref_tmpl=BS_REF_DIRS, ref_root=EFF_REF, sigma=s))

    cases.append(dict(
        name="EffNet L2 seq  sigma=850", layer=L2, gt="model_weights_imageNet",
        paper=0.0174, ref=0.0174,
        repro="N850.0/seed8_mp/HW/imageNet12m_mpOuputs/12240000",
        ref_tmpl=["N850.0/seed8_mp/HW/imageNet12m_mpOuputs/12240000"],
        ref_root=EFF_REF, sigma=850))
    cases.append(dict(
        name="Random L1 BS   sigma=850", layer=L1, gt="model_weights_random",
        paper=0.8576, ref=0.8576,
        repro="N850.0/BS/HW/randInpsOuputs/12240000",
        ref_tmpl=["N850.0/BS/HW/randInpsOuputs/12240000"],
        ref_root=EFF_REF, sigma=850))
    cases.append(dict(
        name="Random L2 BS   sigma=850", layer=L2, gt="model_weights_random",
        paper=0.8056, ref=0.7639,   # ref = the CHANNEL-CORRECTED BSfix run
        note="paper=pre-bugfix BS; ref=BSfix (corrected). Expect ~.7639.",
        repro="N850.0/BS/HW/randInpsOuputs/12240000",
        ref_tmpl=["N850.0/BSfix/HW/randInpsOuputs/12240000"],
        ref_root=EFF_REF, sigma=850))
    cases.append(dict(
        name="CIFAR conv1 seq sigma=850", layer=C1, gt="model_weights",
        paper=0.3831, ref=0.3831,
        repro="cifar/N850.0/cifar12m_mp/HW/cifar6mOuputs/12240000",
        ref_tmpl=["N850.0/cifar12m_mp/HW/cifar6mOuputs/12240000"],
        ref_root=CIF_REF, sigma=850))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", default=".")
    ap.add_argument("--tol", type=float, default=0.0, help="abs tolerance for PASS")
    ap.add_argument("--show-missing", action="store_true")
    args = ap.parse_args()

    cases = build_cases()
    hdr = f"{'result':<26} {'paper':>7} {'ref':>7} {'repro':>8} {'d(ref)':>8}  status"
    print(hdr)
    print("-" * len(hdr))

    n_done = n_pass = 0
    for c in cases:
        gt_dir = os.path.join(args.repro, c["gt"])
        # cifar's GT lives beside its outputs
        if c["layer"] == "conv1":
            gt_dir = os.path.join(args.repro, "cifar", "model_weights")

        out = os.path.join(args.repro, c["repro"])
        layer_dir = os.path.join(out, c["layer"])
        r = score(layer_dir, gt_dir, c["layer"]) if os.path.isdir(layer_dir) else None

        if r is None:
            if args.show_missing:
                print(f"{c['name']:<26} {fmt(c['paper']):>7} {fmt(c['ref']):>7} "
                      f"{'-':>8} {'-':>8}  pending")
            continue

        n_done += 1
        acc = r[0] / r[1]
        nf = r[2]
        d = acc - c["ref"]
        ok = abs(d) <= args.tol
        n_pass += ok
        # A sigma point scored on <32 filters is NOT a result -- a dead node can
        # silently drop tasks, and averaging over the survivors looks plausible.
        expected = 32
        incomplete = nf < expected
        flag = "INCOMPLETE" if incomplete else ("PASS" if ok else "DIFF")
        print(f"{c['name']:<26} {fmt(c['paper']):>7} {fmt(c['ref']):>7} "
              f"{acc:8.4f} {d:+8.4f}  {flag}"
              + (f"  ({nf}/{expected} filters!)" if incomplete else ""))
        if incomplete:
            n_pass -= ok  # do not count an incomplete point as passing
        if c.get("note") and not ok:
            print(f"{'':<26} note: {c['note']}")

    print("-" * len(hdr))
    print(f"{n_done}/{len(cases)} results present; {n_pass}/{n_done} within tol={args.tol}"
          if n_done else "no repro results on disk yet")


def fmt(v):
    return "-" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    main()
