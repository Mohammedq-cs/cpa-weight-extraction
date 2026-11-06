"""
Post-processing for the paper figures: raw experiment results -> tidy CSVs.

Every notebook in this folder only reads the CSVs written here (./data/<name>.csv)
and plots them; all result parsing lives in this script.

    python postprocess.py                      # all figures
    python postprocess.py --only fig_data_vs_layers
    python postprocess.py --root /path/to/results_parent   # results tree laid out like ../

`--root` is the folder holding results/ as the launchers write it:

    results/{arch}/sweep_bnprefix/K*_N*.json                         (run_sweep.sbatch)
    results/{arch}/pgd_transfer_sources_eps5255_fp32_seed{S}/*.json  (run_pgd.sbatch)

It defaults to the parent folder, i.e. the layout produced by the run_*.sbatch
scripts in ../.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# (arch, display label, conv/linear layers) -- plotting order and colours follow this.
ARCHS = [('resnet18', 'ResNet-18', 21), ('vgg11bn', 'VGG-11', 11), ('cnn8', 'CNN', 8)]
TOL_DATA = 1.1            # labeled images: first epoch with test acc >= victim - 1.1
SEEDS = (8, 9, 10)        # training seeds of the attacked surrogates; the figure plots the median


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def save_csv(rows, out, name, columns):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f'{name}.csv')
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    print(f'[postprocess] wrote {path} ({len(rows)} rows)')


def layers_finetuned(k, n_layers):
    """x axis: K = leading layers kept frozen; K = -1 (scratch) trains every layer."""
    return n_layers if k == -1 else n_layers - k


# --- fig_data_vs_layers ------------------------------------------------------
def min_labeled_images(sweep_dir, tol):
    """-> ({K: smallest N whose cell reached victim - tol, or None}, victim acc).

    A cell 'reaches' at the first evaluated epoch with test acc >= victim - tol."""
    cells = []
    for path in glob.glob(os.path.join(sweep_dir, 'K*_N*.json')):
        with open(path) as f:
            c = json.load(f)
        if c.get('source', 'cifar10') == 'cifar10':
            cells.append(c)
    if not cells:
        raise SystemExit(f"no cifar10 cells in {sweep_dir}")
    victim = cells[0]['target']
    threshold = victim - tol
    images = defaultdict(list)
    for c in cells:
        k = c['num_recovered']
        images[k]   # every depth gets an entry, reached or not
        if any(h.get('test_acc') is not None and h['test_acc'] >= threshold
               for h in c['history']):
            images[k].append(c['train_size'])
    return {k: (min(v) if v else None) for k, v in images.items()}, victim


def fig_data_vs_layers(root, out):
    rows = []
    for arch, label, n_layers in ARCHS:
        images, victim = min_labeled_images(
            os.path.join(root, 'results', arch, 'sweep_bnprefix'), TOL_DATA)
        # Fine-tuned depths first (K > 0; K = 0 is the scratch model under the
        # BN-prefix rule, so it is not drawn twice), then the scratch surrogate.
        ks = [k for k in images if k > 0] + ([-1] if -1 in images else [])
        for k in ks:
            rows.append((arch, label, n_layers, k, layers_finetuned(k, n_layers),
                         k == -1, images[k], victim, round(victim - TOL_DATA, 2)))
    save_csv(rows, out, 'fig_data_vs_layers',
             ['arch', 'label', 'n_layers', 'num_recovered', 'layers_finetuned',
              'is_surrogate', 'labeled_images', 'victim_acc', 'threshold_acc'])


# --- fig_transfer_vs_layers_archs_eps5255 ------------------------------------
def fig_transfer_vs_layers_archs_eps5255(root, out):
    rows = []
    for arch, label, n_layers in ARCHS:
        per_k, white = defaultdict(dict), None
        for seed in SEEDS:
            d = os.path.join(root, 'results', arch, f'pgd_transfer_sources_eps5255_fp32_seed{seed}')
            for path in glob.glob(os.path.join(d, '*_seed*.json')):
                with open(path) as f:
                    e = json.load(f)
                if e['source_kind'] == 'victim':
                    white = white if white is not None else round(1.0 - e['target_adv_acc'], 6)
                else:
                    # target_adv_acc is a count / 10000: round off the 1 - x float noise
                    per_k[e['num_recovered']][seed] = (round(1.0 - e['target_adv_acc'], 6), e['epoch'],
                                                       e['ckpt_test_acc'], e['train_size'])
        for k in sorted(per_k, key=lambda k: layers_finetuned(k, n_layers)):
            runs = per_k[k]
            asrs = sorted(r[0] for r in runs.values())
            m = len(asrs)
            median = asrs[(m - 1) // 2] if m % 2 else round((asrs[m // 2 - 1] + asrs[m // 2]) / 2, 6)
            rows.append((arch, label, n_layers, k, layers_finetuned(k, n_layers), k == -1,
                         runs[SEEDS[0]][3] if SEEDS[0] in runs else None,
                         *[runs[s][0] if s in runs else None for s in SEEDS],
                         *[runs[s][2] if s in runs else None for s in SEEDS],
                         m, median, white))
    save_csv(rows, out, 'fig_transfer_vs_layers_archs_eps5255',
             ['arch', 'label', 'n_layers', 'num_recovered', 'layers_finetuned', 'is_surrogate',
              'labeled_images']
             + [f'asr_seed{s}' for s in SEEDS] + [f'clean_acc_seed{s}' for s in SEEDS]
             + ['n_seeds', 'asr_median', 'white_box_asr'])


FIGURES = {f.__name__: f for f in [fig_data_vs_layers, fig_transfer_vs_layers_archs_eps5255]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=os.path.dirname(HERE), help='folder holding results/ (default: ..)')
    parser.add_argument('--out', default=os.path.join(HERE, 'data'), help='CSV output folder (default: ./data)')
    parser.add_argument('--only', nargs='+', choices=list(FIGURES), help='run only these notebooks\' post-processing')
    args = parser.parse_args()
    root, out = os.path.abspath(args.root), os.path.abspath(args.out)
    for name in args.only or FIGURES:
        print(f'[postprocess] {name}')
        FIGURES[name](root, out)


if __name__ == '__main__':
    main()
