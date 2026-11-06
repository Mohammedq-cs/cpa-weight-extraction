"""
Post-processing for the ResNet-18 per-layer bitwise-accuracy figure: recovered filters on
disk -> tidy CSV.

resnet_e2e_bitwise_acc_squashed.ipynb only reads the CSV written here
(./data/resnet_bitwise_acc.csv) and plots it.

    python postprocess.py
    python postprocess.py --root /path/to/results --run <run dir>

`--root` is the folder demo.py writes into; it must contain
model_weights/<layer>/filter_*.pkl (ground truth). `--run` is the recovered-filter dir of
the 200k run, default N0.0/seed8/resnet_fwk/HD/cifar6moutput/200000 under --root
(run200k.sbatch).

Recovered weights are read directly from the per-filter pickles. For complete layers the
bitwise accuracy is over the whole layer; for a crash layer (only some filters present)
it is over the present filters, each vs the same-index ground-truth filter.
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_NAME = 'resnet_bitwise_acc'
COLUMNS = ['method', 'layer', 'label', 'bitwise_acc', 'n_present', 'n_total']

# Conv/Linear layers in forward (named_modules) order; FC kept last.
LAYER_ORDER = [
    'conv1',
    'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2',
    'layer2.0.conv1', 'layer2.0.conv2', 'layer2.0.downsample.0', 'layer2.1.conv1', 'layer2.1.conv2',
    'layer3.0.conv1', 'layer3.0.conv2', 'layer3.0.downsample.0', 'layer3.1.conv1', 'layer3.1.conv2',
    'layer4.0.conv1', 'layer4.0.conv2', 'layer4.0.downsample.0', 'layer4.1.conv1', 'layer4.1.conv2',
    'fc',
]
# Sequential display labels: conv1 .. conv20, then FC.
LABELS = [('FC' if l == 'fc' else f'conv{i + 1}') for i, l in enumerate(LAYER_ORDER)]


def load(p):
    with open(p, 'rb') as f:
        return np.asarray(pickle.load(f)).reshape(-1).astype(np.float16)


def present_indices(layer_dir):
    """Filter indices actually written for a layer (ignores *_chosen_patches.pkl)."""
    idx = []
    for p in glob.glob(os.path.join(layer_dir, 'filter_*.pkl')):
        b = os.path.basename(p)
        if b.endswith('_chosen_patches.pkl'):
            continue
        try:
            idx.append(int(b[len('filter_'):-len('.pkl')]))
        except ValueError:
            pass
    return sorted(idx)


def bitwise_bits(gt, rec):
    """bit-exact mask over flat float16 weights (== calcAccBitwise)."""
    a = gt.astype(np.float16).view(np.uint16)
    b = rec.astype(np.float16).view(np.uint16)
    return (a == b)


def layer_bitwise_acc(gt_dir, exp_base, layer):
    """Bitwise acc over the PRESENT filters of `layer`, each vs same-index GT.
    Returns (acc, n_present, n_total) or None if the layer is absent for this run."""
    ldir = os.path.join(exp_base, layer)
    if not os.path.isdir(ldir):
        return None
    idx = present_indices(ldir)
    if not idx:
        return None
    gt_total = len(glob.glob(os.path.join(gt_dir, layer, 'filter_*.pkl')))
    masks = []
    for i in idx:
        gt = load(os.path.join(gt_dir, layer, f'filter_{i}.pkl'))
        rec = load(os.path.join(ldir, f'filter_{i}.pkl'))
        n = min(gt.size, rec.size)
        masks.append(bitwise_bits(gt[:n], rec[:n]))
    allbits = np.concatenate(masks)
    return float(allbits.mean()), len(idx), gt_total


def run_rows(method, gt_dir, exp_base):
    rows = []
    for layer, lab in zip(LAYER_ORDER, LABELS):
        r = layer_bitwise_acc(gt_dir, exp_base, layer)
        acc, npres, ntot = r if r is not None else (np.nan, 0, 0)
        rows.append((method, layer, lab, acc, npres, ntot))
        print(f'  {method:12s} {layer:22s} ' + ('--' if r is None else f'{acc:.4f} ({npres}/{ntot})'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=os.path.dirname(HERE), help='experiment results folder (default: ..)')
    parser.add_argument('--run', default='N0.0/seed8/resnet_fwk/HD/cifar6moutput/200000',
                        help='recovered-filter dir of the run, relative to --root or absolute')
    parser.add_argument('--out', default=os.path.join(HERE, 'data'), help='CSV output folder (default: ./data)')
    args = parser.parse_args()
    root, out = os.path.abspath(args.root), os.path.abspath(args.out)
    gt_dir = os.path.join(root, 'model_weights')
    csv = os.path.join(out, f'{CSV_NAME}.csv')

    rows = run_rows('Adaptive RF', gt_dir, os.path.join(root, args.run))

    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(csv, index=False)
    print(f'[postprocess] wrote {csv} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
