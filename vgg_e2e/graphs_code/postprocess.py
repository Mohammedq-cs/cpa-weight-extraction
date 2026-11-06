"""
Post-processing for the VGG-11 end-to-end figure: recovered filters on disk -> tidy CSVs.

bitwise_acc_up_to_failure.ipynb only reads the CSVs written here (./data/*.csv) and plots them.

    python postprocess.py
    python postprocess.py --root /path/to/results

`--root` is the folder the launchers write into; it must contain
model_weights/<layer>/filter_*.pkl (ground truth) and
N0.0/seed8/vgg_e2e/HD/cifar6moutput/<traces>/<layer>/filter_*.pkl (recovered weights).
A layer counts only if ALL its filters are present (the crash layer is the last fully
extracted layer; the orchestrator's partial next layer is ignored). These are the
as-extracted weights, so a few sign-corrected mid-layers may read slightly below the
logged post-correction figures -- the crash points are unchanged.
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# --- layer order + friendly names (forward order) ---
LAYER_ORDER = ['features.0', 'features.4', 'features.8', 'features.11', 'features.15',
               'features.18', 'features.22', 'features.25', 'classifier.0', 'classifier.3', 'classifier.6']
LAYER_NAMES = {'features.0': 'conv1', 'features.4': 'conv2', 'features.8': 'conv3', 'features.11': 'conv4',
               'features.15': 'conv5', 'features.18': 'conv6', 'features.22': 'conv7', 'features.25': 'conv8',
               'classifier.0': 'fc1', 'classifier.3': 'fc2', 'classifier.6': 'fc3'}
FAIL_THRESH = 0.30


def label(t):
    return f'{t//1000}k' if t >= 1000 else str(t)


def n_filters(gt_base, layer):
    """Output-filter count of a layer = number of ground-truth filter pickles."""
    return len(glob.glob(f'{gt_base}/{layer}/filter_*.pkl'))


def load_stack(base, layer, n):
    """Concatenated float16 weights for filters 0..n-1; None if any filter is missing
    (i.e. the layer was not fully extracted in this run)."""
    out = []
    for i in range(n):
        p = f'{base}/{layer}/filter_{i}.pkl'
        if not os.path.exists(p):
            return None
        with open(p, 'rb') as f:
            out.append(np.asarray(pickle.load(f), dtype=np.float16).ravel())
    return np.concatenate(out)


def bitwise_acc(gt, ext):
    """Fraction of float16 weights recovered bit-exact (== calcAccBitwise)."""
    return float((gt.astype(np.float16).view(np.uint16) == ext.astype(np.float16).view(np.uint16)).mean())


def crashed_at(data_run):
    reached = [l for l in LAYER_ORDER if l in data_run]
    bad = next((l for l in reached if data_run[l] <= FAIL_THRESH), None)
    if bad is not None:
        return LAYER_NAMES[bad]
    # no sub-threshold layer: complete if the final layer was reached, else still running
    return '-' if (reached and reached[-1] == LAYER_ORDER[-1]) else 'in progress'


def bitwise_acc_up_to_failure(root, out):
    gt_base = os.path.join(root, 'model_weights')
    rec_base = os.path.join(root, 'N0.0/seed8/vgg_e2e/HD/cifar6moutput')

    traces = sorted(int(os.path.basename(d.rstrip("/"))) for d in glob.glob(f'{rec_base}/*/'))
    print('trace dirs found:', [label(t) for t in traces])
    gt = {l: load_stack(gt_base, l, n_filters(gt_base, l)) for l in LAYER_ORDER}   # ground truth, loaded once

    rows, crash = [], []
    for t in traces:
        data_run = {}
        for layer in LAYER_ORDER:
            ext = load_stack(f'{rec_base}/{t}', layer, n_filters(gt_base, layer))   # None if not fully extracted
            if ext is not None:
                data_run[layer] = bitwise_acc(gt[layer], ext)
                rows.append((t, label(t), layer, LAYER_NAMES[layer], data_run[layer]))
        print(f'  {label(t):>4s}: {len(data_run)} full layers -> {[LAYER_NAMES[l] for l in data_run]}')
        crash.append((t, label(t), crashed_at(data_run)))

    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows, columns=['traces', 'num_traces', 'layer_key', 'layer', 'bitwise_acc']).to_csv(
        os.path.join(out, 'bitwise_acc.csv'), index=False)
    pd.DataFrame(crash, columns=['traces', 'num_traces', 'crashed_at']).to_csv(
        os.path.join(out, 'crash_layer_by_traces.csv'), index=False)
    print(f'[postprocess] wrote {out}/bitwise_acc.csv ({len(rows)} rows), {out}/crash_layer_by_traces.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=os.path.dirname(HERE), help='experiment results folder (default: ..)')
    parser.add_argument('--out', default=os.path.join(HERE, 'data'), help='CSV output folder (default: ./data)')
    args = parser.parse_args()
    bitwise_acc_up_to_failure(os.path.abspath(args.root), os.path.abspath(args.out))


if __name__ == '__main__':
    main()
