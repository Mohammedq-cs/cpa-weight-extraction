"""
Post-processing for the paper figures: raw experiment results -> tidy CSVs.

Every notebook in this folder only reads the CSVs written here (./data/<name>.csv)
and plots them; all model loading / GPU work lives in this script.

    python postprocess.py
    python postprocess.py --root /path/to/results --first-patch-root /path/to/first_patch_results

`--root` is the folder this pipeline's launchers write into (N0/S8/..., N8/S8/...,
model_cifar.pt, cifar-10-batches-py). `--first-patch-root` holds the single-RF
(cpa_sim_first_patch) runs the figures compare against; it defaults to
../../cpa_sim_first_patch.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import pandas as pd

from utilsFuncs import *  # noqa: F401,F403  (loadModel, layerWeightsFunc, torch, ...)
from datasets import CIFAR10_ROOT  # shared <repo>/data (../datasets.py)


def save_csv(rows, out, name, columns):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f'{name}.csv')
    rows = [tuple(x.item() if isinstance(x, np.generic) else x for x in r) for r in rows]  # float16 -> exact float
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    print(f'[postprocess] wrote {path} ({len(rows)} rows)')


def model_path(base, noise, leakage, num_trace_k):
    # cpa_sim_first_patch stores some runs under /N{n}/S8/... and some under /N{n}/... -> try both.
    fname = f"seed8ModelN{noise}{num_trace_k}k.pth"
    cands = [os.path.join(base, f"N{noise}/S8/model_extract/cifar6m/{leakage}/{num_trace_k}k/{fname}"),
             os.path.join(base, f"N{noise}/model_extract/cifar6m/{leakage}/{num_trace_k}k/{fname}")]
    for p in cands:
        if os.path.exists(p):
            return p
    return cands[0]


def sign_correct_last_layer(model, loader, threshold=10.0):
    """fc2 last-layer sign-flip fix for single-RF (cpa_sim_first_patch) models: flip fc2
    rows of classes with per-class acc < threshold, keep only if overall acc
    improves. Multi-patch models are saved already-corrected."""
    model.eval()
    last = model.fc2
    n_classes = last.weight.data.shape[0]
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device).half()).max(1)[1].cpu().numpy()
            y = y.numpy()
            for c in range(n_classes):
                m = (y == c)
                total[c] += int(m.sum())
                correct[c] += int((pred[m] == c).sum())
    per = 100.0 * correct / np.clip(total, 1, None)
    low = [c for c in range(n_classes) if per[c] < threshold]
    if not low:
        return
    before = calculate_accuracy(model, loader)
    with torch.no_grad():
        for c in low:
            last.weight.data[c] *= -1
    after = calculate_accuracy(model, loader)
    if after <= before:
        with torch.no_grad():
            for c in low:
                last.weight.data[c] *= -1


NOISE_SPECS = [  # (method, traces_k, leakage, noise levels); single-RF models get the sign fix
    ('multi_patch', 200, 'HD', [6, 8, 10, 12, 14, 16]),
    ('multi_patch', 400, 'HD', [6, 8, 10, 12, 14, 16]),
    ('first_patch', 200, 'HD', [6, 8, 10, 12, 14, 16]),
    ('first_patch', 400, 'HD', [6, 8, 10, 12, 14, 16]),
]


def noise_ext_comp(root, first_patch_root, out):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10_test_data = CIFAR10(root=CIFAR10_ROOT, train=False, transform=transform, download=True)
    loader = DataLoader(cifar10_test_data, batch_size=1024, shuffle=True)

    rows = []
    for method, k, leak_mod, noises in NOISE_SPECS:
        base = root if method == 'multi_patch' else first_patch_root
        for n in noises:
            model = loadModel(model_path(base, n, leak_mod, k))
            if method == 'first_patch':
                sign_correct_last_layer(model, loader)
            rows.append((method, leak_mod, k, n, calculate_accuracy(model, loader) / 100.0))
    save_csv(rows, out, 'noise_ext_comp', ['method', 'leakage', 'traces_k', 'sigma', 'accuracy'])


FIGURES = {f.__name__: f for f in [noise_ext_comp]}


def main():
    code_root = os.path.dirname(HERE)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=code_root, help='experiment results folder (default: ..)')
    parser.add_argument('--first-patch-root', default=os.path.join(os.path.dirname(code_root), 'cpa_sim_first_patch'),
                        help='cpa_sim_first_patch results folder (default: ../../cpa_sim_first_patch)')
    parser.add_argument('--out', default=os.path.join(HERE, 'data'), help='CSV output folder (default: ./data)')
    parser.add_argument('--only', nargs='+', choices=list(FIGURES), help='run only these notebooks\' post-processing')
    args = parser.parse_args()
    root, fp_root, out = map(os.path.abspath, (args.root, args.first_patch_root, args.out))
    for name in args.only or FIGURES:
        print(f'[postprocess] {name}')
        FIGURES[name](root, fp_root, out)


if __name__ == '__main__':
    main()
