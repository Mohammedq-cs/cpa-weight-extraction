"""
gen_sim_layer2.py — generate SIMULATED inputs for layer 2 (features.1.0.block.0.0)
from the model with the EXTRACTED layer-1 weights (NOT the zeroed model, NOT the
true weights). This is the attacker's reconstruction: layer-2 activations as the
attacker would see them given their recovered features.0.0.

Materializes ONLY the patches the baseline already selected for layer 2 (reused
via selection_dir) so the sim and baseline patch lists align 1:1 for the extractor.
"""
import argparse
import os
import pickle
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader

from modelUtils import updateLayerWeights
import patchInputsMaterialized as PM
# Reuse the exact dataset/augmentation from demo_12m so traces align with baseline.
from demo_12m import RepeatAugImageNet, draft_loader, seed_worker, IMAGENET_MEAN, IMAGENET_STD
import torchvision.transforms as transforms
from torchvision import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
L1 = 'features.0.0'
L2 = 'features.1.0.block.0.0'


def load_extracted_layer(out_dir, layer, num_filters):
    """Stack per-filter recovered weights {out_dir}/{layer}/filter_i.pkl -> (nf, pfnw)."""
    rows = []
    for i in range(num_filters):
        w = np.asarray(pickle.load(open(f"{out_dir}/{layer}/filter_{i}.pkl", "rb"))).reshape(-1)
        rows.append(w.astype(np.float16))
    return np.stack(rows, axis=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Generate layer-2 sim inputs from extracted L1 weights.")
    p.add_argument("--num_traces", type=int, default=12_240_000)
    p.add_argument("--target_size", type=int, default=12_240_000)
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--data_dir", type=str, default="${IMAGENET_ROOT:-./imagenet}")
    p.add_argument("--l1_out_dir", required=True,
                   help="dir with extracted L1 weights: {l1_out_dir}/features.0.0/filter_i.pkl")
    p.add_argument("--baseline_dir", required=True,
                   help="baseline input dir (…/<numTraces>) holding the L2 selection to reuse")
    p.add_argument("--sim_base_dir", required=True,
                   help="output base dir for sim inputs (numTraces appended)")
    p.add_argument("--num_workers", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=1024)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    g = torch.Generator(); g.manual_seed(args.seed)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).half().to(device)
    model.eval()

    # Inject the EXTRACTED layer-1 weights (attacker's reconstruction).
    named = dict(model.named_modules())
    nf1 = named[L1].out_channels
    ext_l1 = load_extracted_layer(args.l1_out_dir, L1, nf1)
    updateLayerWeights(named[L1], ext_l1)
    print(f"Injected extracted L1 weights {ext_l1.shape} into {L1}")

    # Same RepeatAug ImageNet stream as the baseline (so traces align 1:1).
    base_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    base = datasets.ImageFolder(root=f"{args.data_dir}/train", loader=draft_loader)
    effective = min(args.target_size, args.num_traces)
    ds = RepeatAugImageNet(base, effective, base_transform, aug_transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, worker_init_fn=seed_worker,
                        generator=g, persistent_workers=True, prefetch_factor=2)
    numTraces = len(loader.dataset)
    print(f"numTraces={numTraces}; reusing L2 selection from {args.baseline_dir}")

    # Materialize ONLY the baseline-selected L2 patches, from the EXTRACTED-L1 model.
    PM.generate_baseline_materialized(
        model, loader, numTraces,
        base_dir=args.sim_base_dir,
        layer_names=[L2],
        fwd_batch=args.batch_size,
        epochs=1,
        selection_dir=args.baseline_dir,
    )
    print("Done: layer-2 sim inputs materialized.")
