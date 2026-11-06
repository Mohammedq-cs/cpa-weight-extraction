"""
demo_12m.py — ImageNet-input baseline generation for the EfficientNet CPA attack.

Same setup as demo_random.py, but instead of random-uniform inputs + randomized
weights, this uses *real ImageNet inputs* and the *pretrained* EfficientNet-B0
weights. It captures the "first activation patch" (one trace per image) for the
first two layers (features.0.0 and features.1.0.block.0.0), reaching 12.24M
traces by repeating ImageNet train (~1.28M imgs) with light augmentation, the
same way the CIFAR experiment reaches 12M from CIFAR-5M.

Like demo_random.py, this only generates the baseline inputs and saves the
ground-truth weights, then exits. The CPA attack (simInps + per-filter SLURM
jobs) is run separately.

Output layout (identical conventions to demo_random.py so the existing
extractFilterChkptOptimized.py worker can consume it later):
    ./model_weights/<layer>/filter_<i>.pkl                         (ground truth)
    ./baseLineInps/<bsInpsPath>/<numTraces>/<layer>_patch<k>.pkl   (real activations)
"""

import argparse
import gc
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

# Reuse existing helpers (unmodified).
from utils import save_model_weights
from generateFilterHW import batch_to_realInps, save_chunk
import patchInputsMaterialized as PM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# First two layers of EfficientNet-B0: first conv (in_ch=3, 3x3 -> 27 cols) and
# the first depthwise conv (in_ch=1, 3x3 -> 9 cols).
TARGET_LAYERS = ['features.0.0', 'features.1.0.block.0.0']

# ImageNet normalization (standard, matches torchvision pretrained EfficientNet-B0).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def draft_loader(path):
    """
    ImageFolder loader that uses libjpeg-turbo draft mode: decode the JPEG
    directly at a reduced DCT scale near the target (256) instead of full
    resolution, then convert. ~15-30% faster decode, no quality impact for a
    256->224 pipeline.
    """
    img = Image.open(path)
    img.draft('RGB', (256, 256))
    return img.convert('RGB')


class RepeatAugImageNet(Dataset):
    """
    Deterministic repeat-with-augmentation wrapper over ImageNet train.

    Reaches `target_size` traces from a base ImageFolder of ~1.28M images by
    cycling through it (idx % base_len). The first full pass (copy 0) uses the
    plain center-crop transform; every later copy uses a seeded augmentation so
    the repeated images are not bit-identical. Augmentation is seeded by global
    `idx`, so the same idx always yields the same sample (reproducible, and a
    future simInps pass can be aligned to this baseline).
    """

    def __init__(self, base, target_size, base_transform, aug_transform):
        self.base = base
        self.base_len = len(base)
        self.target_size = target_size
        self.base_transform = base_transform
        self.aug_transform = aug_transform

    def __len__(self):
        return self.target_size

    def __getitem__(self, idx):
        img, _ = self.base[idx % self.base_len]
        copy_id = idx // self.base_len

        if copy_id == 0:
            x = self.base_transform(img)
        else:
            # Reseed the (per-worker) global torch RNG per-idx so augmentation is
            # reproducible. Safe to clobber: every item reseeds before use.
            torch.manual_seed(idx)
            x = self.aug_transform(img)

        return x.half(), 0


def setup_multi_hooks(model, target_names):
    """Register a forward hook on each target layer, capturing its INPUT activation."""
    activations = {}
    layers = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            activations[name] = inp[0].detach()
        return hook_fn

    for name, layer in model.named_modules():
        if name in target_names and isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(make_hook(name)))
            layers[name] = layer

    missing = [n for n in target_names if n not in layers]
    if missing:
        raise ValueError(f"Target layers not found / not Conv2d|Linear: {missing}")

    return hooks, activations, layers


def extract_baseline_two_layers(model, dataloader, numTraces, bsInpsPath, chunk_size=4_500_000):
    """
    One forward pass over the whole dataloader; for each target layer, slice the
    first activation patch, buffer per layer, and flush patches to disk using the
    same chunking/naming as generateFilterHW.save_chunk.
    """
    base_dir = f"./baseLineInps/{bsInpsPath}"

    hooks, activations, layers = setup_multi_hooks(model, TARGET_LAYERS)

    buffers = {name: [] for name in TARGET_LAYERS}
    buffer_rows = {name: 0 for name in TARGET_LAYERS}
    patch_idx = {name: 0 for name in TARGET_LAYERS}

    total = len(dataloader.dataset)
    t_start = time.time()
    seen = 0
    log_every = 50  # batches

    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device).half()

            _ = model(inputs)

            seen += inputs.shape[0]
            if bi % log_every == 0:
                el = time.time() - t_start
                ips = seen / el if el > 0 else 0
                eta = (total - seen) / ips / 60 if ips > 0 else float('inf')
                print(f"[progress] batch {bi} | {seen}/{total} imgs | "
                      f"{ips:.0f} img/s | elapsed {el/60:.1f} min | ETA {eta:.1f} min",
                      flush=True)

            for name in TARGET_LAYERS:
                arr = batch_to_realInps(layers[name], activations[name])
                buffers[name].append(arr)
                buffer_rows[name] += arr.shape[0]

                if buffer_rows[name] >= chunk_size:
                    combined = np.concatenate(buffers[name], axis=0)
                    save_chunk(
                        name=name,
                        numTraces=numTraces,
                        arr=combined[:chunk_size],
                        patch_idx=patch_idx[name],
                        base_dir=base_dir,
                    )
                    patch_idx[name] += 1

                    remain = combined[chunk_size:]
                    buffers[name] = [remain] if len(remain) > 0 else []
                    buffer_rows[name] = remain.shape[0]
                    del combined, remain

            del inputs, _
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Flush remaining rows per layer.
    for name in TARGET_LAYERS:
        if buffer_rows[name] > 0:
            combined = np.concatenate(buffers[name], axis=0)
            save_chunk(
                name=name,
                numTraces=numTraces,
                arr=combined,
                patch_idx=patch_idx[name],
                base_dir=base_dir,
            )
            del combined
            gc.collect()

    for h in hooks:
        h.remove()

    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate ImageNet baseline inputs for CPA on EfficientNet-B0.")
    # Same CLI as demo_random.py so submit_experiment.sh -s demo_12m works unchanged.
    parser.add_argument("--model_save_path", required=True, help="Accepted for compatibility; unused (baseline-only).")
    parser.add_argument("--num_traces", type=int, required=True)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--leakage_model", type=str, default='HW', choices=['HW', 'HD'])
    parser.add_argument("--exper_name", type=str, default='accuracy_experiment')
    parser.add_argument("--dataset_name", type=str, default='imageNet', choices=['imageNet'])
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="${IMAGENET_ROOT:-./imagenet}")
    parser.add_argument("--target_size", type=int, default=12_240_000)
    parser.add_argument("--weights_dir", type=str, default="./model_weights",
                        help="Where to save ground-truth weights. NOTE: the CPA worker reads "
                             "./model_weights; pointing elsewhere avoids clobbering a prior run's truth.")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="DataLoader workers; set to the SLURM cpus-per-task for best decode throughput.")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--multipatch", action="store_true",
                        help="Save only the per-weight best-activation patches (union, column-major "
                             "memmaps) via patchInputsMaterialized, instead of the single top-left "
                             "patch. Depthwise layer-2 is stored per filter/channel.")
    parser.add_argument("--selection_images", type=int, default=8000,
                        help="Images used to pick each weight's best patch (multipatch only). The "
                             "argmax is stable well below 12M; held in RAM, so keep it modest "
                             "(~8k -> ~9GB; 40k would need ~44GB).")
    args = parser.parse_args()

    traces_use = int(args.num_traces)
    noise_level = args.noise_level
    leakage_model = args.leakage_model
    exper_name = args.exper_name
    dataset_name = args.dataset_name
    seed = args.seed
    data_dir = args.data_dir
    target_size = args.target_size
    weights_dir = args.weights_dir

    # -------------------------------
    # Reproducibility (verbatim from demo_random.py)
    # -------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Using deterministic CuDNN: {torch.backends.cudnn.deterministic}")

    g = torch.Generator()
    g.manual_seed(seed)

    # -------------------------------
    # Load EfficientNet-B0 with PRETRAINED ImageNet weights (NOT randomized).
    # -------------------------------
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    ).half().to(device)
    model.eval()
    print(f"Seed = {seed}")
    print(f"CuDNN enabled = {torch.backends.cudnn.enabled}")

    # Ground-truth weights (for the two target layers, among others).
    print(f"Saving ground-truth weights to {weights_dir}")
    save_model_weights(model, weights_dir)

    # -------------------------------
    # ImageNet dataset, repeated+augmented to reach target_size traces.
    # EfficientNet-B0 expects 3x224x224 input.
    # -------------------------------
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    base = datasets.ImageFolder(root=f"{data_dir}/train", loader=draft_loader)
    print(f"ImageNet train base size = {len(base)} images")

    # Cap the effective target at the requested number of traces.
    effective_target = min(target_size, traces_use)
    final_dataset = RepeatAugImageNet(base, effective_target, base_transform, aug_transform)
    print(f"final_dataset size (target traces) = {len(final_dataset)} "
          f"(~{len(final_dataset) / len(base):.2f}x over base)")

    train_loader = DataLoader(
        final_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True,
        prefetch_factor=2,
    )

    numTraces = len(train_loader.dataset)
    exp_name = f'seed{seed}/{exper_name}/{leakage_model}/{dataset_name}'
    bsInpsPath = f'{exp_name}/{dataset_name}'

    print(f"\nnumTraces = {numTraces}, noiseLevel = {noise_level}, leakageModel = {leakage_model}")
    print(f"Target layers = {TARGET_LAYERS}")
    print(f"Baseline output dir = ./baseLineInps/{bsInpsPath}/{numTraces}/")
    print(f"Storage mode = {'MULTIPATCH (best-activation, union)' if args.multipatch else 'single top-left patch'}\n")

    if args.multipatch:
        # Patch selection must see a REPRESENTATIVE sample. ImageFolder is class-
        # ordered, so the first N images are ~1-2 classes (biased). Draw a seeded
        # RANDOM cross-class subset (base center-crop transform) for selection.
        from torch.utils.data import Subset
        sel_base = datasets.ImageFolder(root=f"{data_dir}/train",
                                        transform=base_transform, loader=draft_loader)
        sel_rng = np.random.default_rng(seed)
        sel_idx = sel_rng.permutation(len(sel_base))[:args.selection_images].tolist()
        sel_loader = DataLoader(
            Subset(sel_base, sel_idx), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g,
        )
        print(f"selection sample = {len(sel_idx)} seeded cross-class images")
        PM.generate_baseline_materialized(
            model, train_loader, numTraces,
            base_dir=f"./baseLineInps/{bsInpsPath}",
            layer_names=TARGET_LAYERS,
            selection_images=args.selection_images,
            fwd_batch=args.batch_size,
            epochs=1,
            selection_source=sel_loader,
        )
    else:
        extract_baseline_two_layers(model, train_loader, numTraces, bsInpsPath)

    print("Done: baseline inputs + ground-truth weights generated. Exiting (CPA attack runs separately).")
