"""
How many labeled images does it take to turn a partial CPA extraction into a
surrogate as accurate as the victim?

Extraction is sequential (first layer -> ... -> last), so an attack that stops
or degrades partway leaves a recovered PREFIX. Cell (K, N):

  * keeps the first K conv/linear layers at their CPA-recovered weights, FROZEN
    (with the victim's bias for those layers and their paired BatchNorm);
  * re-initializes the remaining layers -- weight and bias, with the
    architecture's own constructor init -- and their paired BatchNorm (scale 1,
    shift 0, running stats reset; see bnPrefix.py), and trains them;
  * trains on a class-balanced subset of N labeled CIFAR-10 train images;
  * K = -1 is the from-scratch surrogate: every layer and every BatchNorm random
    and trainable, nothing given.

x = "layers fine-tuned" in the figures is n_layers - K (n_layers for scratch).

Every cell gets the same compute budget, capped both ways:

    epochs = min(--max_epochs, --max_samples / N)

SGD (momentum 0.9, Nesterov, lr 0.05, wd 5e-4), cosine schedule over the
budget, batch 128, random-crop + flip augmentation. The test accuracy history
(CIFAR-10 test split) is written to {out_dir}/{cell}.json.

Run one cell, or a whole architecture's K x N grid as a sharded SLURM array
(run_sweep.sbatch):

    python sweepRepair.py --arch resnet18 --num_recovered 19 --train_size 1000 --out_dir results/resnet18/sweep_bnprefix
    python sweepRepair.py --arch vgg11bn --shard_id 0 --num_shards 12 --out_dir results/vgg11bn/sweep_bnprefix

With --checkpoint_at_threshold the cell also saves the model at the first epoch
whose test accuracy reaches the threshold (victim - 1) and stops -- this is how
the attacked surrogates are produced (run_sources.sbatch).

Finished cells are skipped on restart, so a preempted shard resumes.
"""

import argparse
import functools
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

import archs
import bnPrefix
from common import (
    device,
    collect_conv_linear_layers,
    build_transforms,
    evaluate_fp32,
    evaluate_fp16,
    load_recovered_model,
    reinitialize_layers,
    freeze_all_but,
    train_one_epoch,
    snapshot_frozen,
    assert_frozen_unchanged,
)

# CIFAR-10 (cifar-10-batches-py/) from the repo-wide data/ folder (see the top-level README);
# CPA_DATA overrides it, --data_root overrides both.
DEFAULT_DATA_ROOT = os.path.normpath(os.environ.get(
    'CPA_DATA', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')))

_TRAIN_DS_CACHE = {}


def build_train_dataset(transform, data_root=DEFAULT_DATA_ROOT, cache=False):
    """The attacker's labeled pool: the CIFAR-10 train split."""
    if cache and data_root in _TRAIN_DS_CACHE:
        # Same object, same ordering: only the transform differs per cell.
        ds = _TRAIN_DS_CACHE[data_root]
        ds.transform = transform
        return ds
    ds = CIFAR10(root=data_root, train=True, transform=transform, download=True)
    if cache:
        _TRAIN_DS_CACHE[data_root] = ds
    return ds


def balanced_subset(dataset, size, seed):
    """A class-balanced subset of `size` images, deterministic in `seed`.

    Balanced rather than random so small budgets are not confounded by class
    imbalance -- an attacker choosing what to label would balance too.
    """
    if size >= len(dataset):
        return dataset, len(dataset)
    targets = np.asarray(dataset.targets)
    per_class = size // 10
    rng = np.random.RandomState(seed)
    picked = []
    for cls in range(10):
        cls_idx = np.where(targets == cls)[0]
        picked.extend(rng.choice(cls_idx, per_class, replace=False))
    picked = np.sort(np.asarray(picked))
    return Subset(dataset, picked.tolist()), len(picked)


def cell_name_for(num_recovered, train_size):
    return f"K{num_recovered}_N{train_size}_cifar10_bnprefix"


def save_checkpoint(model, path, dtype='fp32'):
    """Persist one checkpoint atomically (.tmp + os.replace: a preemption
    mid-save must not leave a truncated .pt for the attack to load)."""
    state = model.state_dict()
    if dtype == 'fp16':
        state = {k: (v.half() if v.is_floating_point() else v)
                 for k, v in state.items()}
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)


def build_cell_model(num_recovered, layer_order, *, arch):
    """Model for one depth cell, plus the names of the layers it will train.

    K >= 0: leading K layers frozen at recovered values, the rest re-initialized.
    K < 0:  everything random and trainable, BatchNorm included (scratch).
    """
    if num_recovered < 0:
        model = arch.builder().to(device)  # the constructor's own init
        for param in model.parameters():
            param.requires_grad = True
        trainable = list(layer_order)
        n_train = sum(p.numel() for p in model.parameters())
        print(f"From-scratch reference: all {n_train:,} params trainable "
              f"(BatchNorm randomized too)")
        return model, trainable, [p for p in model.parameters()]

    model = load_recovered_model(arch)
    trainable = list(layer_order[num_recovered:])
    if trainable:
        reinitialize_layers(model, trainable, init_layer=arch.init_layer)
    params = freeze_all_but(model, trainable) if trainable else []
    return model, trainable, params


def main():
    parser = argparse.ArgumentParser(
        description="Labeled-data cost of turning a partial CPA extraction into "
                    "a victim-accuracy surrogate (frozen recovered prefix).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", type=str, default='resnet18', choices=sorted(archs.ARCHS))
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cell_id", type=int, default=None,
                        help="Index into the architecture's K x N grid; runs one cell.")
    parser.add_argument("--shard_id", type=int, default=None,
                        help="Run every grid cell with index == shard_id (mod "
                             "--num_shards), sequentially.")
    parser.add_argument("--num_shards", type=int, default=12)
    parser.add_argument("--grid_k", type=str, default=None,
                        help="Keep only grid cells whose K is in this comma-separated "
                             "list before sharding. Default: all.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute cells whose JSON already exists.")
    parser.add_argument("--num_recovered", type=int, default=19,
                        help="K: leading layers kept at recovered values. "
                             "-1 = from-scratch surrogate.")
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--max_samples", type=int, default=5_000_000,
                        help="Compute budget in samples seen (5M = 100 epochs of 50k).")
    parser.add_argument("--max_epochs", type=int, default=300,
                        help="Hard epoch cap, so tiny subsets do not spin forever.")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="After --eval_dense_epochs, evaluate only every Nth epoch.")
    parser.add_argument("--eval_dense_epochs", type=int, default=20,
                        help="Evaluate every epoch for this many epochs first.")
    parser.add_argument("--stop_after_target", type=int, default=10,
                        help="Stop this many epochs after first reaching "
                             "--stop_threshold. Cells that never reach it run the "
                             "full budget. 0 disables.")
    parser.add_argument("--target", type=float, default=None,
                        help="Victim accuracy. Default: the architecture's own.")
    parser.add_argument("--stop_threshold", type=float, default=None,
                        help="Accuracy that arms --stop_after_target. Default: the "
                             "architecture's victim - 1 std of a 10k-sample estimate. "
                             "The figure counts the first epoch at victim - 1.1, so "
                             "nothing it uses happens after this crossing.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=8,
                        help="Draws both the labeled subset and the initialization.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_dataset", action="store_true",
                        help="Keep the train set in RAM across the cells of a shard.")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint_at_threshold", type=float, default=None,
                        help="Persist ONE checkpoint, at the first epoch whose test "
                             "accuracy is >= this value, then stop. Evaluates every "
                             "epoch. If the cell never reaches it, the highest-test-"
                             "accuracy epoch is persisted instead, flagged "
                             "selection='best_test_acc_fallback'.")
    parser.add_argument("--checkpoint_best", action="store_true",
                        help="Evaluate every epoch, run the full budget, and persist "
                             "ONE checkpoint: the highest-test-accuracy epoch.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Default: {out_dir}/checkpoints.")
    parser.add_argument("--checkpoint_dtype", choices=['fp32', 'fp16'], default='fp32')
    args = parser.parse_args()

    arch = archs.ARCHS[args.arch]
    if args.target is None:
        args.target = arch.expected_clean_acc
    if args.stop_threshold is None:
        args.stop_threshold = arch.stop_threshold
    os.makedirs(args.out_dir, exist_ok=True)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.out_dir, 'checkpoints')
    if args.checkpoint_at_threshold is not None:
        args.checkpoint_best = True   # tracked in RAM; written only as the fallback
    if args.checkpoint_best:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    archs.validate(arch)
    grid = arch.grid()
    if args.grid_k is not None:
        wanted = {int(k) for k in args.grid_k.split(',')}
        grid = [c for c in grid if c[0] in wanted]
    if args.shard_id is not None:
        cells = grid[args.shard_id::args.num_shards]
        print(f"shard {args.shard_id} of {args.num_shards}: {len(cells)} cells\n")
    elif args.cell_id is not None:
        if not 0 <= args.cell_id < len(grid):
            raise SystemExit(f"cell_id {args.cell_id} outside 0..{len(grid) - 1}")
        cells = [grid[args.cell_id]]
    else:
        cells = [(args.num_recovered, args.train_size)]

    for position, (num_recovered, train_size) in enumerate(cells, 1):
        cell_name = cell_name_for(num_recovered, train_size)
        out_path = os.path.join(args.out_dir, f'{cell_name}.json')
        if os.path.exists(out_path) and not args.force:
            print(f"[{position}/{len(cells)}] {cell_name}: already done, skipping")
            continue
        print(f"\n[{position}/{len(cells)}] starting {cell_name}")
        run_cell(num_recovered, train_size, args, out_path)


def run_cell(num_recovered, train_size, args, out_path):
    """Train one (depth, data-size) cell and write its JSON."""
    cell_name = cell_name_for(num_recovered, train_size)
    print(f"=== cell {cell_name} (recovered={num_recovered}, train_size={train_size}) ===\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    arch = archs.ARCHS[args.arch]
    train_transform, test_transform = build_transforms(arch.norm_mean, arch.norm_std)
    full_train = build_train_dataset(train_transform, args.data_root, cache=args.cache_dataset)
    test_data = CIFAR10(root=args.data_root, train=False, transform=test_transform,
                        download=True)
    train_subset, actual_size = balanced_subset(full_train, train_size, args.seed)
    train_loader = DataLoader(train_subset, batch_size=min(args.batch_size, actual_size),
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    epochs = min(args.max_epochs, max(1, math.ceil(args.max_samples / actual_size)))
    print(f"Train subset {actual_size} images, budget {epochs} epochs "
          f"({epochs * actual_size:,} samples seen)\n")

    # Building a probe model consumes the same RNG draws as the published runs,
    # so every cell's initialization reproduces them exactly.
    probe = arch.builder()
    _, layer_order = collect_conv_linear_layers(probe)
    del probe
    assert len(layer_order) == arch.n_layers

    model, trainable_names, trainable_params, reinit_bn = bnPrefix.build_source_model(
        functools.partial(build_cell_model, arch=arch), num_recovered, layer_order)
    frozen_names = [n for n in layer_order if n not in trainable_names]
    print(f"Frozen ({len(frozen_names)}): {frozen_names if frozen_names else 'none'}")
    print(f"Trainable ({len(trainable_names)}): {trainable_names}\n")

    start_acc = evaluate_fp16(model, test_loader)
    print(f"Pre-training accuracy: {start_acc:.2f}\n")
    if not trainable_params:
        raise SystemExit("Nothing trainable in this cell; there is no repair to measure.")

    # Audit a spread of this cell's frozen layers (weights, all params, and the
    # paired BatchNorm): the prefix must leave training bit-identical.
    audit_names = sorted({frozen_names[i] for i in
                          (0, len(frozen_names) // 2, len(frozen_names) - 1)}) \
        if frozen_names else []
    frozen_snapshot = snapshot_frozen(model, audit_names) if audit_names else None
    bias_snapshot = bnPrefix.snapshot_params(model, audit_names) if audit_names else None
    paired = bnPrefix.paired_bn_names(model)
    bn_audit_names = [paired[n] for n in audit_names if paired.get(n)]
    bn_snapshot = bnPrefix.snapshot_params(model, bn_audit_names) if bn_audit_names else None
    audits = ((frozen_snapshot, assert_frozen_unchanged),
              (bias_snapshot, bnPrefix.assert_params_unchanged),
              (bn_snapshot, bnPrefix.assert_params_unchanged))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=0.9,
                                nesterov=True, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_acc = 0.0
    epochs_to_target = None
    samples_to_target = None
    epochs_to_stop = None
    steps_per_epoch = math.ceil(actual_size / min(args.batch_size, actual_size))
    ckpt_records = []
    best_ckpt = None
    run_start = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

        evaluated = (args.checkpoint_best
                     or epoch < args.eval_dense_epochs
                     or (epoch + 1) % args.eval_every == 0
                     or epoch == epochs - 1)
        if not evaluated:
            history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                            'train_acc': train_acc, 'test_acc': None,
                            'samples_seen': (epoch + 1) * actual_size})
            continue
        test_acc = evaluate_fp32(model, test_loader)

        if args.checkpoint_best and test_acc > best_acc:
            best_ckpt = {'epoch': epoch + 1, 'samples_seen': (epoch + 1) * actual_size,
                         'steps': (epoch + 1) * steps_per_epoch, 'test_acc': test_acc,
                         'state': {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}}
        best_acc = max(best_acc, test_acc)
        samples_seen = (epoch + 1) * actual_size
        if epochs_to_stop is None and test_acc >= args.stop_threshold:
            epochs_to_stop = epoch + 1
        if epochs_to_target is None and test_acc >= args.target:
            epochs_to_target = epoch + 1
            samples_to_target = samples_seen
            print(f"*** reached {args.target} at epoch {epoch + 1} "
                  f"({samples_seen:,} samples seen) ***")

        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'train_acc': train_acc, 'test_acc': test_acc,
                        'samples_seen': samples_seen})
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch + 1:4d}/{epochs}  loss {train_loss:.4f}  "
                  f"train {train_acc:6.2f}  test {test_acc:6.2f}  best {best_acc:6.2f}")

        if (args.checkpoint_at_threshold is not None
                and test_acc >= args.checkpoint_at_threshold):
            for snap, check in audits:
                if snap is not None:
                    check(model, snap)
            cp = os.path.join(args.checkpoint_dir, f'{cell_name}_ep{epoch + 1:04d}.pt')
            save_checkpoint(model, cp, args.checkpoint_dtype)
            ckpt_records.append({'epoch': epoch + 1, 'samples_seen': samples_seen,
                                 'steps': (epoch + 1) * steps_per_epoch,
                                 'test_acc': test_acc, 'path': cp,
                                 'selection': 'threshold',
                                 'threshold': args.checkpoint_at_threshold})
            best_ckpt = None   # the fallback is not needed
            print(f"threshold checkpoint at epoch {epoch + 1} (test {test_acc:.2f} "
                  f">= {args.checkpoint_at_threshold}); stopping cell")
            break

        # Only ever triggers after the threshold is met, so cells that fail
        # still consume their full budget and their best_acc stays comparable.
        if (args.stop_after_target and epochs_to_stop is not None
                and (epoch + 1) - epochs_to_stop >= args.stop_after_target):
            print(f"early stop at epoch {epoch + 1}: target held for "
                  f"{args.stop_after_target} epochs")
            break

    for snap, check in audits:
        if snap is not None:
            check(model, snap)

    if best_ckpt is not None:
        cp = os.path.join(args.checkpoint_dir, f"{cell_name}_ep{best_ckpt['epoch']:04d}.pt")
        state = best_ckpt.pop('state')
        if args.checkpoint_dtype == 'fp16':
            state = {k: (v.half() if v.is_floating_point() else v)
                     for k, v in state.items()}
        torch.save(state, cp + '.tmp')
        os.replace(cp + '.tmp', cp)
        ckpt_records.append({**best_ckpt, 'path': cp,
                             'selection': ('best_test_acc_fallback'
                                           if args.checkpoint_at_threshold is not None
                                           else 'best_test_acc')})
        print(f"best checkpoint at epoch {best_ckpt['epoch']} "
              f"(test {best_ckpt['test_acc']:.2f}) -> {cp}")

    layers, _ = collect_conv_linear_layers(model)
    result = {
        'cell': cell_name,
        'arch': arch.name,
        'n_layers': arch.n_layers,
        'bias_rule': 'prefix',
        'frozen_bias_tensors': [n for n in frozen_names
                                if getattr(layers[n], 'bias', None) is not None],
        'num_recovered': num_recovered,
        'train_size': actual_size,
        'source': 'cifar10',
        'trainable_layers': trainable_names,
        'num_trainable_layers': len(trainable_names),
        'num_trainable_params': sum(p.numel() for p in trainable_params),
        'bn_mode': 'prefix',
        'reinit_bn_layers': reinit_bn,
        'bn_prefix_stats': 'recalibrated',
        'epochs_budget': epochs,
        'epochs_run': len(history),
        'stopped_early': len(history) < epochs,
        'samples_budget': epochs * actual_size,
        'pre_training_acc': start_acc,
        'best_acc': best_acc,
        'final_acc': history[-1]['test_acc'],
        'epoch_1_acc': history[0]['test_acc'],
        'target': args.target,
        'reached_target': epochs_to_target is not None,
        'stop_threshold': args.stop_threshold,
        'epochs_to_stop_threshold': epochs_to_stop,
        'epochs_to_target': epochs_to_target,
        'samples_to_target': samples_to_target,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'seed': args.seed,
        'wall_seconds': time.time() - run_start,
        'history': history,
    }
    result.update(arch.provenance())
    if ckpt_records:
        result.update({'checkpoints': ckpt_records,
                       'checkpoint_dir': args.checkpoint_dir,
                       'checkpoint_dtype': args.checkpoint_dtype,
                       'steps_per_epoch': steps_per_epoch})
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{cell_name}: best {best_acc:.2f}, epoch-1 {history[0]['test_acc']:.2f}, "
          f"target {'reached at epoch ' + str(epochs_to_target) if epochs_to_target else 'NOT reached'}")
    print(f"-> {out_path}")


if __name__ == '__main__':
    main()
