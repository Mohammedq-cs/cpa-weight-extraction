"""PGD transfer from the trained surrogates to the victim.

For every cell JSON under --ckpt_results_dir (written by sweepRepair.py
--checkpoint_at_threshold, i.e. run_sources.sbatch), the saved checkpoint is used
as the SOURCE model: L-inf PGD adversarial examples are crafted on it over the
CIFAR-10 test set and scored on the victim. The victim attacking itself
(`white_box`) is run first as a reference and as a live check of eps/alpha.

Every source is attacked with an identical image order and identical random-start
noise (shuffle=False plus a per-seed generator), so cross-source differences are
a paired comparison.

ASR = 1 - target_adv_acc (advTransfer.asr). One JSON per (source, attack seed):
{out_dir}/{cell}_ep{epoch:04d}_seed{seed}.json and white_box_seed{seed}.json.

    python pgdSourceSweep.py --arch vgg11bn \\
        --ckpt_results_dir results/vgg11bn/transfer_sources \\
        --out_dir results/vgg11bn/pgd_transfer_sources_eps5255_fp32 \\
        --eps 0.0196078431372549 --alpha 0.00784313725490196 --steps 75 --fp32
"""

import argparse
import glob
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from advTransfer import adv_serial_transfer, asr, source_asr
import archs
from common import build_transforms
from sweepRepair import DEFAULT_DATA_ROOT


def load_source(path, device, builder, fp32=False):
    """Every checkpoint here is a bare state_dict for its architecture;
    load_state_dict casts, so building fp32 and halving once normalizes all of
    them to the same footing."""
    model = builder().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.float().eval() if fp32 else model.half().eval()


def checkpoint_roster(args):
    """White box, then every checkpoint of every cell under --ckpt_results_dir.

    The checkpoints are read out of the cell JSONs rather than globbed off disk:
    a cell JSON is only written once its checkpoint is complete, so this never
    attacks a half-written .pt left by a preempted job.
    """
    roster = [('white_box', 'victim', args.victim,
               archs.ARCHS[args.arch].n_layers, {})]
    for jp in sorted(glob.glob(os.path.join(args.ckpt_results_dir, 'K*_N*.json'))):
        with open(jp) as f:
            cell = json.load(f)
        for c in cell.get('checkpoints', []):
            roster.append((
                f"{cell['cell']}_ep{c['epoch']:04d}",
                f"repair_bn_{cell.get('bn_mode', 'prefix')}_ckpt",
                c['path'], cell['num_recovered'],
                {'cell': cell['cell'], 'epoch': c['epoch'],
                 'samples_seen': c['samples_seen'], 'steps': c['steps'],
                 'ckpt_test_acc': c['test_acc'],
                 'ckpt_selection': c.get('selection'),
                 'steps_per_epoch': cell.get('steps_per_epoch'),
                 'bn_mode': cell.get('bn_mode', 'prefix'),
                 'train_source': cell['source'],
                 'train_size': cell['train_size'],
                 'train_seed': cell.get('seed'),
                 'checkpoint_dtype': cell.get('checkpoint_dtype')}))
    return roster


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arch', default='resnet18', choices=sorted(archs.ARCHS),
                   help="Which victim to attack; selects constructor, victim "
                        "checkpoint and normalization stats.")
    p.add_argument('--ckpt_results_dir', required=True,
                   help="The cell JSONs whose 'checkpoints' records name the .pt "
                        "files to attack.")
    p.add_argument('--out_dir', required=True)
    p.add_argument('--victim', default=None,
                   help='Default: the registry checkpoint for --arch.')
    p.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--eps', type=float, default=5 / 255)
    p.add_argument('--alpha', type=float, default=2 / 255)
    p.add_argument('--steps', type=int, default=75)
    p.add_argument('--loss_scale', type=float, default=1024.0,
                   help="fp16 only: multiplies the loss before the backward pass so "
                        "the CE gradient does not underflow. Only grad.sign() is "
                        "used, so the attack direction is unchanged.")
    p.add_argument('--fp32', action='store_true',
                   help="Run source AND victim in fp32 and take the CE loss in "
                        "fp64 (forces --loss_scale 1). What the paper uses.")
    p.add_argument('--batch_size', type=int, default=500)
    p.add_argument('--max_batches', type=int, default=None,
                   help="Attack only the first N test batches (smoke tests).")
    p.add_argument('--attack_seeds', type=int, nargs='+', default=[0],
                   help="One paired replicate per seed. Seeds fix the random "
                        "start, so every source sees identical noise.")
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    arch = archs.ARCHS[args.arch]
    if args.victim is None:
        args.victim = arch.checkpoint
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # The attack de-normalizes with the architecture's own stats to clip in
    # [0,1] pixel space, so eps is comparable across architectures.
    _, test_transform = build_transforms(arch.norm_mean, arch.norm_std)
    test_data = CIFAR10(root=args.data_root, train=False,
                        transform=test_transform, download=True)
    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    if args.fp32:
        args.loss_scale = 1.0
    victim = load_source(args.victim, device, arch.builder, fp32=args.fp32)

    roster = checkpoint_roster(args)
    print(f"arch {arch.name} | victim {args.victim} | {len(roster)} sources "
          f"x {len(args.attack_seeds)} seeds\n")

    for label, kind, path, k, extra in roster:
        if not os.path.exists(path):
            print(f"  SKIP {label}: no checkpoint at {path}")
            continue
        for seed in args.attack_seeds:
            out_path = os.path.join(args.out_dir, f'{label}_seed{seed}.json')
            if os.path.exists(out_path) and not args.force:
                print(f"  {label} seed{seed}: already done")
                continue
            source = load_source(path, device, arch.builder, fp32=args.fp32)

            t0 = time.time()
            stats = adv_serial_transfer(source, victim, loader,
                                        mean=arch.norm_mean,
                                        std=arch.norm_std,
                                        device=str(device), eps=args.eps,
                                        alpha=args.alpha, steps=args.steps,
                                        loss_scale=args.loss_scale,
                                        source_fp16=not args.fp32,
                                        target_fp16=not args.fp32,
                                        loss_fp64=args.fp32,
                                        max_batches=args.max_batches,
                                        verbose=False, seed=seed)
            stats.update({'source_label': label, 'source_kind': kind,
                          'source_path': path, 'num_recovered': k,
                          'arch': arch.name, 'n_layers': arch.n_layers,
                          'attack_seed': seed, 'victim_path': args.victim,
                          'wall_seconds': time.time() - t0,
                          'torch_version': torch.__version__})
            # PGD config under its own keys BEFORE the merge: extra['steps'] is the
            # FINE-TUNING step count of the source checkpoint.
            stats.update({'pgd_eps': args.eps, 'pgd_alpha': args.alpha,
                          'pgd_steps': args.steps,
                          'pgd_loss_scale': args.loss_scale,
                          'pgd_precision': 'fp32+ce64' if args.fp32 else 'fp16'})
            stats.update(extra)
            with open(out_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"  {label:<36} seed{seed}  clean {stats['source_clean_acc']:.4f}  "
                  f"srcASR {source_asr(stats):.4f}  "
                  f"ASR {asr(stats):.4f}  cond {stats['conditional_transfer_rate']:.4f}  "
                  f"({stats['wall_seconds']:.0f}s)")

    print(f"\n-> {args.out_dir}")


if __name__ == '__main__':
    main()
