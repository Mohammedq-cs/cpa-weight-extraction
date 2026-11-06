"""PGD adversarial-example transfer harness.

The same L-inf PGD as cpa_sim_first_patch/graphs_code (pgd_attack_exp), made
importable with three things parameterized:

  * **Normalization.** The perturbation budget is applied in [0,1] PIXEL space --
    `pgd` de-normalizes on the way in and re-normalizes on the way out -- so
    eps = 5/255 means the same perturbation for every architecture regardless of
    its Normalize() stats.
  * **Precision.** fp16 or fp32 forward passes; `loss_fp64` takes the
    cross-entropy in fp64. The paper's runs use fp32 models + fp64 CE: very
    confident sources round the fp32/fp16 CE to exactly 0 on many images and PGD
    then stalls, which shows up as false dips in transferability.
  * **Determinism.** `adv_serial_transfer` takes a `seed` and drives the random
    start from a dedicated generator, so every source is attacked with identical
    noise on an identically ordered test set (a paired comparison).

ASR convention: `1 - target_adv_acc`, the victim's error rate on adversarial
examples crafted against the source. `asr(stats)` names it in one place.
"""

import torch
import torch.nn.functional as F

# The stats ResNet-18 / VGG-11-BN were trained under.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# What the small CIFAR CNN was trained under.
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)


def _stat_tensors(mean, std, device, dtype=torch.float32):
    """(1,3,1,1) tensors so de/re-normalization broadcasts over a batch."""
    m = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
    return m, s


def pgd(model, x_norm, y, mean=CIFAR10_MEAN, std=CIFAR10_STD,
        eps=2 / 255, alpha=2 / 255, steps=75, targeted=False, target_y=None,
        random_start=True, model_fp16=True, generator=None, loss_scale=1024.0,
        loss_fp64=False):
    """L-inf PGD in pixel space. Takes and returns NORMALIZED images.

    eps/alpha are in [0,1] pixel units regardless of `mean`/`std`.
    """
    model.eval()
    m, s = _stat_tensors(mean, std, x_norm.device)

    # [-1,1] or whatever the normalization produced -> [0,1]
    x0 = (x_norm * s + m).detach()

    if random_start:
        if generator is not None:
            noise = torch.empty_like(x0).uniform_(-eps, eps, generator=generator)
        else:
            noise = torch.empty_like(x0).uniform_(-eps, eps)
        x_adv = torch.clamp(x0 + noise, 0.0, 1.0)
    else:
        x_adv = x0.clone()

    x_adv = x_adv.float().detach()
    x_adv.requires_grad_(True)

    for _ in range(steps):
        x_in = (x_adv - m) / s
        logits = model(x_in.half()) if model_fp16 else model(x_in)
        # Very confident sources (early checkpoints, margin 15-18, logits up to
        # ~60) round the cross-entropy to exactly 0 in fp32 for a third to half
        # of the images, and PGD stalls on them (VGG K6 ep2: self-ASR 0.69).
        # Taking the loss in fp64 removes the rounding (0.97) without touching
        # the model's forward pass or the attack's objective.
        if loss_fp64:
            logits = logits.double()

        # LOSS SCALING. With fp16 weights the cross-entropy gradient underflows
        # to zero for confident models, and the attack then stalls: measured
        # source ASR sat at 0.818 for a model with ~8 logit margin and did NOT
        # improve with more steps (0.820 at 150), a larger alpha (0.816 at
        # 4/255) or a larger eps (0.805 at 0.05) -- the signature of underflow
        # rather than of a budget limit. Scaling the loss before the backward
        # pass lifts the gradients back into fp16's representable range and
        # gives 0.999-1.000. This is the standard AMP remedy and it changes
        # NOTHING about the attack: only grad.sign() is used, which is invariant
        # to any positive scaling, and the forward pass is untouched.
        if targeted:
            assert target_y is not None, "targeted=True needs target_y"
            loss = F.cross_entropy(logits, target_y) * loss_scale
            grad_sign = torch.sign(torch.autograd.grad(loss, x_adv)[0])
            x_adv = x_adv - alpha * grad_sign      # descend toward the target
        else:
            loss = F.cross_entropy(logits, y) * loss_scale
            grad_sign = torch.sign(torch.autograd.grad(loss, x_adv)[0])
            x_adv = x_adv + alpha * grad_sign      # ascend away from the truth

        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)   # L-inf ball
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()             # valid pixels
        x_adv.requires_grad_(True)

    return (x_adv - m) / s


def adv_serial_transfer(source_model, target_model, dataloader,
                        mean=CIFAR10_MEAN, std=CIFAR10_STD,
                        device='cuda:0', eps=2 / 255, alpha=2 / 255, steps=75,
                        random_start=True, targeted=False, target_class=None,
                        max_batches=None, verbose=True,
                        source_fp16=True, target_fp16=True, seed=None,
                        loss_scale=1024.0, loss_fp64=False):
    """Craft adversarial examples on `source_model`, score them on `target_model`.

    Returns a stats dict. `seed`, when given, fixes the random
    start so different sources are attacked with identical noise.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    source_model = source_model.to(device).eval()
    target_model = target_model.to(device).eval()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    total = 0
    correct_src_clean = 0
    correct_src_adv = 0
    correct_tgt_adv = 0
    correct_tgt_clean = 0
    both_misclassified = 0
    fooled_on_src_total = 0
    # The victim's fooled count existed only as the float target_adv_acc, and
    # "neither fooled" not at all, so agreement had to be reconstructed by
    # inclusion-exclusion. These keep the integers, which makes the result
    # self-validating: neither == total - fooled_src - fooled_tgt + both.
    fooled_on_tgt_total = 0
    neither_fooled = 0
    # Label agreement: does the victim predict the SAME wrong class? Unlike the
    # fooled/not-fooled agreement this does not collapse when the attack
    # saturates the source (at source ASR 1 every other transfer measure becomes
    # victim ASR), so it is the fidelity measure that survives a large eps.
    pred_agree_adv_total = 0
    batch_records = []

    for batch_idx, (imgs, labels) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        imgs = imgs.to(device).float()
        labels = labels.to(device)

        with torch.no_grad():
            src_clean = source_model(imgs.half() if source_fp16 else imgs).argmax(1)
            tgt_clean = target_model(imgs.half() if target_fp16 else imgs).argmax(1)
        correct_src_clean += (src_clean == labels).sum().item()
        correct_tgt_clean += (tgt_clean == labels).sum().item()

        if targeted:
            assert target_class is not None, "targeted=True needs target_class"
            target_y = torch.full_like(labels, target_class)
        else:
            target_y = None

        adv_imgs = pgd(source_model, imgs, labels, mean=mean, std=std,
                       eps=eps, alpha=alpha, steps=steps, loss_scale=loss_scale,
                       targeted=targeted, target_y=target_y,
                       random_start=random_start, model_fp16=source_fp16,
                       generator=generator, loss_fp64=loss_fp64)

        with torch.no_grad():
            src_adv = source_model(adv_imgs.half() if source_fp16 else adv_imgs).argmax(1)
            tgt_adv = target_model(adv_imgs.half() if target_fp16 else adv_imgs).argmax(1)

        batch_size = labels.size(0)
        total += batch_size
        correct_src_adv += (src_adv == labels).sum().item()
        correct_tgt_adv += (tgt_adv == labels).sum().item()
        fooled_on_src = (src_adv != labels).sum().item()
        fooled_on_src_total += fooled_on_src
        both = ((src_adv != labels) & (tgt_adv != labels)).sum().item()
        both_misclassified += both
        fooled_on_tgt = (tgt_adv != labels).sum().item()
        fooled_on_tgt_total += fooled_on_tgt
        neither = ((src_adv == labels) & (tgt_adv == labels)).sum().item()
        neither_fooled += neither
        pred_agree_adv_total += (src_adv == tgt_adv).sum().item()

        batch_records.append({"batch_idx": batch_idx, "fooled_source": fooled_on_src,
                              "fooled_target": fooled_on_tgt, "fooled_both": both,
                              "neither": neither, "batch_size": batch_size})
        if verbose:
            print(f"[Batch {batch_idx}] src_adv fooled={fooled_on_src}/{batch_size}, "
                  f"both={both}, total_so_far={total}")

    stats = {
        "source_clean_acc": correct_src_clean / total,
        "source_adv_acc": correct_src_adv / total,
        "target_clean_acc": correct_tgt_clean / total,
        "target_adv_acc": correct_tgt_adv / total,
        "overall_transfer_rate": both_misclassified / total,
        "conditional_transfer_rate": (both_misclassified / fooled_on_src_total
                                      if fooled_on_src_total > 0 else 0.0),
        "both_misclassified": both_misclassified,
        "fooled_on_src_total": fooled_on_src_total,
        "fooled_on_tgt_total": fooled_on_tgt_total,
        "neither_fooled": neither_fooled,
        # Agreement on the fooled / not-fooled indicator.
        "agreement_rate": (both_misclassified + neither_fooled) / total,
        # Agreement on the predicted label itself.
        "pred_agreement_adv": pred_agree_adv_total / total,
        "total": total,
        "eps": eps, "alpha": alpha, "steps": steps, "seed": seed,
        "loss_scale": loss_scale,
        "batch_records": batch_records,
    }
    if verbose:
        print(f"  source clean {stats['source_clean_acc']:.4f} -> adv {stats['source_adv_acc']:.4f}")
        print(f"  target clean {stats['target_clean_acc']:.4f} -> adv {stats['target_adv_acc']:.4f}")
        print(f"  ASR on target {asr(stats):.4f}  "
              f"(overall transfer {stats['overall_transfer_rate']:.4f}, "
              f"conditional {stats['conditional_transfer_rate']:.4f})")
    return stats


def source_asr(stats):
    """Did the attack work on the model it was crafted against?

    Reported rather than assumed: a transfer number is only interpretable
    against the attack's success on its own source.
    """
    return 1.0 - stats["source_adv_acc"]


def asr(stats):
    """Attack success rate: victim error on adversarial examples."""
    return 1.0 - stats["target_adv_acc"]
