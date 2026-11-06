"""
Last-layer sign-flip correction (ported from the multi-patch
last_layer_flip_test.py). The terminal Linear (fc2) has NO
BatchNorm, so the HW/HD sign-ambiguity cannot be resolved during extraction the
way it is for the BN'd layers. A flipped output row negates that class's logit ->
that class is ~never predicted -> its per-class accuracy collapses to ~0. Flip
those rows back; keep the change only if overall accuracy improves. Uses the
labeled dataset as the functional signal -- no refit.
"""
import sys
import numpy as np
import torch

from utils import calculate_accuracy


def correct_last_layer_signs(model, last_layer, test_loader, device, acc_threshold=10.0):
    model.eval()
    n_classes = last_layer.weight.data.shape[0]

    correct = np.zeros(n_classes, dtype=np.int64)
    total = np.zeros(n_classes, dtype=np.int64)
    with torch.no_grad():
        for inputs, labels in test_loader:
            preds = model(inputs.to(device).half()).max(1)[1].cpu().numpy()
            labels = labels.cpu().numpy()
            for c in range(n_classes):
                m = labels == c
                total[c] += int(m.sum())
                correct[c] += int((preds[m] == c).sum())
    per_class = 100.0 * correct / np.clip(total, 1, None)
    low = [c for c in range(n_classes) if per_class[c] < acc_threshold]

    acc_before = calculate_accuracy(model, test_loader)
    print(f"[last-layer-fix] per-class acc = {np.round(per_class, 1).tolist()}")
    print(f"[last-layer-fix] classes < {acc_threshold}% -> {low}; acc_before = {acc_before:.2f}")
    sys.stdout.flush()

    if not low:
        return {"flipped_classes": [], "acc_before": acc_before,
                "acc_after": acc_before, "applied": False}

    with torch.no_grad():
        for c in low:
            last_layer.weight.data[c] *= -1
    acc_after = calculate_accuracy(model, test_loader)

    if acc_after > acc_before:
        print(f"[last-layer-fix] KEPT flips {low}: acc {acc_before:.2f} -> {acc_after:.2f}")
        sys.stdout.flush()
        return {"flipped_classes": low, "acc_before": acc_before,
                "acc_after": acc_after, "applied": True}

    with torch.no_grad():
        for c in low:
            last_layer.weight.data[c] *= -1
    print(f"[last-layer-fix] flips {low} did NOT help "
          f"({acc_before:.2f} -> {acc_after:.2f}); REVERTED")
    sys.stdout.flush()
    return {"flipped_classes": [], "acc_before": acc_before,
            "acc_after": acc_before, "applied": False}
