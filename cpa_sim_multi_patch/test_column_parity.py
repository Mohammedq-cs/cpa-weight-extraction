"""Foundational test: ColumnProvider.column(p,w) byte-identical to F.unfold(A)[:,w,p]
across conv configs (incl. padded edges) + Linear. CPU-only, fast."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from columnProvider import ColumnProvider

torch.manual_seed(0)
np.random.seed(0)


def check_conv(C_in, H, W, kH, kW, stride, pad, dil, N=37):
    layer = nn.Conv2d(C_in, 5, (kH, kW), stride=stride, padding=pad, dilation=dil)
    A = torch.randn(N, C_in, H, W).half()
    prov = ColumnProvider(A, layer)
    u = F.unfold(A, kernel_size=(kH, kW), dilation=dil, padding=pad, stride=stride)
    u = u[:, :prov.num_weights, :].numpy().astype(np.float16)  # (N, numWeights, L)
    assert u.shape[2] == prov.num_patches, (u.shape[2], prov.num_patches)
    # exhaustive over all (p, w)
    mism = 0
    for w in range(prov.num_weights):
        for p in range(prov.num_patches):
            col = prov.column(p, w)
            ref = u[:, w, p]
            if not np.array_equal(col, ref):
                mism += 1
    tag = f"conv C={C_in} {H}x{W} k={kH}x{kW} s={stride} p={pad} d={dil} L={prov.num_patches} W={prov.num_weights}"
    print(("OK  " if mism == 0 else f"FAIL({mism}) ") + tag)
    return mism == 0


def check_linear(in_f, N=37):
    layer = nn.Linear(in_f, 3)
    A = torch.randn(N, in_f).half()
    prov = ColumnProvider(A, layer)
    ok = all(np.array_equal(prov.column(0, w), A.numpy().astype(np.float16)[:, w])
             for w in range(in_f))
    print(("OK  " if ok else "FAIL ") + f"linear in_features={in_f} patches={prov.num_patches}")
    return ok


allok = True
# the real net's conv configs (k=3, pad=1, s=1, d=1) at every spatial size
for C_in, H in [(3, 32), (32, 32), (32, 16), (64, 16), (64, 8), (128, 8)]:
    allok &= check_conv(C_in, H, H, 3, 3, (1, 1), (1, 1), (1, 1))
# stress: stride/dilation/asym/no-pad to exercise edge + bounds logic
allok &= check_conv(4, 10, 12, 3, 3, (2, 2), (1, 1), (1, 1))
allok &= check_conv(2, 9, 9, 3, 3, (1, 1), (0, 0), (1, 1))
allok &= check_conv(3, 11, 11, 3, 3, (1, 1), (2, 2), (2, 2))
allok &= check_conv(2, 8, 6, 2, 4, (1, 1), (1, 2), (1, 1))
# linear (fc1, fc2)
allok &= check_linear(2048)
allok &= check_linear(128)

print("\nALL PASS" if allok else "\nSOME FAILED")
