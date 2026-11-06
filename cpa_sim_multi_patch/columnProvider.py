"""
On-the-fly column source for the UNCHANGED multi-patch CPA extractor.

`ColumnProvider` is the extractor's input source: it exposes `num_patches`,
`num_traces`, `num_weights`, and `column(p, w)` (a contiguous float16 array of
length num_traces). It holds the layer's COMPACT input activation in host RAM and
produces any `(patch, weight)` column by a direct strided gather (== F.unfold
zero-pad semantics), so:

  * nothing is materialised to disk -- columns come straight from RAM,
  * the exposed patch set is whatever the caller passes as `selection` (the
    per-weight best-patch union), or ALL L patches when `selection=None`.

The extractor, CPA math, IncrementalLeakage, select_patch, etc. consume this
`column(p, w)` interface directly.

`column(p, w)` is byte-identical to `F.unfold(A, ...)[:, w, p]`:
  weight w -> (c_in, kh, kw) = (w // (kH*kW), (w % (kH*kW)) // kW, w % kW)
  patch  p -> (oh, ow)       = (p // oW, p % oW)
  source  -> A[:, c_in, oh*sH - pH + kh*dH, ow*sW - pW + kw*dW]  (0 if out of bounds)
"""

from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

# Picklable description of a layer's unfold geometry, so a worker process can
# rebuild a ColumnProvider over a shared-memory activation without the nn.Module.
LayerSpec = namedtuple("LayerSpec", "is_conv kernel stride padding dilation")


def make_spec(layer):
    if isinstance(layer, nn.Conv2d):
        return LayerSpec(True, tuple(layer.kernel_size), tuple(layer.stride),
                         tuple(layer.padding), tuple(layer.dilation))
    return LayerSpec(False, None, None, None, None)


def _to_np_f16(activation):
    """Accept a torch tensor (CPU/GPU) or numpy array; return contiguous np.float16."""
    if isinstance(activation, torch.Tensor):
        activation = activation.detach().to("cpu").to(torch.float16).numpy()
    return np.ascontiguousarray(np.asarray(activation, dtype=np.float16))


class ColumnProvider:
    """On-the-fly columns from a held activation.

    Conv:  activation `A` of shape (N, C_in, H, W)  -> num_patches = oH*oW = L.
    Linear: activation `A` of shape (N, in_features) -> num_patches = 1.
    """

    def __init__(self, activation, layer, copy=True, selection=None):
        # `layer` may be an nn.Module or a LayerSpec. `copy=False` wraps the array
        # as-is (used for zero-copy shared-memory views in worker processes).
        # `selection` (conv only): the subset of spatial patches to expose -- the
        # per-weight best-patch union, like the original. Restricting to these
        # "relevant patches" is what keeps the extractor fast (the accumulator
        # updates + UCM run over ~tens of patches, not all L). `column(p,w)` and
        # `columns_all_patches` index into the selection; `num_patches=len(selection)`.
        # Activation is stored with the TRACE axis LAST -- conv (C,H,W,N), linear
        # (in_features,N) -- so a column (one (channel,position) across all traces)
        # is a CONTIGUOUS slice instead of a strided gather (cache-friendly: fixes
        # the slow per-weight reads / low GPU util). The transpose is done ONCE here
        # in the driver (copy=True); workers receive the already-transposed array
        # via shared memory (copy=False) and skip it.
        A = _to_np_f16(activation) if copy else activation
        spec = layer if isinstance(layer, LayerSpec) else make_spec(layer)
        self.spec = spec
        self.is_conv = spec.is_conv

        if self.is_conv:
            self.kH, self.kW = spec.kernel
            self.sH, self.sW = spec.stride
            self.pH, self.pW = spec.padding
            self.dH, self.dW = spec.dilation
            self._kHW = self.kH * self.kW
            # (N,C,H,W) -> (C,H,W,N)
            self.A = np.ascontiguousarray(A.transpose(1, 2, 3, 0)) if copy else A
            self.C_in, self.H, self.W, self.N = self.A.shape
            self.oH = (self.H + 2 * self.pH - self.dH * (self.kH - 1) - 1) // self.sH + 1
            self.oW = (self.W + 2 * self.pW - self.dW * (self.kW - 1) - 1) // self.sW + 1
            self._L = self.oH * self.oW
            # patch_ids maps the exposed index p -> the actual spatial patch.
            self._patch_ids = (np.arange(self._L, dtype=np.int64) if selection is None
                               else np.asarray(selection, dtype=np.int64).ravel())
            self.num_patches = int(self._patch_ids.shape[0])
            self.num_weights = self.C_in * self.kH * self.kW
        else:
            if copy:
                if A.ndim != 2:
                    A = A.reshape(A.shape[0], -1)
                self.A = np.ascontiguousarray(A.T)         # (N,in_features) -> (in_features,N)
            else:
                self.A = A                                 # already (in_features,N)
            self.in_features, self.N = self.A.shape
            self.num_patches = 1
            self.num_weights = self.in_features

        self.num_traces = self.N
        self._zero = np.zeros(self.N, dtype=np.float16)

    # ---- the extractor's whole input API: column(p, w) + the three counts ----

    def column(self, p, w):
        """Logical column w of patch p as a contiguous (num_traces,) float16 array.
        Trace axis is last, so this is a contiguous slice (no strided gather)."""
        if not self.is_conv:
            return self.A[w]                                # (in_features,N) -> row w

        c_in = w // self._kHW
        rem = w % self._kHW
        kh, kw = divmod(rem, self.kW)
        oh, ow = divmod(int(self._patch_ids[p]), self.oW)   # exposed idx -> actual patch
        ih = oh * self.sH - self.pH + kh * self.dH
        iw = ow * self.sW - self.pW + kw * self.dW
        if 0 <= ih < self.H and 0 <= iw < self.W:
            return self.A[c_in, ih, iw]                     # (C,H,W,N) -> (N,) contiguous
        return self._zero  # padded position -> zeros (matches F.unfold)

    def truncate_traces(self, n):
        """Restrict to the first n traces (trace axis is last)."""
        if n < self.num_traces:
            self.A = np.ascontiguousarray(self.A[..., :n])
            self.N = n
            self.num_traces = n
            self._zero = np.zeros(n, dtype=np.float16)
