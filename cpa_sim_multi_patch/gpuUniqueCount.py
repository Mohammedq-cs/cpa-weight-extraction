"""
GPU distinct-value counting for float16 columns (CuPy), for patch SELECTION over
ALL traces.

`uniqueCount.count_distinct` (CPU, NumPy-only) counts the distinct float16 values
in one column with a 65536-slot presence buffer. It is exact but serial (one
column at a time), so counting every trace for every (patch, weight) column is
slow on the CPU.

This module does the same count on the GPU, batched over all columns at once, so
selection can use EVERY trace. The method is identical in spirit to
count_distinct and produces byte-identical results (see the equality sanity
check in tests):

  * A column's float16 values are reinterpreted as uint16 codes (65536 possible).
  * Per column we keep a packed PRESENCE BITSET of 65536 bits == 2048 uint32
    words. Memory per column is 8 KB regardless of the number of traces, so
    counting over all traces costs the same RAM regardless of trace count.
  * For every (column, trace) we OR the code's bit into the column's bitset with
    atomicOr, then popcount + sum the words to get the distinct count.

The same float16 value-equality canonicalisation as count_distinct is applied
inline on the uint16 code, so the result matches np.unique(col).size exactly:
  * -0.0 (0x8000) folds to +0.0 (0x0000)  -- distinct bit patterns, equal value
  * every NaN bit pattern folds to one canonical NaN (0x7E00)
  * infinities are left as-is (np.unique keeps them distinct), matching isnan.

GPU-only: requires CuPy + CUDA (same as cpaCupy.py). Kept separate from
uniqueCount.py so that module stays CuPy-free for the per-filter CPU paths.
"""

import numpy as np
import cupy as cp

# 65536 possible uint16 codes -> 65536 bits -> 2048 uint32 words per column.
_BITS = 65536
_WORDS = _BITS // 32  # 2048


# Scatter every (column, trace) code's bit into that column's presence bitset.
# codes: (M, N) row-major uint16 (M columns, N traces).
# bitset: (M, _WORDS) uint32, zero-initialised by the caller.
# The float16 -> canonical uint16 folding is done here so it matches
# uniqueCount.count_distinct exactly.
_scatter_bits_kernel = cp.RawKernel(r'''
extern "C" __global__
void scatter_bits(const unsigned short* __restrict__ codes,
                  unsigned int* __restrict__ bitset,
                  const long M, const long N, const int WORDS) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = M * N;
    if (idx >= total) return;
    long m = idx / N;                 // column index
    unsigned short c = codes[idx];
    // canonicalise (mirror count_distinct): -0.0 -> +0.0, any NaN -> 0x7E00
    if (c == 0x8000u) {
        c = 0x0000u;
    } else if (((c & 0x7C00u) == 0x7C00u) && ((c & 0x03FFu) != 0u)) {
        c = 0x7E00u;
    }
    int word = c >> 5;
    int bit  = c & 31;
    atomicOr(&bitset[m * (long)WORDS + word], (1u << bit));
}
''', 'scatter_bits')

# Population count of a uint32 (CUDA __popc), used to tally set bits per bitset.
_popc_u32 = cp.ElementwiseKernel(
    'uint32 x', 'int32 y', 'y = __popc(x);', 'popc_u32',
)


def _to_uint16(arr):
    """View a float16/uint16/int16 cupy array's bit pattern as uint16."""
    a = cp.ascontiguousarray(arr)
    if a.dtype == cp.uint16:
        return a
    if a.dtype == cp.float16 or a.dtype == cp.int16:
        return a.view(cp.uint16)
    raise TypeError(f"expected float16/int16/uint16, got {a.dtype}")


def _scatter_into(bitset, codes, threads=256):
    """OR each (column, trace) code's bit into the per-column presence `bitset`
    (M, _WORDS) uint32. `codes` is (M, N) uint16. Accumulates -- the caller may
    call this repeatedly with successive trace-batches into the same bitset."""
    M, N = codes.shape
    total = M * N
    if total == 0:
        return
    blocks = (total + threads - 1) // threads
    _scatter_bits_kernel(
        (int(blocks),), (int(threads),),
        (codes.ravel(), bitset.ravel(),
         np.int64(M), np.int64(N), np.int32(_WORDS)),
    )


def gpu_distinct_counts_2d(cols):
    """Distinct float16 values per row of `cols` (M, N), on the GPU.

    `cols` is a (num_columns, num_traces) array (numpy or cupy, float16/uint16).
    Returns an (M,) int32 numpy array; entry m == np.unique(cols[m]).size, i.e.
    byte-identical to uniqueCount.count_distinct on each row.
    """
    codes = _to_uint16(cp.asarray(cols))
    M = codes.shape[0]
    bitset = cp.zeros((M, _WORDS), dtype=cp.uint32)
    _scatter_into(bitset, codes)
    counts = _popc_u32(bitset).sum(axis=1)
    return cp.asnumpy(counts).astype(np.int32)


def _bitset_bytes(num_cols):
    return num_cols * _WORDS * 4


def _torch_f16_to_cupy_u16(t):
    """Reinterpret a contiguous CUDA float16 torch tensor's bits as a cupy uint16
    array, copied so it does not alias torch-managed memory."""
    import torch
    import torch.utils.dlpack as dlpack
    t16 = t.contiguous().view(torch.int16)
    cp_view = cp.from_dlpack(dlpack.to_dlpack(t16)).view(cp.uint16)
    return cp.ascontiguousarray(cp_view).copy()


def gpu_unique_count_matrix(layer_inp, layer, mem_budget_bytes=1_500_000_000,
                            unfold_batch_bytes=400_000_000):
    """Per-(patch, weight) distinct-value matrix of shape (L, numWeights) for a
    Conv2d layer, computed over ALL traces on the GPU.

    `layer_inp` is the captured `(N, C, H, W)` activation (CPU tensor, as returned
    by patchInputs.capture_layer_input). Patches are the `F.unfold` spatial kernel
    applications, in F.unfold order, so patch index `p` here matches
    ColumnProvider's patch `p` exactly.

    Memory is bounded two ways: patches are processed in blocks so the presence
    bitset stays under `mem_budget_bytes`, and traces are streamed in batches so
    the transient unfold stays under `unfold_batch_bytes`. Each result column is
    byte-identical to uniqueCount.count_distinct over the full trace set.
    """
    import torch
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_ch = layer.in_channels
    kH, kW = layer.kernel_size
    numWeights = in_ch * kH * kW
    N = layer_inp.shape[0]

    # L (number of spatial patches) from a tiny one-trace unfold.
    with torch.no_grad():
        probe = F.unfold(
            layer_inp[:1].to(device).half(),
            kernel_size=(kH, kW), dilation=layer.dilation,
            padding=layer.padding, stride=layer.stride,
        )
    L = probe.shape[2]
    del probe

    # Patch-block size: keep bitset (numWeights * pb columns) under the budget.
    pb = max(1, min(L, mem_budget_bytes // max(1, _bitset_bytes(numWeights))))
    # Trace-batch size: keep one unfold (numWeights * L per trace) under budget.
    per_trace_bytes = numWeights * L * 2
    tb = max(1, min(N, unfold_batch_bytes // max(1, per_trace_bytes)))

    counts = np.empty((L, numWeights), dtype=np.int32)

    for p0 in range(0, L, pb):
        p1 = min(p0 + pb, L)
        cur_pb = p1 - p0
        M = numWeights * cur_pb
        bitset = cp.zeros((M, _WORDS), dtype=cp.uint32)
        for ts in range(0, N, tb):
            te = min(ts + tb, N)
            with torch.no_grad():
                u = F.unfold(
                    layer_inp[ts:te].to(device).half(),
                    kernel_size=(kH, kW), dilation=layer.dilation,
                    padding=layer.padding, stride=layer.stride,
                )  # (b, numWeights, L)
                blk = u[:, :numWeights, p0:p1].permute(1, 2, 0).contiguous()
                # (numWeights, cur_pb, b) -> rows ordered j*cur_pb + (p - p0)
            codes = _torch_f16_to_cupy_u16(blk).reshape(M, te - ts)
            _scatter_into(bitset, codes)
            del u, blk, codes
        counts_M = _popc_u32(bitset).sum(axis=1)            # (M,)
        block = cp.asnumpy(counts_M).reshape(numWeights, cur_pb)  # [j, p-p0]
        counts[p0:p1, :] = block.T                          # (cur_pb, numWeights)
        del bitset, counts_M

    return counts


def gpu_best_patch_per_weight(layer_inp, layer, **kwargs):
    """For each weight, the patch index with the most distinct float16 values over
    ALL traces. Returns (best_patch_per_weight (numWeights,) int32, L)."""
    counts = gpu_unique_count_matrix(layer_inp, layer, **kwargs)
    L = counts.shape[0]
    return counts.argmax(axis=0).astype(np.int32), L


def gpu_compute_selected_patches(layer_inp, layer, **kwargs):
    """Sorted union over weights of each weight's most-unique patch, decided over
    ALL traces on the GPU. GPU all-trace analogue of
    patchInputs.compute_selected_patches."""
    best, _ = gpu_best_patch_per_weight(layer_inp, layer, **kwargs)
    return np.unique(best).astype(np.int32)
