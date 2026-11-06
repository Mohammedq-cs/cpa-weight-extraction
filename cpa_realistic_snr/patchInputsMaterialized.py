"""
Materialized best-activation patch storage (EfficientNet / ImageNet).

Combines two existing designs:
  * the per-weight "highest-unique filter activation" selection + column-major
    `(numWeights, num_traces)` column-major storage of the multi-patch pipeline, and
  * EfficientNet's depthwise-aware unfolding.

Unlike ../cpa_sim_multi_patch (small CIFAR net, whole layer input held in RAM),
here num_traces can be ~12.24M and a single layer input never fits in memory, so
materialization STREAMS the forward pass: the patch memmaps are filled batch by
batch. Only the union of each weight's best patch is stored (a patch shared by
several weights is written once).

Two layer kinds:
  * standard conv (e.g. features.0.0): the input patches are shared by every
    output filter, so ONE selection + ONE set of patch files per layer.
    per-filter numWeights = in_channels * kH * kW.
  * depthwise conv (e.g. features.1.0.block.0.0, groups == in_channels): each
    filter f sees ONLY channel f, so selection + patch files are PER FILTER.
    per-filter numWeights = kH * kW. Unfold uses channel f only.

Files written under `{base_dir}/{num_traces}/`:
  standard:  {name}.selection.npy
             {name}.best_patch_per_weight.npy
             {name}.patch{i}.npy            (numWeights, num_traces) float16
  depthwise: {name}.f{f}.selection.npy
             {name}.f{f}.best_patch_per_weight.npy
             {name}.f{f}.patch{i}.npy       (numWeights, num_traces) float16
  both:      {name}.meta.npz                (geometry, depthwise flag, ...)
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.format import open_memmap

from uniqueCount import count_distinct, new_present_buffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COLUMN_BUFFER_WINDOW = 40


# ============================================================
# In-process view over a layer's column-major patch memmaps
# ============================================================

class LayerPatchInputs:
    """A (filter's) inputs as a list of column-major `(numWeights, num_traces)`
    arrays (memmaps). Presents the logical `(num_traces, numWeights)` view via
    `column(p, w)`, buffered per patch in windows of `window` columns."""

    def __init__(self, arrays, window=COLUMN_BUFFER_WINDOW):
        if len(arrays) == 0:
            raise ValueError("LayerPatchInputs needs at least one patch array")
        self._a = arrays
        self.num_patches = len(arrays)
        self.num_weights = arrays[0].shape[0]
        self.num_traces = arrays[0].shape[1]
        self._window = max(1, int(window))
        self._reset_buffers()

    def _reset_buffers(self):
        self._buf = [None] * self.num_patches
        self._buf_start = [-1] * self.num_patches

    def column(self, p, w):
        if self._window <= 1:
            return self._a[p][w]
        w0 = (w // self._window) * self._window
        if self._buf_start[p] != w0:
            self._buf[p] = np.ascontiguousarray(self._a[p][w0:w0 + self._window])
            self._buf_start[p] = w0
        return self._buf[p][w - w0]

    def truncate_traces(self, n):
        if n < self.num_traces:
            self._a = [arr[:, :n] for arr in self._a]
            self.num_traces = n
            self._reset_buffers()


def align_inputs(sim, baseline, layer_name):
    if sim.num_patches != baseline.num_patches:
        raise ValueError(
            f"{layer_name}: sim has {sim.num_patches} patches but baseline has "
            f"{baseline.num_patches}")
    if sim.num_weights != baseline.num_weights:
        raise ValueError(
            f"{layer_name}: sim has {sim.num_weights} weights but baseline has "
            f"{baseline.num_weights}")
    n = min(sim.num_traces, baseline.num_traces)
    sim.truncate_traces(n)
    baseline.truncate_traces(n)
    return sim, baseline


# ============================================================
# Geometry / depthwise helpers
# ============================================================

def is_depthwise(layer):
    return (isinstance(layer, nn.Conv2d)
            and layer.groups == layer.in_channels and layer.groups > 1)


def per_filter_numWeights(layer):
    if isinstance(layer, nn.Conv2d):
        kH, kW = layer.kernel_size
        in_per_group = layer.in_channels // layer.groups
        return in_per_group * kH * kW
    return layer.in_features


def num_filters(layer):
    if isinstance(layer, nn.Conv2d):
        return layer.out_channels
    return layer.weight.data.shape[0]


def _conv_geom(layer):
    kH, kW = layer.kernel_size
    sH, sW = layer.stride
    pH, pW = layer.padding
    dH, dW = layer.dilation
    return kH, kW, sH, sW, pH, pW, dH, dW


def _out_grid(H, W, layer):
    kH, kW, sH, sW, pH, pW, dH, dW = _conv_geom(layer)
    oh = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    ow = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    return oh, ow


def _extract_patches(t, layer, patch_list, filter_idx):
    """Gather a SPECIFIC list of spatial patches directly (no full F.unfold), so the
    transient is (n, pfnw, len(patch_list)) instead of (n, pfnw, L). Column order
    (C, kH, kW) matches F.unfold and the filter weight flatten. Depthwise -> only
    channel `filter_idx`. Returns (n, pfnw, len(patch_list))."""
    kH, kW, sH, sW, pH, pW, dH, dW = _conv_geom(layer)
    if is_depthwise(layer):
        t = t[:, filter_idx:filter_idx + 1]
    n, C, H, W = t.shape
    oh, ow = _out_grid(H, W, layer)
    tp = F.pad(t, (pW, pW, pH, pH)) if (pH or pW) else t
    out = torch.empty((n, C * kH * kW, len(patch_list)), dtype=t.dtype, device=t.device)
    for idx, p in enumerate(patch_list):
        p = int(p)
        oy, ox = p // ow, p % ow
        rs, cs = oy * sH, ox * sW
        win = tp[:, :, rs:rs + dH * (kH - 1) + 1:dH, cs:cs + dW * (kW - 1) + 1:dW]  # (n,C,kH,kW)
        out[:, :, idx] = win.reshape(n, C * kH * kW)
    return out


# ============================================================
# Path / key helpers
# ============================================================

def _key(name, layer, filter_idx):
    """Storage key: per-filter for depthwise, shared for standard conv."""
    return f"{name}.f{filter_idx}" if is_depthwise(layer) else name


def _input_dir(base_dir, num_traces):
    return os.path.join(base_dir, str(num_traces))


def _patch_paths(input_dir, key, num_selected):
    return [os.path.join(input_dir, f"{key}.patch{i}.npy") for i in range(num_selected)]


def _selection_path(input_dir, key):
    return os.path.join(input_dir, f"{key}.selection.npy")


def _bpw_path(input_dir, key):
    return os.path.join(input_dir, f"{key}.best_patch_per_weight.npy")


def _meta_path(input_dir, name):
    return os.path.join(input_dir, f"{name}.meta.npz")


def save_selection(input_dir, key, selection, best_patch_per_weight):
    os.makedirs(input_dir, exist_ok=True)
    np.save(_selection_path(input_dir, key), np.asarray(selection, dtype=np.int32))
    np.save(_bpw_path(input_dir, key), np.asarray(best_patch_per_weight, dtype=np.int32))


def load_selection(input_dir, key):
    p = _selection_path(input_dir, key)
    return np.load(p).astype(np.int32) if os.path.exists(p) else None


def save_meta(input_dir, name, layer, L, num_traces):
    os.makedirs(input_dir, exist_ok=True)
    kH, kW = layer.kernel_size
    np.savez(
        _meta_path(input_dir, name),
        is_conv=isinstance(layer, nn.Conv2d),
        is_depthwise=is_depthwise(layer),
        kernel_size=np.array([kH, kW], dtype=np.int64),
        stride=np.array(layer.stride, dtype=np.int64) if isinstance(layer, nn.Conv2d) else np.array([1, 1]),
        padding=np.array(layer.padding, dtype=np.int64) if isinstance(layer, nn.Conv2d) else np.array([0, 0]),
        dilation=np.array(layer.dilation, dtype=np.int64) if isinstance(layer, nn.Conv2d) else np.array([1, 1]),
        in_channels=np.int64(layer.in_channels if isinstance(layer, nn.Conv2d) else layer.in_features),
        per_filter_numWeights=np.int64(per_filter_numWeights(layer)),
        num_filters=np.int64(num_filters(layer)),
        L=np.int64(L),
        num_traces=np.int64(num_traces),
    )


def load_meta(input_dir, name):
    return dict(np.load(_meta_path(input_dir, name), allow_pickle=True))


# ============================================================
# Loading materialized patches for one filter
# ============================================================

def load_selection_map(input_dir, name, layer):
    """Reconstruct the selection_map {filter_idx -> (best_patch_per_weight, union, L)}
    from the persisted selection files, so a SIM pass can materialize the EXACT same
    patches the baseline chose (sim/baseline must align 1:1)."""
    meta = load_meta(input_dir, name)
    L = int(meta["L"])
    sel = {}
    if is_depthwise(layer):
        for f in range(num_filters(layer)):
            key = f"{name}.f{f}"
            union = load_selection(input_dir, key)
            best = np.load(_bpw_path(input_dir, key)).astype(np.int32)
            sel[f] = (best, union, L)
    else:
        union = load_selection(input_dir, name)
        best = np.load(_bpw_path(input_dir, name)).astype(np.int32)
        sel[0] = (best, union, L)
    return sel


def load_filter_inputs(input_dir, name, filter_idx, depthwise):
    """Load the materialized patch memmaps for one filter as a LayerPatchInputs.
    Depthwise -> the filter-specific `{name}.f{f}.patch*.npy`; standard conv ->
    the shared `{name}.patch*.npy` (same for every filter)."""
    key = f"{name}.f{filter_idx}" if depthwise else name
    arrays, i = [], 0
    while True:
        p = os.path.join(input_dir, f"{key}.patch{i}.npy")
        if not os.path.exists(p):
            break
        arrays.append(np.load(p, mmap_mode='r'))
        i += 1
    if not arrays:
        raise FileNotFoundError(f"No materialized patches for {key} in {input_dir}")
    return LayerPatchInputs(arrays)


# ============================================================
# Unique-count cache (per filter; shared across resumes)
# ============================================================

def _ucm_path(input_dir, key):
    return os.path.join(input_dir, f"{key}.unique_counts.pkl")


def save_unique_count_matrix(input_dir, key, ucm, n_traces):
    ucm = np.asarray(ucm, dtype=np.int32)
    num_patches, numWeights = ucm.shape
    payload = {"unique_count_matrix": ucm, "n_traces": int(n_traces),
               "num_patches": int(num_patches), "numWeights": int(numWeights)}
    path = _ucm_path(input_dir, key)
    tmp = path + f".tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def load_unique_count_matrix(input_dir, key, n_traces, num_patches, numWeights):
    path = _ucm_path(input_dir, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if (payload.get("n_traces") != int(n_traces)
            or payload.get("num_patches") != int(num_patches)
            or payload.get("numWeights") != int(numWeights)):
        return None
    mat = np.asarray(payload["unique_count_matrix"], dtype=np.int32)
    return mat if mat.shape == (int(num_patches), int(numWeights)) else None


# ============================================================
# Selection: per-weight best patch on a bounded image sample
# ============================================================

def compute_selection_for_layer(sample_layer_inp, layer, patch_chunk=512):
    """Compute per-filter selection from a sample of this layer's input activation,
    processing spatial patches in chunks so the transient stays (n, pfnw,
    patch_chunk) instead of (n, pfnw, L) -- bounded RAM regardless of L.

    Returns dict filter_idx -> (best_patch_per_weight, union, L). Standard conv:
    one shared selection (key 0); depthwise: one per channel."""
    out = {}
    filters = range(num_filters(layer)) if is_depthwise(layer) else [0]
    _, _, H, W = sample_layer_inp.shape
    oh, ow = _out_grid(H, W, layer)
    L = oh * ow
    pfnw = per_filter_numWeights(layer)
    present = new_present_buffer()
    for f in filters:
        best_count = np.full(pfnw, -1, dtype=np.int64)
        best_patch = np.zeros(pfnw, dtype=np.int32)
        for p0 in range(0, L, patch_chunk):
            plist = list(range(p0, min(p0 + patch_chunk, L)))
            blk = _extract_patches(sample_layer_inp, layer, plist, f)
            blk = blk.to(torch.float16).cpu().numpy()             # (n, pfnw, len)
            for li, p in enumerate(plist):
                Xp = blk[:, :, li]
                for j in range(pfnw):
                    c = count_distinct(np.ascontiguousarray(Xp[:, j]), present)
                    if c > best_count[j]:
                        best_count[j] = c
                        best_patch[j] = p
            del blk
        out[f] = (best_patch.astype(np.int32), np.unique(best_patch).astype(np.int32), L)
    return out


# ============================================================
# Streaming materialization of selected patches
# ============================================================

def _open_patch_memmaps(input_dir, key, numWeights, N, num_selected):
    os.makedirs(input_dir, exist_ok=True)
    return [open_memmap(p, mode='w+', dtype=np.float16, shape=(numWeights, N))
            for p in _patch_paths(input_dir, key, num_selected)]


class MaterializedWriter:
    """Streams a layer's selected patches into column-major memmaps as forward
    batches arrive. One instance per layer; holds memmaps for every filter's
    selected patches (depthwise) or the single shared selection (standard)."""

    def __init__(self, input_dir, name, layer, selection_map, N):
        self.name = name
        self.layer = layer
        self.depthwise = is_depthwise(layer)
        self.N = N
        self.pfnw = per_filter_numWeights(layer)
        self.offset = 0
        self.selection_map = selection_map           # filter_idx -> (best, union, L)
        self.mmaps = {}                              # filter_idx -> [memmap per selected patch]
        self.unions = {}                            # filter_idx -> union array
        for f, (best, union, L) in selection_map.items():
            key = f"{name}.f{f}" if self.depthwise else name
            self.mmaps[f] = _open_patch_memmaps(input_dir, key, self.pfnw, N, len(union))
            self.unions[f] = union

    def write_batch(self, layer_inp_batch):
        """layer_inp_batch: (b, C, H, W) tensor for this layer. Slice ONLY the union
        patches (no full unfold) and write them at the running trace offset. Per-call
        transient is (use, pfnw, |U|)."""
        b = layer_inp_batch.shape[0]
        ts = self.offset
        te = min(ts + b, self.N)
        if ts >= self.N:
            return 0
        use = te - ts
        for f, mmaps in self.mmaps.items():
            union = self.unions[f]
            blk = _extract_patches(layer_inp_batch[:use], self.layer, list(union), f)
            blk = blk.to(torch.float16).cpu().numpy()            # (use, pfnw, |U|)
            for i in range(len(union)):
                mmaps[i][:, ts:te] = blk[:, :, i].T
            del blk
        self.offset = te
        return use

    def flush(self):
        for mmaps in self.mmaps.values():
            for m in mmaps:
                m.flush()
        self.mmaps = {}


# ============================================================
# Baseline generation (selection + streamed materialization)
# ============================================================

def _iter_image_batches(data_source, batch_size):
    """Yield image tensor batches from a DataLoader or a single tensor."""
    if isinstance(data_source, torch.Tensor):
        if data_source.dim() == 3:
            data_source = data_source.unsqueeze(0)
        for b in torch.split(data_source, batch_size, dim=0):
            yield b
    else:
        for batch in data_source:
            yield batch[0] if isinstance(batch, (list, tuple)) else batch


def _install_input_hooks(model, layers):
    """Hook ONLY the target layers, capturing each one's INPUT activation on its
    own device (no per-layer CPU copy, unlike modelUtils.setupHooks which would
    move every layer's input+output to CPU each forward -- far too slow at 12M)."""
    acts, handles = {}, []

    def mk(n):
        def hook(mod, inp, out):
            acts[n] = inp[0].detach()
        return hook

    for n, layer in layers.items():
        handles.append(layer.register_forward_hook(mk(n)))
    return acts, handles


def generate_baseline_materialized(
    model, data_source, num_traces, base_dir, layer_names,
    selection_images=8000, fwd_batch=1000, epochs=1, selection_source=None,
    selection_dir=None,
):
    """Forward the (baseline) model and materialize ONLY the selected best patches
    for the given layers, streaming so a full layer input never sits in RAM.

      1) selection pass: forward a bounded sample, capture each target layer's
         input, compute per-(filter) best patches + union. If `selection_source`
         is given (e.g. a seeded RANDOM cross-class subset) it is used for the
         sample -- IMPORTANT when `data_source` is class-ordered (ImageFolder),
         otherwise the first `selection_images` would be one or two classes only
         and bias the patch choice. If None, the first `selection_images` of
         `data_source` are used.
      2) materialization pass: stream `data_source` (cycling `epochs` times to
         reach `num_traces`), slice the selected patches per batch, write memmaps.

    `layer_names` -> list of module names to process. Only those layers are
    hooked. Inputs are cast to the model's parameter dtype (e.g. half)."""
    model.eval()
    mdtype = next(model.parameters()).dtype
    named = dict(model.named_modules())
    layers = {n: named[n] for n in layer_names}
    input_dir = _input_dir(base_dir, num_traces)
    os.makedirs(input_dir, exist_ok=True)

    acts, handles = _install_input_hooks(model, layers)
    try:
        selections = {}   # name -> selection_map
        if selection_dir is not None:
            # ---- reuse an existing (baseline) selection: SIM pass ----
            for n in layer_names:
                layer = layers[n]
                sel = load_selection_map(selection_dir, n, layer)
                selections[n] = sel
                L = next(iter(sel.values()))[2]
                save_meta(input_dir, n, layer, L, num_traces)
                for f, (best, union, _L) in sel.items():
                    key = f"{n}.f{f}" if is_depthwise(layer) else n
                    save_selection(input_dir, key, union, best)
                print(f"[selection-reuse] {n}: loaded from {selection_dir}")
            sys.stdout.flush()
        else:
            # ---- 1) compute selection on a bounded (representative) sample ----
            samples = {n: [] for n in layer_names}
            seen = 0
            sel_iter = selection_source if selection_source is not None else data_source
            with torch.no_grad():
                for imgs in _iter_image_batches(sel_iter, fwd_batch):
                    take = min(imgs.shape[0], selection_images - seen)
                    imgs = imgs[:take].to(device).to(mdtype)
                    _ = model(imgs)
                    for n in layer_names:
                        samples[n].append(acts[n].detach().cpu())
                    seen += take
                    del imgs
                    if seen >= selection_images:
                        break
            for n in layer_names:
                layer = layers[n]
                sample = torch.cat(samples[n], dim=0)
                sel = compute_selection_for_layer(sample, layer)
                selections[n] = sel
                L = next(iter(sel.values()))[2]
                save_meta(input_dir, n, layer, L, num_traces)
                for f, (best, union, _L) in sel.items():
                    key = f"{n}.f{f}" if is_depthwise(layer) else n
                    save_selection(input_dir, key, union, best)
                    print(f"[selection] {key}: |U|={len(union)} of L={L} "
                          f"(pfnw={per_filter_numWeights(layer)})")
                sys.stdout.flush()
                del sample, samples[n]

        # ---------- 2) streamed materialization to num_traces ----------
        writers = {n: MaterializedWriter(input_dir, n, layers[n], selections[n], num_traces)
                   for n in layer_names}
        written = 0
        nb = 0
        with torch.no_grad():
            for _ep in range(max(1, epochs)):
                if written >= num_traces:
                    break
                for imgs in _iter_image_batches(data_source, fwd_batch):
                    if written >= num_traces:
                        break
                    imgs = imgs.to(device).to(mdtype)
                    _ = model(imgs)
                    use = None
                    for n in layer_names:
                        u = writers[n].write_batch(acts[n])
                        use = u if use is None else use
                    written += use
                    nb += 1
                    del imgs
                    if nb % 50 == 0:
                        print(f"[materialize] {written}/{num_traces} traces")
                        sys.stdout.flush()
        for n in layer_names:
            writers[n].flush()
        print(f"[materialize] done: {written} traces written (target {num_traces})")
        sys.stdout.flush()
        return written
    finally:
        for h in handles:
            h.remove()
