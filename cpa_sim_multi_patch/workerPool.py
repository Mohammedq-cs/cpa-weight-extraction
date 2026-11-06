"""
Single-node multi-GPU executor for the in-process pipeline.

One persistent worker process per GPU (spawn, cupy pinned to one device). Per
layer the orchestrator captures the sim/baseline activations ONCE, plus the
unique-count matrix and the true filter weights, and publishes all four to
shared-memory blocks. Each filter task carries only the (small) shm descriptors;
a worker maps them lazily the first time it sees a new layer, wraps the
activations in ColumnProvider (on-the-fly columns, zero-copy over shared memory),
and runs the UNCHANGED extractor on its assigned filters. The big activation
lives once in host RAM (shared by all workers); only small per-filter results
travel back. Replaces the SLURM filter array -- no NFS, no per-batch --wait, no
QOS churn. Per-layer serialization is kept; intra-layer filters run fully in
parallel across the GPUs.

Lazy per-filter setup avoids any broadcast/round-robin race: every worker that
pulls a filter for a not-yet-seen layer sets that layer up itself.
"""
import os
import pickle
import contextlib
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory


def load_filter_if_present(layer_dir, idx):
    """Per-filter resume: recovered weights for filter idx if a complete
    filter_{idx}.pkl is on disk, else None. Corrupt/half-written pickles (killed
    mid-save) are treated as absent so they get re-extracted. Lets a layer
    interrupted partway (e.g. fc1 killed at the time limit) resume from the
    filters already done instead of redoing the whole layer."""
    p = os.path.join(layer_dir, f"filter_{idx}.pkl")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as fh:
            return np.asarray(pickle.load(fh), dtype=np.float16).reshape(-1)
    except Exception:
        return None


@contextlib.contextmanager
def filter_log(scripts_dir, name, idx):
    """Redirect a filter's extractor stdout to scripts_dir/{name}/filter_{idx}.out."""
    if not scripts_dir:
        yield
        return
    d = os.path.join(scripts_dir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"filter_{idx}.out"), "w") as f, contextlib.redirect_stdout(f):
        yield


def _publish(arr):
    """Copy a numpy array into a fresh SharedMemory block; return (shm, meta)."""
    arr = np.ascontiguousarray(arr)
    shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[:] = arr
    return shm, (shm.name, tuple(arr.shape), str(arr.dtype))


def _map(meta):
    name, shape, dtype = meta
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
    return shm, arr


# ============================================================
# Worker process
# ============================================================

def _worker_loop(gpu_id, task_q, res_q):
    import numpy as np
    import cupy as cp
    # Pin this worker's cupy ops (the extractor uses cupy) to one physical GPU.
    # We pin cupy explicitly rather than via CUDA_VISIBLE_DEVICES: the spawn
    # bootstrap imports the module chain before this runs, so CVD here would be
    # too late for torch -- but cupy is lazy and honours an explicit device.
    cp.cuda.Device(gpu_id).use()
    from columnProvider import ColumnProvider
    from extractFilterMultiPatchOptimized import (
        extract_filter_weights_multi_patch, checkpoint_path, save_outputs,
    )

    cur = {"lid": None, "shms": [], "sim": None, "base": None, "ucm": None,
           "trueW": None, "name": None}

    def release():
        for shm in cur["shms"]:
            try:
                shm.close()
            except Exception:
                pass
        cur["shms"] = []

    def setup(lid, name, sim_meta, base_meta, ucm_meta, trueW_meta, spec, selection):
        release()
        shm_s, sim = _map(sim_meta)
        shm_b, base = _map(base_meta)
        shm_u, ucm = _map(ucm_meta)
        shm_w, trueW = _map(trueW_meta)
        cur.update(lid=lid, shms=[shm_s, shm_b, shm_u, shm_w],
                   sim=ColumnProvider(sim, spec, copy=False, selection=selection),
                   base=ColumnProvider(base, spec, copy=False, selection=selection),
                   ucm=np.ascontiguousarray(ucm), trueW=trueW, name=name)

    res_q.put(("up", gpu_id))
    while True:
        task = task_q.get()
        if task is None:
            break
        (lid, name, sim_meta, base_meta, ucm_meta, trueW_meta, spec, selection,
         idx, leak, noise, thr, outdir, seed, ckpt_every, corr_dir, scripts_dir) = task
        if cur["lid"] != lid:
            setup(lid, name, sim_meta, base_meta, ucm_meta, trueW_meta, spec, selection)
        np.random.seed(seed)
        ckpt = checkpoint_path(outdir, name, idx)
        try:
            with filter_log(scripts_dir, name, idx):   # extractor chatter -> scriptsOut
                w, corr, chosen = extract_filter_weights_multi_patch(
                    cur["sim"], cur["base"], cur["trueW"][idx], leak, noise, thr, ckpt,
                    unique_count_matrix=cur["ucm"], checkpoint_every=ckpt_every)
            # persist per-filter recovered weights (+ corr + chosen) to disk
            save_outputs(outdir, corr_dir or outdir, name, idx, w, corr, chosen)
            if os.path.exists(ckpt):
                os.remove(ckpt)
            res_q.put(("result", idx, np.asarray(w, dtype=np.float16)))
        except Exception as e:
            import traceback
            res_q.put(("error", idx, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    release()


# ============================================================
# Executor
# ============================================================

class PoolExecutor:
    def __init__(self, num_gpus):
        self.n = num_gpus
        ctx = mp.get_context("spawn")
        self.task_q = ctx.Queue()
        self.res_q = ctx.Queue()
        self.procs = []
        for gid in range(num_gpus):
            p = ctx.Process(target=_worker_loop, args=(gid, self.task_q, self.res_q), daemon=True)
            p.start()
            self.procs.append(p)
        ups = 0
        while ups < num_gpus:
            if self.res_q.get()[0] == "up":
                ups += 1
        self._lid = 0

    def run_layer(self, name, num_filters, sim_prov, base_prov, ucm, true_W,
                  leakage_model, noise_level, threshold, output_dir, seed,
                  checkpoint_every=80, corr_dir=None, scripts_dir=None):
        layer_dir = os.path.join(output_dir, name)
        os.makedirs(layer_dir, exist_ok=True)

        # per-filter resume: load filters already on disk; only (re)submit the rest.
        out = [None] * num_filters
        todo = []
        for idx in range(num_filters):
            saved = load_filter_if_present(layer_dir, idx)
            if saved is not None:
                out[idx] = saved
            else:
                todo.append(idx)
        if not todo:
            return out

        self._lid += 1
        lid = self._lid

        shm_s, sim_meta = _publish(sim_prov.A)
        shm_b, base_meta = _publish(base_prov.A)
        shm_u, ucm_meta = _publish(np.asarray(ucm, dtype=np.int32))
        shm_w, trueW_meta = _publish(np.asarray(true_W, dtype=np.float16))
        spec = sim_prov.spec
        selection = sim_prov._patch_ids if sim_prov.is_conv else None
        try:
            for idx in todo:
                self.task_q.put((lid, name, sim_meta, base_meta, ucm_meta, trueW_meta,
                                 spec, selection, idx, leakage_model, noise_level, threshold,
                                 output_dir, seed, checkpoint_every, corr_dir, scripts_dir))
            got = 0
            while got < len(todo):
                msg = self.res_q.get()
                if msg[0] == "error":
                    raise RuntimeError(f"worker failed on filter {msg[1]}: {msg[2]}")
                _, idx, w = msg
                out[idx] = w
                got += 1
            return out
        finally:
            # workers keep their mappings until they advance to the next layer or
            # shut down; unlink the names now (POSIX: existing maps stay valid).
            for shm in (shm_s, shm_b, shm_u, shm_w):
                shm.close()
                shm.unlink()

    def shutdown(self):
        for _ in range(self.n):
            self.task_q.put(None)
        for p in self.procs:
            p.join(timeout=30)
