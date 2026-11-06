import numpy as np
import itertools
import time
import tqdm
import itertools
from utils import *
from attack import *
from modelUtils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from ipdb import set_trace
import os
import pickle
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import gc
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_chunk(name, numTraces, arr, patch_idx, base_dir="./baseLineInps"):
    save_dir = os.path.join(base_dir, str(numTraces))
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"{name}_patch{patch_idx}.pkl")
    with open(path, "wb") as f:
        pickle.dump(arr, f)

    print(f"saved {path} shape={arr.shape}")


def batch_to_realInps(layer, layerInput):
    if isinstance(layer, nn.Conv2d):
        # Store ALL input channels' kH x kW patch (in_ch*kH*kW cols), laid out
        # C-order as [ch0(kHkW), ch1(kHkW), ...]. For a depthwise conv each filter
        # f reads only channel f -> the worker slices cols [f*kHkW:(f+1)*kHkW].
        # For a normal conv (groups=1) in_channels == weight.shape[1], so unchanged.
        in_ch = layer.in_channels
        kH, kW = layer.kernel_size
        numWeights = in_ch * kH * kW

        return (
            layerInput[:, :in_ch, :kH, :kW]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float16)
            .reshape(-1, numWeights)
        )

    elif isinstance(layer, nn.Linear):
        numWeights = layer.in_features
        return (
            layerInput[:, :numWeights]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float16)
        )

    else:
        raise ValueError("Layer must be Conv2d or Linear")


def extractFilterHW(model, input_dataloader, numTraces, exp_name, chunk_size=4_500_000):
    base_dir = f"./baseLineInps/{exp_name}"

    for name, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)) or name != 'features.1.0.block.0.0':
            continue

        print(f"Processing layer: {name}")

        buffer_list = []
        buffer_rows = 0
        patch_idx = 0

        model.eval()
        with torch.no_grad():
            for batch in input_dataloader:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                inputs = inputs.to(device).half()

                _ = model(inputs)
                layerInput, _ = intermediateActivations[layer]

                arr = batch_to_realInps(layer, layerInput)
                buffer_list.append(arr)
                buffer_rows += arr.shape[0]

                if buffer_rows >= chunk_size:
                    combined = np.concatenate(buffer_list, axis=0)

                    save_chunk(
                        name=name,
                        numTraces=numTraces,
                        arr=combined[:chunk_size],
                        patch_idx=patch_idx,
                        base_dir=base_dir
                    )
                    patch_idx += 1

                    remain = combined[chunk_size:]
                    buffer_list = [remain] if len(remain) > 0 else []
                    buffer_rows = remain.shape[0]

                    del combined, remain
                    gc.collect()

                del inputs, _, layerInput, arr
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if buffer_rows > 0:
            combined = np.concatenate(buffer_list, axis=0)
            save_chunk(
                name=name,
                numTraces=numTraces,
                arr=combined,
                patch_idx=patch_idx,
                base_dir=base_dir
            )
            del combined
            gc.collect()

        sys.stdout.flush()



def runFilterHW(model, input_dataloader, numTraces, exp_name, chunk_size=4_500_000):
    hooks = setupHooks(model)
    extractFilterHW(model, input_dataloader, numTraces, exp_name, chunk_size)
    removeHooks(hooks)


def setupHooksMulti(model, target_names):
    """
    Register a forward hook on each named layer, capturing its INPUT activation.

    Generalizes setupHooks() in modelUtils.py, which is hardcoded to
    'features.1.0.block.0.0' and therefore cannot capture the first conv.
    """
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


def extractFilterHWMulti(model, input_dataloader, numTraces, exp_name, target_names,
                         chunk_size=4_500_000, log_every=50):
    """
    ONE forward pass over the dataloader capturing the baseline (real) input
    activations of several layers at once, written with the same chunking and
    naming as extractFilterHW so the existing workers read them unchanged:
        ./baseLineInps/<exp_name>/<numTraces>/<layer>_patch<k>.pkl

    Single-pass matters here: at 12.24M traces, one pass per layer would double
    an already multi-hour job.
    """
    base_dir = f"./baseLineInps/{exp_name}"
    hooks, activations, layers = setupHooksMulti(model, target_names)

    buffers = {name: [] for name in target_names}
    buffer_rows = {name: 0 for name in target_names}
    patch_idx = {name: 0 for name in target_names}

    total = len(input_dataloader.dataset)
    t_start = time.time()
    seen = 0

    print(f"Processing layers: {list(target_names)}", flush=True)

    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(input_dataloader):
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device).half()

            _ = model(inputs)

            seen += inputs.shape[0]
            if bi % log_every == 0:
                el = time.time() - t_start
                ips = seen / el if el > 0 else 0
                eta = (total - seen) / ips / 60 if ips > 0 else float('inf')
                print(f"[progress] batch {bi} | {seen}/{total} | {ips:.0f} smp/s | "
                      f"elapsed {el/60:.1f} min | ETA {eta:.1f} min", flush=True)

            for name in target_names:
                arr = batch_to_realInps(layers[name], activations[name])
                buffers[name].append(arr)
                buffer_rows[name] += arr.shape[0]

                if buffer_rows[name] >= chunk_size:
                    combined = np.concatenate(buffers[name], axis=0)
                    save_chunk(name=name, numTraces=numTraces,
                               arr=combined[:chunk_size],
                               patch_idx=patch_idx[name], base_dir=base_dir)
                    patch_idx[name] += 1

                    remain = combined[chunk_size:]
                    buffers[name] = [remain] if len(remain) > 0 else []
                    buffer_rows[name] = remain.shape[0]
                    del combined, remain
                    gc.collect()

            del inputs, _
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for name in target_names:
        if buffer_rows[name] > 0:
            combined = np.concatenate(buffers[name], axis=0)
            save_chunk(name=name, numTraces=numTraces, arr=combined,
                       patch_idx=patch_idx[name], base_dir=base_dir)
            del combined
            gc.collect()

    for h in hooks:
        h.remove()

    sys.stdout.flush()


def runFilterHWMulti(model, input_dataloader, numTraces, exp_name, target_names,
                     chunk_size=4_500_000):
    extractFilterHWMulti(model, input_dataloader, numTraces, exp_name,
                         target_names, chunk_size)
