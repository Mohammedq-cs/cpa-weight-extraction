"""
Layer-input capture + per-weight patch selection for the in-process CPA pipeline.

  * capture_layer_input -- forward the model and return one layer's input
    activation (CPU fp16).
  * compute_selected_patches -- for each Conv2d weight, the spatial kernel
    application (patch) whose input column has the most distinct float16 values
    (the best CPA target), decided over ALL traces on the GPU. Returns the sorted
    union over weights of that argmax patch; padding/edge patches that are
    degenerate for a weight are never that weight's argmax, so only patches that
    carry signal for some weight survive.

The selection is computed over EVERY trace via gpuUniqueCount.gpu_best_patch_per_weight
(the same method as ../cpa_resnet): the distinct-count is done on the GPU with a packed
presence bitset, and the unfold is streamed in trace-batches so memory is bounded
while using every trace.

Selection is caller-neutral -- it selects from whatever activation it is given.
The in-process pipeline (simulationInproc.load_or_compute_ucm) passes the
SIMULATED/reconstructed model's activation, since a real attacker only has the
reconstructed model, not the true weights. Selected patches are then served on
demand from RAM by columnProvider.ColumnProvider; nothing is materialised to disk.
"""

import numpy as np
import torch

from modelUtils import intermediateActivations

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Layer-input capture
# ============================================================

def capture_layer_input(model, input_tensor, layer, batch_size=10_000):
    """Run the model in its current state and return one layer's concatenated
    input activation (CPU tensor). Assumes hooks are already installed."""
    model.eval()
    parts = []
    with torch.no_grad():
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        for batch in torch.split(input_tensor, batch_size, dim=0):
            batch = batch.to(device, non_blocking=True).half()
            _ = model(batch)
            layer_inp, _ = intermediateActivations[layer]
            parts.append(layer_inp.detach().cpu().clone())
    return torch.cat(parts, dim=0)


# ============================================================
# Per-weight patch selection  (over ALL traces, on the GPU)
#   Caller-neutral: selects from whatever activation it is given. The in-process
#   pipeline (simulationInproc.load_or_compute_ucm) passes the SIMULATED
#   (reconstructed) model's activation -- a real attacker only has the
#   reconstructed model, not the true weights.
# ============================================================

def compute_selected_patches(layer_inp, layer):
    """Sorted union over weights of each weight's most-unique patch, decided over
    ALL traces on the GPU (gpuUniqueCount.gpu_best_patch_per_weight)."""
    from gpuUniqueCount import gpu_best_patch_per_weight
    best_patch_per_weight, _ = gpu_best_patch_per_weight(layer_inp, layer)
    return np.unique(best_patch_per_weight).astype(np.int32)
