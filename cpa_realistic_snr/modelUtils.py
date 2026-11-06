import torch
import torch.nn as nn
import numpy as np

# Hook function to store the intermediate outputs
intermediateActivations = {}

def hookFn(module, input, output):
    """This function will be called during forward pass to capture input and output."""
    intermediateActivations[module] = (input[0].detach().cpu(), output.detach().cpu())

def setupHooks(model: nn.Module) -> list:
    """
    Registers forward hooks for all convolutional and linear layers in the model.

    Args:
        model (nn.Module): The neural network model.

    Returns:
        list: A list of hook handles.
    """
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if name == 'features.1.0.block.0.0':
                hook = layer.register_forward_hook(hookFn)
                hooks.append(hook)
    return hooks

def removeHooks(hooks: list) -> None:
    """
    Removes all registered hooks.

    Args:
        hooks (list): List of hook handles.
    """
    for hook in hooks:
        hook.remove()

def updateLayerWeights(layer: nn.Module, extractedLayer: np.ndarray):
    expected_shape = layer.weight.data.shape
    reshaped_extractedLayer = extractedLayer.reshape(expected_shape)

    # Convert extractedLayer to a torch tensor with the same dtype and device as the layer's weights
    new_weights = torch.from_numpy(reshaped_extractedLayer).to(layer.weight.data.dtype).to(layer.weight.data.device)

    # Update the layer's weights without tracking gradients
    with torch.no_grad():
        layer.weight.data.copy_(new_weights)

def add_noise_to_model_weights(model, noise_std=5.0):  # You can make noise even larger if you want
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            with torch.no_grad():
                # Generate noise of the same shape as the weights
                noise = torch.randn_like(module.weight) * noise_std
                module.weight.add_(noise)

def zero_out_model_weights(model):  
    """
    Set all weights of Conv/Linear layers in the model to zero.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            with torch.no_grad():
                module.weight.zero_()  # directly zero out the weights

