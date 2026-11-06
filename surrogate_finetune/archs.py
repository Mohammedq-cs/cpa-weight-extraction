"""Architecture registry: ResNet-18, VGG-11-BN and the small CIFAR CNN.

One `Arch` record per victim holds everything model-specific the fine-tuning
sweep and the PGD attack need: constructor, victim checkpoint, recovered
(CPA-extracted) weights, normalization stats, layer count and init.

MODEL CLASSES AND VICTIM CHECKPOINTS ARE TAKEN FROM THE SIBLING PIPELINES,
NOT COPIED: ../cpa_resnet (ResNet-18), ../vgg_e2e (VGG-11-BN) and
../cpa_sim_multi_patch (small CNN). Every sibling ships its own convNet.py, so
they are imported under unique module names with importlib instead of sys.path.

NORMALIZATION IS PER-ARCHITECTURE AND LOAD-BEARING. The small CNN was trained
under Normalize(0.5); scoring it with the CIFAR-10 channel stats costs 14.4
points (84.36 -> 69.95). Nothing crashes when it is wrong -- a cell just never
reaches its target -- so `validate()` checks the registry against the model.
"""

import hashlib
import importlib.util
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMIT = os.path.dirname(HERE)
RESNET_SRC = os.path.join(SUBMIT, 'cpa_resnet', 'convNet.py')
VGG_SRC = os.path.join(SUBMIT, 'vgg_e2e', 'convNet.py')
CNN_SRC = os.path.join(SUBMIT, 'cpa_sim_multi_patch', 'convNet.py')

CIFAR10_NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
HALF_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# Labeled-set sizes of the sweep (class-balanced, 1..5000 images per class).
N_VALUES = (50000, 20000, 10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10)


def _load_module(path, name):
    """Import a sibling pipeline's module under a name that cannot shadow ours."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"architecture source not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _resnet18():
    return _load_module(RESNET_SRC, '_cpa_resnet18_convnet').resnet18()


def _vgg11bn():
    return _load_module(VGG_SRC, '_cpa_vgg11bn_convnet').cifar10_vgg11_bn()


def _cnn8():
    return _load_module(CNN_SRC, '_cpa_cnn8_convnet').Net()


def _init_resnet18(layer):
    """ResNet-18's own constructor init: Kaiming fan_out for convs,
    reset_parameters() for Linear."""
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    else:
        layer.reset_parameters()


def _init_vgg11bn(layer):
    """VGG's own init: Kaiming for conv, N(0, 0.01) for Linear."""
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    else:
        nn.init.normal_(layer.weight, 0, 0.01)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def _init_cnn8(layer):
    """The CNN's own constructor init: PyTorch defaults for Conv2d and Linear."""
    layer.reset_parameters()


@dataclass(frozen=True)
class Arch:
    name: str
    builder: Callable[[], nn.Module]
    checkpoint: str                 # victim state_dict
    recovered: str                  # CPA-recovered conv/linear weights (build_recovered.py)
    norm_mean: Sequence[float]
    norm_std: Sequence[float]
    n_layers: int                   # conv/linear layers the sweep walks
    expected_clean_acc: float       # victim, fp16 -- the default --target
    expected_recovered_acc: float   # as-recovered, fp16 -- a smoke-test gate
    stop_threshold: float           # sweep early-stop: victim - 1 std of a 10k-sample estimate
    src: str                        # model-definition file, for provenance
    init_layer: Callable[[nn.Module], None]
    unpaired_bn: tuple = ()         # BN modules legitimately not conv-paired
    k_values: tuple = ()            # K = leading layers kept at their recovered values; -1 = scratch

    def grid(self):
        return [(k, n) for k in self.k_values for n in N_VALUES]

    def provenance(self):
        """Hashes of what this cell actually ran, recorded in the result JSON."""
        return {'arch_checkpoint_sha256': _sha256(self.checkpoint),
                'arch_src_sha256': _sha256(self.src)}


RESNET18 = Arch(
    name='resnet18',
    builder=_resnet18,
    checkpoint=os.path.join(SUBMIT, 'cpa_resnet', 'resnet18.pt'),
    recovered=os.path.join(HERE, 'models', 'recovered_resnet18.pt'),
    norm_mean=CIFAR10_NORM[0], norm_std=CIFAR10_NORM[1],
    n_layers=21,
    expected_clean_acc=92.59,
    expected_recovered_acc=10.38,   # layer4.1.conv2 / fc failed to extract
    stop_threshold=92.33,
    src=RESNET_SRC,
    init_layer=_init_resnet18,
    k_values=tuple(range(20, -2, -1)),
)

VGG11BN = Arch(
    name='vgg11bn',
    builder=_vgg11bn,
    checkpoint=os.path.join(SUBMIT, 'vgg_e2e', 'cifar10_vgg11_bn-eaeebf42.pt'),
    recovered=os.path.join(HERE, 'models', 'recovered_vgg11bn.pt'),
    norm_mean=CIFAR10_NORM[0], norm_std=CIFAR10_NORM[1],
    n_layers=11,
    expected_clean_acc=92.78,
    expected_recovered_acc=92.79,   # extraction is perfect at 20k traces
    stop_threshold=92.52,
    src=VGG_SRC,
    init_layer=_init_vgg11bn,
    k_values=(10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1),
)

CNN8 = Arch(
    name='cnn8',
    builder=_cnn8,
    checkpoint=os.path.join(SUBMIT, 'cpa_sim_multi_patch', 'model_cifar.pt'),
    recovered=os.path.join(HERE, 'models', 'recovered_cnn8.pt'),
    norm_mean=HALF_NORM[0], norm_std=HALF_NORM[1],   # NOT the CIFAR-10 stats
    n_layers=8,
    expected_clean_acc=84.36,
    expected_recovered_acc=84.36,   # extraction is perfect at 150k traces
    stop_threshold=84.00,
    src=CNN_SRC,
    init_layer=_init_cnn8,
    unpaired_bn=('fc1_bn',),        # BatchNorm1d, and dead code in Net.forward
    k_values=(7, 6, 5, 4, 3, 2, 1, 0, -1),
)

ARCHS = {a.name: a for a in (RESNET18, VGG11BN, CNN8)}


def validate(arch):
    """Fail before any GPU time if the registry disagrees with what is on disk."""
    import bnPrefix
    from common import collect_conv_linear_layers

    model = arch.builder()
    _, layer_order = collect_conv_linear_layers(model)
    if len(layer_order) != arch.n_layers:
        raise SystemExit(f"{arch.name}: registry says {arch.n_layers} layers, "
                         f"model has {len(layer_order)}: {layer_order}")
    paired = bnPrefix.paired_bn_names(model)
    all_bn = {n for n, m in model.named_modules()
              if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))}
    unpaired = tuple(sorted(all_bn - {v for v in paired.values() if v}))
    if unpaired != tuple(sorted(arch.unpaired_bn)):
        raise SystemExit(f"{arch.name}: unpaired BatchNorm {unpaired} != "
                         f"registry {tuple(sorted(arch.unpaired_bn))}")
    for path in (arch.checkpoint, arch.recovered):
        if not os.path.exists(path):
            raise SystemExit(f"{arch.name}: missing {path}")
    print(f"{arch.name}: {arch.n_layers} layers, unpaired BN {unpaired or '()'}, "
          f"victim {arch.expected_clean_acc} -- OK")
    return layer_order


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Validate the architecture registry.")
    p.add_argument('--arch', default=None, choices=sorted(ARCHS))
    a = p.parse_args()
    for arch in ([ARCHS[a.arch]] if a.arch else ARCHS.values()):
        validate(arch)
