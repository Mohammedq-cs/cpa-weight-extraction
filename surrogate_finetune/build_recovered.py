"""Pack one CPA extraction run's recovered weights into models/recovered_{arch}.pt.

The extraction pipelines write one pickle per recovered filter:

    {recovered_dir}/{layer}/filter_{i}.pkl        (float16 weights of filter i)

This collects them into {layer: float16 tensor} in the layer's weight shape, the
format common.load_recovered_model reads. The three files in models/ were built
from these runs, at the paths each folder's launcher writes (cpa_resnet/run200k.sbatch,
vgg_e2e/launchTraceSweepChained.sh, cpa_sim_multi_patch/run_zero_noise.sh):

    resnet18  ../cpa_resnet/N0.0/seed8/resnet_fwk/HD/cifar6moutput/200000
    vgg11bn   ../vgg_e2e/N0.0/seed8/vgg_e2e/HD/cifar6moutput/20000
    cnn8      ../cpa_sim_multi_patch/N0.0/seed8/model_extract/HD/cifar6moutput/150000

    python build_recovered.py --arch vgg11bn --recovered_dir ../vgg_e2e/N0.0/seed8/vgg_e2e/HD/cifar6moutput/20000
"""

import argparse
import os
import pickle

import numpy as np
import torch

import archs
from common import collect_conv_linear_layers


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arch', required=True, choices=sorted(archs.ARCHS))
    p.add_argument('--recovered_dir', required=True,
                   help='{recovered_dir}/{layer}/filter_{i}.pkl')
    p.add_argument('--out', default=None, help='Default: the registry path.')
    a = p.parse_args()
    arch = archs.ARCHS[a.arch]
    model = arch.builder()
    layers, layer_order = collect_conv_linear_layers(model)
    out = {}
    for name in layer_order:
        layer = layers[name]
        n = layer.out_channels if isinstance(layer, torch.nn.Conv2d) else layer.weight.shape[0]
        filters = []
        for i in range(n):
            with open(os.path.join(a.recovered_dir, name, f'filter_{i}.pkl'), 'rb') as f:
                filters.append(pickle.load(f))
        w = np.array(filters).reshape(tuple(layer.weight.shape))
        if w.dtype != np.float16:   # stored losslessly, never re-rounded
            raise TypeError(f"{name}: recovered weights are {w.dtype}, expected float16")
        out[name] = torch.from_numpy(w)
    path = a.out or arch.recovered
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(out, path)
    print(f"{arch.name}: {len(out)} layers -> {path}")


if __name__ == '__main__':
    main()
