import numpy as np

import random

from utils import *

from attack import *

from datasets import *

from modelUtils import *


import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, Subset

import torchvision.models as models

import argparse
from generateFilterHW import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# First conv (in_ch=3, 3x3 -> 27 cols) and first depthwise conv
# (in_ch=32 stored, sliced to 9 per filter downstream). Both are needed for
# the paper's random-model layer-1 + layer-2 results.
TARGET_LAYERS = ['features.0.0', 'features.1.0.block.0.0']

class RandomUniformDataset(Dataset):

    """

    Reproducible random dataset.

    Each index always returns the same sample for a given seed.

    """

    def __init__(self, num_samples, shape=(3, 224, 224), low=-2.0, high=2.0, seed=8):

        self.num_samples = num_samples

        self.shape = shape

        self.low = low

        self.high = high

        self.seed = seed

    def __len__(self):

        return self.num_samples

    def __getitem__(self, idx):

        rng = np.random.default_rng(self.seed + idx)

        x = rng.uniform(self.low, self.high, size=self.shape).astype(np.float16)

        x = torch.from_numpy(x)

        y = 0  # dummy label

        return x, y

def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)

    random.seed(worker_seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run CPA on EfficientNet-B0.")

    parser.add_argument("--model_save_path", required=True, help="Path to save your extracted model.")

    parser.add_argument("--num_traces", type=int, required=True)

    parser.add_argument('--noise_level', type=float, default=0.0)

    parser.add_argument(

        '--leakage_model',

        type=str,

        default='HW',

        choices=['HW', 'HD']

    )

    parser.add_argument(

        '--exper_name',

        type=str,

        default='accuracy_experiment'

    )

    parser.add_argument(

        '--dataset_name',

        type=str,

        default='random',

        choices=['random']

    )

    parser.add_argument('--seed', type=int, default=8)

    parser.add_argument('--low', type=float, default=-2.0)

    parser.add_argument('--high', type=float, default=2.0)

    parser.add_argument('--target_size', type=int, default=12_240_000)

    args = parser.parse_args()

    traces_use = int(args.num_traces)

    model_save_path = args.model_save_path

    noise_level = args.noise_level

    leakage_model = args.leakage_model

    exper_name = args.exper_name

    dataset_name = args.dataset_name

    seed = args.seed

    low = args.low

    high = args.high

    target_size = args.target_size

    # -------------------------------

    # Reproducibility

    # -------------------------------

    torch.manual_seed(seed)

    np.random.seed(seed)

    random.seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    print(f"Using deterministic CuDNN: {torch.backends.cudnn.deterministic}")

    g = torch.Generator()

    g.manual_seed(seed)

    # -------------------------------

    # Load EfficientNet-B0

    # -------------------------------

    torch.manual_seed(seed)

    model = models.efficientnet_b0(

        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1

    ).half().to(device)

    model.eval()

    with torch.no_grad():

        for param in model.parameters():

            param.uniform_(-5.0, 5.0)

    print(f"Seed = {seed}")

    print(f"CuDNN enabled = {torch.backends.cudnn.enabled}")

    # Save the random uniform(-5,5) ground-truth weights to a DEDICATED dir so we
    # never clobber ./model_weights (which is the ImageNet pretrained ground truth).
    save_model_weights(model, './model_weights_random')

    # -------------------------------

    # Random uniform dataset in [low, high]

    # EfficientNet expects 3x224x224 input

    # -------------------------------

    final_dataset = RandomUniformDataset(

        num_samples=target_size,

        shape=(3, 224, 224),

        low=low,

        high=high,

        seed=seed

    )

    print(f"final_dataset size = {len(final_dataset)}")

    if traces_use < len(final_dataset):

        final_dataset = Subset(final_dataset, range(traces_use))

        print(f"Using only first {len(final_dataset)} traces")

    # -------------------------------

    # Input loader

    # -------------------------------

    train_loader = DataLoader(

        final_dataset,

        batch_size=1024,

        shuffle=False,

        num_workers=40,

        worker_init_fn=seed_worker,

        generator=g

    )

    exp_name = f'seed{seed}/{exper_name}/{leakage_model}/{dataset_name}'

    input_tensor = train_loader

    numTraces = len(train_loader.dataset)

    print(f"\nnumTraces = {numTraces}, noiseLevel = {noise_level}, leakageModel = {leakage_model}\n")

    bsInpsPath = f'{exper_name}/{dataset_name}'

    # Capture BOTH target layers in ONE pass. runFilterHW() only ever captured
    # 'features.1.0.block.0.0' (modelUtils.setupHooks is hardcoded to it), which
    # left no way to regenerate the first-layer random-input baseline that the
    # paper's 85.76% result needs.
    runFilterHWMulti(model, input_tensor, numTraces, exp_name=bsInpsPath,
                     target_names=TARGET_LAYERS)
    # Baseline generation is the whole job here. Exit 0 -- this is the SUCCESS path, and exiting
    # non-zero makes SLURM report the job FAILED and breaks `--dependency=afterok:`.
    print("Done: baseline inputs + ground-truth weights generated. "
          "Exiting (CPA attack runs separately).")
    exit(0)
