# CPA Weight-Extraction — Paper Code

Code accompanying the paper on **Correlation Power Analysis (CPA) weight extraction** of
convolutional neural networks. Each subfolder is a self-contained pipeline with its own
README; start there for setup and run instructions.

## Contents

| Folder | What it does |
|---|---|
| [`cpa_sim_first_patch/`](cpa_sim_first_patch/) | CPA weight-recovery simulation on a small CNN (CIFAR-10 / CIFAR-5M) using a **single / first patch** per layer. The reference pipeline + paper-figure notebooks. |
| [`cpa_sim_multi_patch/`](cpa_sim_multi_patch/) | Same attack with **per-weight multi-patch selection**, run **single-node in-process** (one SLURM allocation, optional multi-GPU). |
| [`vgg_e2e/`](vgg_e2e/) | **VGG-11 end-to-end**: sign-ambiguous CPA extraction across every layer followed by reactive sign correction. |
| [`cpa_resnet/`](cpa_resnet/) | **ResNet-18 layer-by-layer**: CPA weight recovery across the full network under the *first-weight-known* assumption, with per-weight multi-patch selection decided over **all** traces, and recovered weights propagated layer to layer. |
| [`surrogate_finetune/`](surrogate_finetune/) | **Surrogates from a partial extraction**: keep the first *K* layers of ResNet-18 / VGG-11-BN / the CIFAR CNN frozen at the weights recovered by the CPA attacks (the extracted models of [`cpa_resnet/`](cpa_resnet/), [`vgg_e2e/`](vgg_e2e/) and [`cpa_sim_multi_patch/`](cpa_sim_multi_patch/)), re-initialize and fine-tune the rest on *N* labeled CIFAR-10 images; smallest *N* that recovers the victim's accuracy vs. layers fine-tuned, and PGD (eps = 5/255) transferability of those surrogates to the victim (median over 3 training seeds). |
| [`cpa_realistic_snr/`](cpa_realistic_snr/) | **Section 8.1 — realistic SNR**: CPA weight recovery of EfficientNet-B0 (ImageNet and random inputs) and the CIFAR CNN at the calibrated noise level σ=850 over 12.24M traces, sequential vs GT-oracle, plus the chained layer-2 attack. |

## Common setup

All pipelines share the same environment and conventions:

- **Python 3.9.19**, **PyTorch / torchvision**, **NumPy**, **CuPy** (CUDA 12.x), **pandas**,
  **matplotlib**. All pinned dependencies are in the single top-level
  [`requirements.txt`](requirements.txt) (shared by all folders):
  ```bash
  conda create -n cpa python=3.9.19 && conda activate cpa
  pip install -r requirements.txt
  ```
- A **GPU node with CUDA 12.2** (`LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64/`).
- **Slurm** for fanning out per-filter jobs (configure `slurm_config.py` in each folder to
  match your cluster — `nodelist` / `partition` / `constraint`).
- **Datasets:** downloaded **once** into `data/` at the repo root and shared by all folders
  (see [Datasets](#datasets) below).
- **Leakage models:** Hamming Weight (`HW`) and Hamming Distance (`HD`).

Pre-trained / reference model weights needed to run are bundled inside each folder. Large
run outputs, Slurm logs, and dataset caches are intentionally not committed (see
`.gitignore`).

See each subfolder's `README.md` for the exact entry points, flags, and figure-generation
steps.

## Datasets

All folders read the same two datasets from `data/` at the repo root (set `CPA_DATA=<dir>` to
use another location). Download them once, from the repo root:

```bash
mkdir -p data
# CIFAR-10 (python version) -> data/cifar-10-batches-py/
python -c "import torchvision; torchvision.datasets.CIFAR10('data', download=True)"
# CIFAR-5M: https://github.com/preetum/cifar5m -> data/cifar5m/part0.npz ... part5.npz
```

The scripts pass CIFAR-5M as `--dataset_name cifar6m` (CIFAR-5M + CIFAR-10 train).
`cpa_realistic_snr` additionally needs ImageNet (`IMAGENET_ROOT`, see its README).
