#!/bin/bash
#SBATCH --job-name=cifar_gen
#SBATCH --output=./scriptsOut/cifar_gen_%j.out
#SBATCH --error=./scriptsErr/cifar_gen_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=160000
#SBATCH --gpus=1
#SBATCH --partition=${SLURM_PARTITION:-gpu}
#SBATCH --constraint=${GPU_CONSTRAINT:-}   # e.g. a specific GPU model
#SBATCH --exclude=${BAD_NODES:-}   # set to any nodes with faulty GPUs

# Phase 1 (CIFAR): multipatch (adaptive RF) baseline for the CIFAR CNN's
# conv1 at 12.24M traces. conv1's input is the image, so this doubles as its sim
# inputs. ~2.6 GB. Runs in $RUN/cifar (see run_all.sh), next to the extraction job.
# CIFAR-10 / CIFAR-5M are read from the repo-wide data/ folder (datasets.py,
# override with CPA_DATA); the CIFAR CNN is
# ../cpa_sim_first_patch/model_cifar.pt, linked into $RUN/cifar below.

SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$PWD}"
PY="${PY:-python}"

# Some clusters lack a system CUDA matching CuPy's build, which fails at import with
# "libnvrtc.so.12: cannot open shared object file". The conda env ships its own copy,
# so prepend it and the job becomes node-independent.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib:${CUDA_HOME:-/usr/local/cuda-12.2}/lib64/"

cd "$RUN"


$PY "$SRC/tools/assert_gpu.py" || exit 1

[ -e model_cifar.pt ] || ln -s "$SRC/../cpa_sim_first_patch/model_cifar.pt" model_cifar.pt

$PY "$SRC"/demo_cifar.py \
    --model_save_path /tmp/cifar_unused.pth \
    --num_traces 12240000 \
    --noise_level 0 \
    --leakage_model HW \
    --exper_name cifar12m_mp \
    --dataset_name cifar6m \
    --seed 8 \
    --multipatch \
    --layers conv1 \
    --selection_images 8000
