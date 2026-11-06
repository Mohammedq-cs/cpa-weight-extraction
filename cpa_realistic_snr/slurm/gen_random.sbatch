#!/bin/bash
#SBATCH --job-name=gen_rand
#SBATCH --output=./scriptsOut/gen_rand_%j.out
#SBATCH --error=./scriptsErr/gen_rand_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=100000
#SBATCH --gpus=1
#SBATCH --partition=${SLURM_PARTITION:-gpu}
#SBATCH --constraint=${GPU_CONSTRAINT:-}   # e.g. a specific GPU model
#SBATCH --exclude=${BAD_NODES:-}   # set to any nodes with faulty GPUs

# Phase 1 / repro of paper 8.1 Experiment A: uniform(-5,5) EfficientNet-B0 with
# random uniform[-2,2] inputs. Now captures BOTH features.0.0 (27 col) and
# features.1.0.block.0.0 (288 col) in ONE pass via runFilterHWMulti -- the old
# code only ever captured the depthwise layer, which is why the paper's 85.76%
# first-layer number had no live regeneration path. Ground truth -> ./model_weights_random.

SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$PWD}"
PY="${PY:-python}"

# Some clusters lack a system CUDA matching CuPy's build, which fails at import with
# "libnvrtc.so.12: cannot open shared object file". The conda env ships its own copy,
# so prepend it and the job becomes node-independent.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib:${CUDA_HOME:-/usr/local/cuda-12.2}/lib64/"

cd "$RUN"


$PY "$SRC/tools/assert_gpu.py" || exit 1

$PY "$SRC"/demo_random.py \
    --model_save_path /tmp/rand_unused.pth \
    --num_traces 12240000 \
    --target_size 12240000 \
    --noise_level 0 \
    --leakage_model HW \
    --exper_name randInps \
    --dataset_name random \
    --seed 8
