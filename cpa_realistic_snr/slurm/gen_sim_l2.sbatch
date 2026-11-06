#!/bin/bash
#SBATCH --job-name=simL2
#SBATCH --output=./scriptsOut/simL2_%j.out
#SBATCH --error=./scriptsErr/simL2_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300000
#SBATCH --gpus=1
#SBATCH --partition=${SLURM_PARTITION:-gpu}
#SBATCH --constraint=${GPU_CONSTRAINT:-}   # e.g. a specific GPU model
#SBATCH --exclude=${BAD_NODES:-}   # set to any nodes with faulty GPUs

# Phase 3: layer-2 hypothesis inputs, generated from the model carrying
# the EXTRACTED layer-1 weights (this is what makes the chain error-propagating).
# Reuses the baseline's layer-2 patch selection so sim/baseline rows align 1:1.
# Depends on the sigma=850 sequential layer-1 result existing. ~59 GB.

SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$PWD}"
PY="${PY:-python}"

# Some clusters lack a system CUDA matching CuPy's build, which fails at import with
# "libnvrtc.so.12: cannot open shared object file". The conda env ships its own copy,
# so prepend it and the job becomes node-independent.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib:${CUDA_HOME:-/usr/local/cuda-12.2}/lib64/"

cd "$RUN"

$PY "$SRC/tools/assert_gpu.py" || exit 1

$PY "$SRC"/gen_sim_layer2.py \
    --num_traces 12240000 --target_size 12240000 --seed 8 \
    --l1_out_dir   "./N850.0/seed8_mp/HW/imageNet12m_mpOuputs/12240000" \
    --baseline_dir "./baseLineInps/seed8/imageNet12m_mp/HW/imageNet/imageNet/12240000" \
    --sim_base_dir "./simInps/seed8_mp/HW/imageNet/imageNet" \
    --num_workers 48 --batch_size 1024
