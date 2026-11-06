#!/bin/bash
#SBATCH --job-name=cifar_x
#SBATCH --output=./scriptsOut/cifar_x_%A_%a.out
#SBATCH --error=./scriptsErr/cifar_x_%A_%a.err
#SBATCH --time=03:00:00   # CIFAR conv1 filters run 19-40+ min (4-patch adaptive RF); 40min timed one out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60000
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --array=0-31%32
#SBATCH --partition=${SLURM_PARTITION:-gpu}
#SBATCH --constraint=${GPU_CONSTRAINT:-}   # e.g. a specific GPU model
#SBATCH --exclude=${BAD_NODES:-}   # set to any nodes with faulty GPUs

# Phase 2 (CIFAR): conv1 extraction at sigma=850, HW, sequential
# (error-propagating). Target: 331/864 = 0.3831, the paper's 38.3%.

SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$PWD}"
PY="${PY:-python}"

# Some clusters lack a system CUDA matching CuPy's build, which fails at import with
# "libnvrtc.so.12: cannot open shared object file". The conda env ships its own copy,
# so prepend it and the job becomes node-independent.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib:${CUDA_HOME:-/usr/local/cuda-12.2}/lib64/"

cd "$RUN"

BASE="./baseLineInps/cifar12m_mp/cifar6m/12240000"

$PY "$SRC/tools/assert_gpu.py" || exit 1

$PY "$SRC"/extractFilterMaterialized.py \
    --baseLineInputDir   "$BASE" \
    --simulatedInputsDir "$BASE" \
    --corrStatSavePath   "./N850.0/cifar12m_mp/HW/cifar6mCorr/12240000" \
    --outputDir          "./N850.0/cifar12m_mp/HW/cifar6mOuputs/12240000" \
    --layerName          "conv1" \
    --filterIdx          ${SLURM_ARRAY_TASK_ID} \
    --NoiseLevel         850 \
    --leakage_model      HW \
    --weightsDir         ./model_weights
