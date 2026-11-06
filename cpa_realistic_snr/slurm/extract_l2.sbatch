#!/bin/bash
#SBATCH --job-name=L2
#SBATCH --output=./scriptsOut/L2_%A_%a.out
#SBATCH --error=./scriptsErr/L2_%A_%a.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60000
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --array=0-31%32
#SBATCH --partition=${SLURM_PARTITION:-gpu}
#SBATCH --constraint=${GPU_CONSTRAINT:-}   # e.g. a specific GPU model
#SBATCH --exclude=${BAD_NODES:-}   # set to any nodes with faulty GPUs

# Phase 3: chained multipatch extraction of the depthwise
# features.1.0.block.0.0 at sigma=850. TRUE leakage from the baseline (real model)
# inputs; hypotheses from the sim inputs built on EXTRACTED layer-1 weights.
# Target: 5/288 = 0.0174, the paper's 1.74%.

SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$PWD}"
PY="${PY:-python}"

# Some clusters lack a system CUDA matching CuPy's build, which fails at import with
# "libnvrtc.so.12: cannot open shared object file". The conda env ships its own copy,
# so prepend it and the job becomes node-independent.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib:${CUDA_HOME:-/usr/local/cuda-12.2}/lib64/"

cd "$RUN"

BASE="./baseLineInps/seed8/imageNet12m_mp/HW/imageNet/imageNet/12240000"
SIM="./simInps/seed8_mp/HW/imageNet/imageNet/12240000"

$PY "$SRC/tools/assert_gpu.py" || exit 1

$PY "$SRC"/extractFilterMaterialized.py \
    --baseLineInputDir   "$BASE" \
    --simulatedInputsDir "$SIM" \
    --corrStatSavePath   "./N850.0/seed8_mp/HW/imageNet12m_mpCorr/12240000" \
    --outputDir          "./N850.0/seed8_mp/HW/imageNet12m_mpOuputs/12240000" \
    --layerName          "features.1.0.block.0.0" \
    --filterIdx          ${SLURM_ARRAY_TASK_ID} \
    --NoiseLevel         850 \
    --leakage_model      HW \
    --weightsDir         ./model_weights_imageNet
