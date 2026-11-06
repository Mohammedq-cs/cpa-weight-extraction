#!/bin/bash
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

# Phase 2: EfficientNet features.0.0, multipatch (adaptive RF), HW,
# noise = $NOISE via --export. Mode: BS (BS).
# Submit: sbatch -J BS_N${n} --export=ALL,NOISE=${n} extract_l1_BS.sbatch

SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$PWD}"
PY="${PY:-python}"

# Some clusters lack a system CUDA matching CuPy's build, which fails at import with
# "libnvrtc.so.12: cannot open shared object file". The conda env ships its own copy,
# so prepend it and the job becomes node-independent.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib:${CUDA_HOME:-/usr/local/cuda-12.2}/lib64/"

cd "$RUN"

BASE="./baseLineInps/seed8/imageNet12m_mp/HW/imageNet/imageNet/12240000"

$PY "$SRC/tools/assert_gpu.py" || exit 1

$PY "$SRC"/extractFilterMaterialized.py \
    --baseLineInputDir   "$BASE" \
    --simulatedInputsDir "$BASE" \
    --corrStatSavePath   "./N${NOISE}.0/BS/HW/imageNet12m_mpCorr/12240000" \
    --outputDir          "./N${NOISE}.0/BS/HW/imageNet12m_mpOuputs/12240000" \
    --layerName          "features.0.0" \
    --filterIdx          ${SLURM_ARRAY_TASK_ID} \
    --NoiseLevel         ${NOISE} \
    --leakage_model      HW \
    --prev_known \
    --weightsDir         ./model_weights_imageNet
