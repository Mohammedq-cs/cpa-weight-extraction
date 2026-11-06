#!/bin/bash
#SBATCH --job-name=randBS
#SBATCH --output=./scriptsOut/randBS_%A_%a.out
#SBATCH --error=./scriptsErr/randBS_%A_%a.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60000
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --array=0-63%64
#SBATCH --partition=${SLURM_PARTITION:-gpu}
#SBATCH --constraint=${GPU_CONSTRAINT:-}   # e.g. a specific GPU model
#SBATCH --exclude=${BAD_NODES:-}   # set to any nodes with faulty GPUs

# Phase 2, Experiment A: the BarraCUDA calibration run.
# Random-init EfficientNet-B0 (uniform(-5,5)) + random uniform[-2,2] inputs,
# sigma=850, error propagation SUPPRESSED (oracle: prevSum advanced with TRUE
# previous weights). Fixed RF / legacy single-patch pipeline, as the original.
#   array 0-31  -> features.0.0            filter = idx      (paper: 85.76%)
#   array 32-63 -> features.1.0.block.0.0  filter = idx - 32 (paper: 80.56%)
# Oracle => simulatedInputsDir == baseLineInputDir.
#
# NOTE ON THE EXPECTED L2 NUMBER: our regenerated baseline is channel-CORRECTED by
# construction (288 cols, filter f reads channel f), so this is the "BSfix"
# equivalent. The reference BSfix run scores 220/288 = 0.7639, NOT the paper's
# 0.8056 -- the paper's figure came from the pre-bugfix 9-col baseline. Expect ~76%.

SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$PWD}"
PY="${PY:-python}"

# Some clusters lack a system CUDA matching CuPy's build, which fails at import with
# "libnvrtc.so.12: cannot open shared object file". The conda env ships its own copy,
# so prepend it and the job becomes node-independent.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib:${CUDA_HOME:-/usr/local/cuda-12.2}/lib64/"

cd "$RUN"

BASE="./baseLineInps/randInps/random/12240000"

if [ ${SLURM_ARRAY_TASK_ID} -lt 32 ]; then
    LAYER="features.0.0";            FIDX=${SLURM_ARRAY_TASK_ID}
else
    LAYER="features.1.0.block.0.0";  FIDX=$(( SLURM_ARRAY_TASK_ID - 32 ))
fi

$PY "$SRC/tools/assert_gpu.py" || exit 1

$PY "$SRC"/BSextractFilterChkptOptimized.py \
    --baseLineInputDir   "$BASE" \
    --simulatedInputsDir "$BASE" \
    --corrStatSavePath   "./N850.0/BS/HW/randInpsCorr/12240000" \
    --outputDir          "./N850.0/BS/HW/randInpsOuputs/12240000" \
    --layerName          "$LAYER" \
    --filterIdx          ${FIDX} \
    --NoiseLevel         850 \
    --leakage_model      HW \
    --weightsDir         ./model_weights_random
