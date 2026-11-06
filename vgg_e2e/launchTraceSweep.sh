#!/bin/bash
# Launch one end-to-end extraction orchestrator per trace count.
#
# Each trace count gets its own Slurm job (the orchestrator in
# demoSignAmbiguous.py, same body as runSignAmbiguous.sbatch). The orchestrator
# in turn fans out the per-filter array jobs. Outputs land under
#   ./N0/S8/vgg_e2e/cifar6m/HD/<label>/
# which is what graphs_code/bitwise_acc_up_to_failure.ipynb reads back.
#
# Usage:  ./launchTraceSweep.sh                  # default sweep below
#         ./launchTraceSweep.sh 500 1000 5000    # custom trace counts
set -euo pipefail

# Default sweep: 500, 1k, 2k, 5k, 10k, 20k, 40k (the points in the figure).
TRACE_COUNTS=("$@")
if [ ${#TRACE_COUNTS[@]} -eq 0 ]; then
    TRACE_COUNTS=(500 1000 2000 5000 10000 20000 40000)
fi

NOISE_LEVEL=0
LEAKAGE_MODEL=HD
EXPER_NAME=vgg_e2e
DATASET_NAME=cifar6m
SEED=8

# <label>: matches the notebook's label() — "<n>k" for >=1000, else the raw count.
label() {
    if [ "$1" -ge 1000 ]; then echo "$(( $1 / 1000 ))k"; else echo "$1"; fi
}

for NUM_TRACES in "${TRACE_COUNTS[@]}"; do
    LABEL=$(label "$NUM_TRACES")
    RUN_DIR="./N0/S${SEED}/${EXPER_NAME}/${DATASET_NAME}/${LEAKAGE_MODEL}/${LABEL}"
    MODEL_SAVE_PATH="${RUN_DIR}/seed${SEED}ModelN0${LABEL}.pth"
    mkdir -p "$RUN_DIR"

    echo "submitting ${LABEL} (${NUM_TRACES} traces) -> ${RUN_DIR}"
    sbatch \
        --job-name="e2eN0${LABEL}" \
        --output="${RUN_DIR}/exper.out" \
        --error="${RUN_DIR}/exper.err" \
        --time=1400 \
        --signal=USR1@120 \
        --nodes=1 \
        --ntasks=1 \
        --mem=60000 \
        --cpus-per-task=1 \
        --gpus=1 \
        --wrap="export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64/ ; \
python demoSignAmbiguous.py \
    --model_save_path ${MODEL_SAVE_PATH} \
    --num_traces ${NUM_TRACES} \
    --noise_level ${NOISE_LEVEL} \
    --leakage_model ${LEAKAGE_MODEL} \
    --exper_name ${EXPER_NAME} \
    --dataset_name ${DATASET_NAME} \
    --seed ${SEED}"
done
