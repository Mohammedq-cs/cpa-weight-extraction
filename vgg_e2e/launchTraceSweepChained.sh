#!/bin/bash
# Launch the end-to-end extraction orchestrator once per trace count, CHAINED so
# the runs execute strictly one after another instead of all at once.
#
# Same body as launchTraceSweep.sh, except each trace count's Slurm job is
# submitted with `--dependency=afterany:<previous job id>`, so trace count N+1
# only starts after trace count N has finished (success OR failure -- `afterany`,
# matching run_exps.sh elsewhere in this repo). Use this when you want the sweep
# to share the cluster politely (one orchestrator + its per-filter array at a
# time) rather than contending for GPUs in parallel.
#
# Each orchestrator (demoSignAmbiguous.py) still fans out its own per-filter
# array jobs. Outputs land under
#   ./N0/S8/vgg_e2e/cifar6m/HD/<label>/
# which is what graphs_code/bitwise_acc_up_to_failure.ipynb reads back.
#
# Usage:  ./launchTraceSweepChained.sh                  # default sweep below
#         ./launchTraceSweepChained.sh 500 1000 5000    # custom trace counts
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

# <label>: matches the notebook's label() -- "<n>k" for >=1000, else the raw count.
label() {
    if [ "$1" -ge 1000 ]; then echo "$(( $1 / 1000 ))k"; else echo "$1"; fi
}

prev_job_id=""

for NUM_TRACES in "${TRACE_COUNTS[@]}"; do
    LABEL=$(label "$NUM_TRACES")
    RUN_DIR="./N0/S${SEED}/${EXPER_NAME}/${DATASET_NAME}/${LEAKAGE_MODEL}/${LABEL}"
    MODEL_SAVE_PATH="${RUN_DIR}/seed${SEED}ModelN0${LABEL}.pth"
    mkdir -p "$RUN_DIR"

    # Chain on the previous trace count (afterany: run regardless of its exit state).
    DEP_ARGS=()
    if [ -n "$prev_job_id" ]; then
        DEP_ARGS=(--dependency="afterany:${prev_job_id}")
    fi

    job_id=$(sbatch --parsable \
        "${DEP_ARGS[@]}" \
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
    --seed ${SEED}")

    if [ -n "$prev_job_id" ]; then
        echo "submitted ${LABEL} (${NUM_TRACES} traces) -> job ${job_id} (after ${prev_job_id}) -> ${RUN_DIR}"
    else
        echo "submitted ${LABEL} (${NUM_TRACES} traces) -> job ${job_id} (head of chain) -> ${RUN_DIR}"
    fi
    prev_job_id=$job_id
done
