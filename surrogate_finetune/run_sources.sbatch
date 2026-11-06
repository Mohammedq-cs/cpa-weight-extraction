#!/bin/bash
#SBATCH --job-name=ftSources
#SBATCH --output=./logs/sources_%a.out
#SBATCH --error=./logs/sources_%a.err
#SBATCH --time=240
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --array=0-119
# Set --nodelist / --partition / --constraint to match your cluster, e.g.:
# #SBATCH --nodelist=<your-nodes>
# #SBATCH --partition=<your-partition>

# Re-trains, WITH a saved model, the surrogates the transferability figure
# attacks: one line of cells/transfer_sources.txt per array task
# (ARCH K N MODE THRESHOLD SEED). Same recipe as the sweep cell (K, N), plus:
#   MODE=threshold -> save the first epoch with test acc >= THRESHOLD (victim - 1.1 or
#                     victim - 1.0 per line, see cells/transfer_sources.txt)
#                     and stop;
#   MODE=best      -> the depth never reached it (VGG-11, 8 layers fine-tuned):
#                     run the full budget, save the highest-test-accuracy epoch.
# Seeds 9 and 10 re-draw the labeled subset and the initialization.
#
#   mkdir -p logs && sbatch run_sources.sbatch
#
# -> results/{arch}/transfer_sources_seed{SEED}/{cell}.json + checkpoints/{cell}_ep{E}.pt

export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64/
export PYTHONUNBUFFERED=1

CELLS=${CELLS:-cells/transfer_sources.txt}
read -r ARCH K N MODE THR SEED < <(grep -v '^#' "$CELLS" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ "$MODE" = threshold ]; then SEL=(--checkpoint_at_threshold "$THR"); else SEL=(--checkpoint_best --stop_after_target 0); fi

echo "node $(hostname)  $ARCH K=$K N=$N $MODE $THR seed=$SEED"
python -u sweepRepair.py \
    --arch "$ARCH" --num_recovered "$K" --train_size "$N" \
    --max_samples 5000000 --max_epochs 300 --seed "$SEED" \
    --num_workers 4 --checkpoint_dtype fp32 "${SEL[@]}" \
    --out_dir "./results/$ARCH/transfer_sources_seed$SEED"
