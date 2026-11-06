#!/bin/bash
#SBATCH --job-name=ftSweep
#SBATCH --output=./logs/sweep_%x_%a.out
#SBATCH --error=./logs/sweep_%x_%a.err
#SBATCH --time=360
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=0-11
# Set --nodelist / --partition / --constraint to match your cluster, e.g.:
# #SBATCH --nodelist=<your-nodes>
# #SBATCH --partition=<your-partition>

# Figure "labeled images vs layers fine-tuned": the full K x N grid of one
# architecture on CIFAR-10 (K = its whole depth ladder incl. -1 = scratch,
# N = 10 .. 50000), seed 8, sharded over the array (strided, 12 shards):
#
#   mkdir -p logs
#   sbatch --export=ALL,ARCH=resnet18 run_sweep.sbatch
#   sbatch --export=ALL,ARCH=vgg11bn  run_sweep.sbatch
#   sbatch --export=ALL,ARCH=cnn8     run_sweep.sbatch
#
# Each cell writes results/{arch}/sweep_bnprefix/K{K}_N{N}_cifar10_bnprefix.json
# (its test-accuracy history). Finished cells are skipped, so a requeued shard
# resumes. A cell early-stops 10 epochs after crossing the architecture's
# stop threshold (victim - 1 std); the figure counts the first crossing of
# victim - 1.1, which always happens before that.

export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64/
export PYTHONUNBUFFERED=1

ARCH=${ARCH:?set ARCH=resnet18|vgg11bn|cnn8}
python -u sweepRepair.py --arch "$ARCH" \
    --shard_id "${SLURM_ARRAY_TASK_ID}" --num_shards 12 \
    --max_samples 5000000 --max_epochs 300 --seed 8 \
    --num_workers 4 --cache_dataset \
    --out_dir "./results/$ARCH/sweep_bnprefix"
