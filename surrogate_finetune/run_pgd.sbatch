#!/bin/bash
#SBATCH --job-name=ftPgd
#SBATCH --output=./logs/pgd_%x.out
#SBATCH --error=./logs/pgd_%x.err
#SBATCH --time=240
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# Set --nodelist / --partition / --constraint to match your cluster, e.g.:
# #SBATCH --nodelist=<your-nodes>
# #SBATCH --partition=<your-partition>

# PGD transfer from one architecture's surrogates of one training seed to its
# victim: CIFAR-10 test set, L-inf eps = 5/255, alpha = 2/255, 75 steps, random
# start (attack seed 0), fp32 models + fp64 cross-entropy. White box first.
#
#   for A in resnet18 vgg11bn cnn8; do for S in 8 9 10; do
#     sbatch -J pgd_${A}_$S --export=ALL,ARCH=$A,SEED=$S run_pgd.sbatch; done; done
#
# -> results/{arch}/pgd_transfer_sources_eps5255_fp32_seed{SEED}/*_seed0.json

export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64/
export PYTHONUNBUFFERED=1

ARCH=${ARCH:?set ARCH=resnet18|vgg11bn|cnn8}
SEED=${SEED:?set SEED=8|9|10}
python -u pgdSourceSweep.py --arch "$ARCH" \
    --ckpt_results_dir "./results/$ARCH/transfer_sources_seed$SEED" \
    --out_dir "./results/$ARCH/pgd_transfer_sources_eps5255_fp32_seed$SEED" \
    --eps 0.0196078431372549 --alpha 0.00784313725490196 --steps 75 \
    --fp32 --attack_seeds 0
