#!/bin/bash
#SBATCH --job-name=e2eN040k
#SBATCH --output=./N0/S8/vgg_e2e/cifar6m/HD/40k/exper.out
#SBATCH --error=./N0/S8/vgg_e2e/cifar6m/HD/40k/exper.err
#SBATCH --time=1400
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60000
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1

# The orchestrator runs baseline generation + per-layer forward passes here and
# submits the per-filter array jobs. Memmap inputs keep both the orchestrator
# and the per-filter jobs at low RSS, so 60GB is plenty (no 130GB reservation).

export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64/

python demoSignAmbiguous.py --model_save_path ./N0/S8/vgg_e2e/cifar6m/HD/40k/seed8ModelN040k.pth \
               --num_traces 40000 \
               --noise_level 0 \
               --leakage_model HD \
               --exper_name vgg_e2e \
               --dataset_name cifar6m \
               --seed 8
