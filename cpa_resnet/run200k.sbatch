#!/bin/bash
#SBATCH --job-name=resnetN0200k
#SBATCH --output=./N0/S8/resnet_fwk/cifar6m/HD/200k/exper.out
#SBATCH --error=./N0/S8/resnet_fwk/cifar6m/HD/200k/exper.err
#SBATCH --time=1400
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60000
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
# Set --nodelist / --partition / --constraint to match your cluster, e.g.:
# #SBATCH --nodelist=<your-nodes>
# #SBATCH --partition=<your-partition>
# #SBATCH --constraint=geforce_rtx_3090

# Orchestrator (demo.py) for a single 200k-trace ResNet-18 extraction run. There
# is no up-front baseline capture: per layer it derives the patch selection from
# the SIMULATED (working-model) activations over ALL traces on the GPU, writes the
# simulated AND baseline patch memmaps for that selection together, caches the
# per-layer unique-count matrix, then fans out the per-filter array jobs (their
# Slurm resources come from slurm_config.py). Recovered weights are propagated
# layer to layer; final CIFAR-10 accuracy is printed at the end.
#
# Make the output dir first:  mkdir -p ./N0/S8/resnet_fwk/cifar6m/HD/200k
#
# To resume a run that died mid-way, add --start_at_layer <layerName> (earlier
# layers are loaded from their recovered pickles instead of re-extracted). Add
# --keep_layer_memmaps only for debugging: the per-layer patch memmaps are tens of
# GB per layer and are deleted after each layer by default.

export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64/

python demo.py --model_save_path ./N0/S8/resnet_fwk/cifar6m/HD/200k/seed8ModelN0200k.pth \
               --num_traces 200000 \
               --noise_level 0 \
               --leakage_model HD \
               --exper_name resnet_fwk \
               --dataset_name cifar6m \
               --seed 8
