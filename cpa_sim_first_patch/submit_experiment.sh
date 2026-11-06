#!/bin/bash
# Usage: ./submit_experiment.sh ExperName LEAKAGE_MODEL NOISE_LEVEL kTraces Dataset_name [dependency_job_id]
# Example: ./submit_experiment.sh myExp HD 24 800 cifar1m 12345


ExperName=$1
LEAKAGE_MODEL=$2
NOISE_LEVEL=$3
kTraces=$4
Dataset_name=$5
ScriptName=${6:-demo}   # If 6th argument not provided, default to demo 
dependency_job_id=${7:-0}   # If 6th argument not provided, default to 0
# Derive any other needed variables
NUMTRACES=${kTraces}000

# Path for the model save
Out_path="./N${NOISE_LEVEL}/${ExperName}/${Dataset_name}/${LEAKAGE_MODEL}/${kTraces}k"
MODEL_PATH="${Out_path}/seed8ModelN${NOISE_LEVEL}${kTraces}k.pth"
mkdir -p "${Out_path}"   # Slurm writes exper.out/err here

# Create the sbatch script dynamically
TMP_SCRIPT="temp_${NOISE_LEVEL}_${kTraces}_${LEAKAGE_MODEL}.sbatch"

cat > ${TMP_SCRIPT} <<EOT
#! /bin/bash
#SBATCH --job-name=N${NOISE_LEVEL}${kTraces}k
#SBATCH --output=${Out_path}/exper.out
#SBATCH --error=${Out_path}/exper.err
#SBATCH --time=1400
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60000
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
EOT

# Add dependency line only if dependency_job_id is non-zero
if [ "${dependency_job_id}" != "0" ]; then
  echo "#SBATCH --dependency=afterany:${dependency_job_id}" >> ${TMP_SCRIPT}
fi

cat >> ${TMP_SCRIPT} <<EOT

export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64/

python ${ScriptName}.py --model_save_path ${MODEL_PATH} \
               --num_traces ${NUMTRACES} \
               --noise_level ${NOISE_LEVEL} \
               --leakage_model ${LEAKAGE_MODEL} \
               --exper_name ${ExperName} \
               --dataset_name ${Dataset_name}
EOT

# Submit the newly created job script
sbatch --parsable ${TMP_SCRIPT}
