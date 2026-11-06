#!/bin/bash
# Usage: ./run_expers_with_noise.sh [dep_job_id]   (optional: chain after an existing job)
# Noisy runs used by graphs_code/noise_ext_comp (and the Fixed RF curve of
# ../cpa_sim_multi_patch/graphs_code/noise_ext_comp). Jobs are chained, one after another.
# Each line: ExperName LEAKAGE_MODEL NOISE_LEVEL kTraces Dataset_name script_name
jobs=(
  "model_extract HD 4 200 cifar6m demo"
  "model_extract HD 6 200 cifar6m demo"
  "model_extract HD 8 200 cifar6m demo"
  "model_extract HD 10 200 cifar6m demo"
  "model_extract HD 12 200 cifar6m demo"
  "model_extract HD 14 200 cifar6m demo"
  "model_extract HD 16 200 cifar6m demo"
  "model_extract HW 4 200 cifar6m demo"
  "model_extract HW 6 200 cifar6m demo"
  "model_extract HW 10 200 cifar6m demo"
  "model_extract HW 12 200 cifar6m demo"
  "model_extract HW 14 200 cifar6m demo"
  "model_extract HW 16 200 cifar6m demo"
  "model_extract HD 4 400 cifar6m demo"
  "model_extract HD 6 400 cifar6m demo"
  "model_extract HD 8 400 cifar6m demo"
  "model_extract HD 10 400 cifar6m demo"
  "model_extract HD 12 400 cifar6m demo"
  "model_extract HD 14 400 cifar6m demo"
  "model_extract HD 16 400 cifar6m demo"
  "model_extract HW 4 400 cifar6m demo"
  "model_extract HW 6 400 cifar6m demo"
  "model_extract HW 10 400 cifar6m demo"
  "model_extract HW 12 400 cifar6m demo"
  "model_extract HW 14 400 cifar6m demo"
  "model_extract HW 16 400 cifar6m demo"
  "model_extract HD 4 800 cifar6m demo"
  "model_extract HD 6 800 cifar6m demo"
  "model_extract HD 10 800 cifar6m demo"
  "model_extract HD 12 800 cifar6m demo"
  "model_extract HD 14 800 cifar6m demo"
  "model_extract HD 16 800 cifar6m demo"
  "model_extract HW 8 800 cifar6m demo"
  "model_extract HW 10 800 cifar6m demo"
  "model_extract HW 12 800 cifar6m demo"
  "model_extract HW 14 800 cifar6m demo"
)


prev_job_id="${1:-}"

for job in "${jobs[@]}"; do
  read -r ExperName LEAKAGE_MODEL NOISE_LEVEL kTraces Dataset_name script_name <<< "$job"
  
  if [ -z "$prev_job_id" ]; then
    # first job no dependency
    job_id=$(./submit_experiment.sh "$ExperName" "$LEAKAGE_MODEL" "$NOISE_LEVEL" "$kTraces" "$Dataset_name" "$script_name")
  else
    job_id=$(./submit_experiment.sh "$ExperName" "$LEAKAGE_MODEL" "$NOISE_LEVEL" "$kTraces" "$Dataset_name" "$script_name" "$prev_job_id")
  fi
  
  echo "Submitted job $job_id for $ExperName"
  prev_job_id=$job_id
done