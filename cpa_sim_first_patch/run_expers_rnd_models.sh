#!/bin/bash
# Usage: ./submit_multiple.sh
# Define jobs in order: each line has: ExperName LEAKAGE_MODEL NOISE_LEVEL kTraces Dataset_name

jobs=(
  "rnd_model HD 0 10 randInit model_choice"
  "rnd_model HD 0 10 randUniform model_choice"
)


prev_job_id=""

for job in "${jobs[@]}"; do
  read -r ExperName LEAKAGE_MODEL NOISE_LEVEL kTraces model script_name <<< "$job"
  
  if [ -z "$prev_job_id" ]; then
    # first job no dependency
    job_id=$(./submit_experiment_rnd_model.sh "$ExperName" "$LEAKAGE_MODEL" "$NOISE_LEVEL" "$kTraces" "$model" "$script_name")
  else
    job_id=$(./submit_experiment_rnd_model.sh "$ExperName" "$LEAKAGE_MODEL" "$NOISE_LEVEL" "$kTraces" "$model" "$script_name" "$prev_job_id")
  fi
  
  echo "Submitted job $job_id for $ExperName"
  prev_job_id=$job_id
done