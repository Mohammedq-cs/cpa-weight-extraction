#!/bin/bash
# Usage: ./run_exps.sh
# Define jobs in order: each line has: ExperName LEAKAGE_MODEL NOISE_LEVEL kTraces Dataset_name
jobs=(
  "model_extract HD 0 2 cifar6m demo"
  "model_extract HD 0 3 cifar6m demo"
  "model_extract HD 0 4 cifar6m demo" 
  "model_extract HD 0 5 cifar6m demo"
  "model_extract HD 0 6 cifar6m demo"
  "model_extract HD 0 7 cifar6m demo"
  "model_extract HD 0 10 cifar6m demo"
  "model_extract HD 0 20 cifar6m demo"
  "model_extract HD 0 30 cifar6m demo"
  "model_extract HD 0 40 cifar6m demo"
  "model_extract HD 0 50 cifar6m demo"
  "model_extract HD 0 70 cifar6m demo"
  "model_extract HD 0 150 cifar6m demo"
  "inp_dis_comp HD 0 10 cifar10 inpDist"
  "inp_dis_comp HD 0 10 cifar6m inpDist"
  "inp_dis_comp HD 0 10 random inpDist"
  "inp_dis_comp HD 0 10 c6m_low_acc inpDist"
  "inp_dis_comp HD 0 10 c6m_high_acc inpDist"
  "inp_dis_comp HW 0 10 cifar10 inpDist"
  "inp_dis_comp HW 0 10 cifar6m inpDist"
  "inp_dis_comp HW 0 10 random inpDist"
  "inp_dis_comp HW 0 10 c6m_low_acc inpDist"
  "inp_dis_comp HW 0 10 c6m_high_acc inpDist"
  "model_extract HW 0 2 cifar6m demo"
  "model_extract HW 0 3 cifar6m demo"
  "model_extract HW 0 4 cifar6m demo"
  "model_extract HW 0 5 cifar6m demo"
  "model_extract HW 0 6 cifar6m demo"
  "model_extract HW 0 7 cifar6m demo"
  "model_extract HW 0 10 cifar6m demo"
  "model_extract HW 0 20 cifar6m demo"
  "model_extract HW 0 30 cifar6m demo"
  "model_extract HW 0 40 cifar6m demo"
  "model_extract HW 0 50 cifar6m demo"
  "model_extract HW 0 70 cifar6m demo"
  "model_extract HW 0 150 cifar6m demo"
)


prev_job_id="${1:-}"   # optional: chain after an existing job

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