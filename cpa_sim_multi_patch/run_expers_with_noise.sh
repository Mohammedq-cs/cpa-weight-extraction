#!/bin/bash
# Noisy CNN sweep: chained SLURM jobs for the noise-CNN extraction experiments.
# Mirrors run_zero_noise.sh but at non-zero noise levels: the (noise, leakage, kTraces)
# runs used by graphs_code/noise_ext_comp (cifar6m, seed 8, model_extract). Each job depends on the previous
# (`-j ${prev_job_id}`) so the cluster runs them serially -- the multi-patch
# extractor uses a kTraces-sized memmap per patch, so concurrent big-K jobs
# would compete for RAM/IO.
#
# Usage:
#   ./run_expers_with_noise.sh                 # start fresh (no initial dependency)
#   ./run_expers_with_noise.sh <dep_job_id>    # chain after an existing job (optional dependency)

set -euo pipefail

prev_job_id="${1:-}"
default_seed=8

# Each line: ExperName LEAKAGE_MODEL NOISE_LEVEL kTraces Dataset_name script_name
jobs=(
  "model_extract HD 6  200 cifar6m demo"
  "model_extract HD 8  200 cifar6m demo"
  "model_extract HD 10 200 cifar6m demo"
  "model_extract HD 12 200 cifar6m demo"
  "model_extract HD 14 200 cifar6m demo"
  "model_extract HD 16 200 cifar6m demo"
  "model_extract HD 6  400 cifar6m demo"
  "model_extract HD 8  400 cifar6m demo"
  "model_extract HD 10 400 cifar6m demo"
  "model_extract HD 12 400 cifar6m demo"
  "model_extract HD 14 400 cifar6m demo"
  "model_extract HD 16 400 cifar6m demo"
)

echo "About to chain ${#jobs[@]} jobs"
[[ -n "$prev_job_id" ]] && echo "Initial dependency: $prev_job_id"
echo

for job in "${jobs[@]}"; do
  read -r ExperName LEAKAGE_MODEL NOISE_LEVEL kTraces Dataset_name script_name <<< "$job"
  seed="$default_seed"

  if [[ -n "$prev_job_id" ]]; then
    job_id=$(./submit_experiment.sh \
      -e "$ExperName" -l "$LEAKAGE_MODEL" -n "$NOISE_LEVEL" \
      -k "$kTraces" -d "$Dataset_name" -s "$script_name" \
      -r "$seed" -j "$prev_job_id")
  else
    job_id=$(./submit_experiment.sh \
      -e "$ExperName" -l "$LEAKAGE_MODEL" -n "$NOISE_LEVEL" \
      -k "$kTraces" -d "$Dataset_name" -s "$script_name" \
      -r "$seed")
  fi

  echo "Submitted $job_id  -- $ExperName ($LEAKAGE_MODEL, N=$NOISE_LEVEL, k=${kTraces}k, data=$Dataset_name, script=$script_name)"
  prev_job_id="$job_id"
done

echo
echo "Last job in chain: $prev_job_id"
