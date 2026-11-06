#!/bin/bash
# Zero-noise sweep: 36 chained SLURM jobs covering the zero-noise experiments
# (trace-count sweep and input-distribution comparison). Each job depends on the previous (`-j ${prev_job_id}`)
# so the cluster runs them serially -- the multi-patch extractor uses a
# kTraces-sized memmap per patch, so concurrent big-K jobs would compete for
# RAM/IO.
#
# Usage:
#   ./run_zero_noise.sh                 # start fresh (no initial dependency)
#   ./run_zero_noise.sh <dep_job_id>    # chain after an existing job (optional dependency)

set -euo pipefail

prev_job_id="${1:-}"
default_seed=8

# 13 kTraces points x 2 leakage models = 26 model_extract jobs (cifar6m).
# Trace points: matches run_exps.sh + a 70k/150k tail.
ktraces=(2 3 4 5 6 7 10 20 30 40 50 70 150)
model_extract_jobs=()
for L in HD HW; do
  for K in "${ktraces[@]}"; do
    model_extract_jobs+=("model_extract $L 0 $K cifar6m demo")
  done
done

# 5 datasets x 2 leakage models = 10 inp_dis_comp jobs at 10k.
datasets=(cifar10 cifar6m random c6m_low_acc c6m_high_acc)
inp_dist_jobs=()
for L in HD HW; do
  for D in "${datasets[@]}"; do
    inp_dist_jobs+=("inp_dis_comp $L 0 10 $D inpDist")
  done
done

jobs=("${model_extract_jobs[@]}" "${inp_dist_jobs[@]}")

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
