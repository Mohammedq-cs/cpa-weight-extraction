#!/bin/bash
# Feed the sigma sweep in waves, respecting the cluster's per-user submitted-job cap.
#
# Each sigma point is a 32-task array and there are 16 of them (8 sigmas x 2 modes),
# so they cannot all be queued at once. This waits for slots and submits the next
# point as capacity frees. Filters are independent, so wave order does not affect
# results -- only wall-clock.
#
# Called by run_all.sh; can also be run standalone:
#     SRC=<code dir> RUN=<run dir> ./submit_waves.sh
#
# NOTE: sigma=0 is submitted separately, before this, as the correctness gate.
# Experiment A is submitted by run_all.sh (it depends on the random baseline, not
# on this sweep), so it is deliberately not launched here.
set -uo pipefail

SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$PWD}"
cd "$RUN"

# Keep the submitted-task count below CAP before adding another 32-task array so a
# submission is never rejected for exceeding the cap. Lower if your cluster's limit
# is tighter: sacctmgr show assoc user=$USER format=MaxSubmitJobs
CAP="${WAVE_CAP:-32}"
LOG="$RUN/wave_submissions.log"

wait_for_slot () {
  while [ "$(squeue -u "${SLURM_USER:-$USER}" -h -r 2>/dev/null | wc -l)" -ge "$CAP" ]; do
    sleep 180
  done
}

for n in 100 200 300 400 500 600 700 850; do
  for mode in seq BS; do
    wait_for_slot
    # Retry on rejection: a refused sbatch must NOT silently skip a sigma point --
    # that leaves a hole in the sweep which only surfaces at scoring time.
    for attempt in $(seq 1 10); do
      jid=$(sbatch --parsable -J "${mode}_N${n}" \
            --output="./scriptsOut/${mode}_N${n}_%A_%a.out" \
            --error="./scriptsErr/${mode}_N${n}_%A_%a.err" \
            --export=ALL,NOISE=${n} "$SRC/slurm/extract_l1_${mode}.sbatch" 2>&1)
      case "$jid" in
        ''|*[!0-9]*)
          echo "$(date +%H:%M:%S) sigma=${n} ${mode} REJECTED (attempt $attempt): $jid" | tee -a "$LOG"
          sleep 180; wait_for_slot ;;
        *)
          echo "$(date +%H:%M:%S) sigma=${n} ${mode} -> ${jid}" | tee -a "$LOG"; break ;;
      esac
    done
  done
done

echo "$(date +%H:%M:%S) ALL SIGMA POINTS SUBMITTED" | tee -a "$LOG"
