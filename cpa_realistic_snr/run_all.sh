#!/bin/bash
# ---------------------------------------------------------------------------
# One-command reproduction of Section 8.1 (realistic SNR).
#
#     ./run_all.sh
#
# Submits every stage at once, wired together with SLURM dependencies, and
# returns immediately -- nothing needs to stay attached except the sigma-sweep
# feeder (see NOTE below). Re-running is safe: finished filters are simply
# recomputed, and each stage writes into its own directory.
#
# Configure via environment (all optional):
#     RUN=<dir>             where outputs go                 (default ./run)
#     IMAGENET_ROOT=<dir>   ImageFolder-layout ImageNet      (default ./imagenet)
#     PY=<python>           interpreter                      (default python)
#     SLURM_PARTITION, GPU_CONSTRAINT, BAD_NODES              (cluster specifics)
#     SKIP_CIFAR=1          skip the CIFAR CNN experiment
#
# NOTE ON THE SWEEP: clusters cap how many jobs one user may have queued, so the
# 16 sigma-point arrays (512 tasks) cannot all be submitted at once. run_all.sh
# therefore launches submit_waves.sh in the background to feed them as slots free.
# Run this script under `nohup` or inside tmux/screen if your session may end:
#     nohup ./run_all.sh > run_all.log 2>&1 &
# ---------------------------------------------------------------------------
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${RUN:-$PWD/run}"
PY="${PY:-python}"
export SRC RUN PY

mkdir -p "$RUN"/{scriptsOut,scriptsErr} "$RUN"/cifar/{scriptsOut,scriptsErr}
cd "$RUN"

say () { echo "$(date +%H:%M:%S) [run_all] $*"; }

say "SRC=$SRC"
say "RUN=$RUN"
say "IMAGENET_ROOT=${IMAGENET_ROOT:-./imagenet}"

# --- Stage 1: baselines (independent, run in parallel) ----------------------
IMG=$(sbatch --parsable "$SRC/slurm/gen_imagenet_mp.sbatch")
say "1/6 ImageNet baseline      -> $IMG   (~2 h, ~63 GB)"

RND=$(sbatch --parsable "$SRC/slurm/gen_random.sbatch")
say "1/6 random-input baseline  -> $RND   (~7 GB)"

if [ "${SKIP_CIFAR:-0}" != "1" ]; then
  CIF=$(cd "$RUN/cifar" && RUN="$RUN/cifar" sbatch --parsable "$SRC/slurm/gen_conv1_mp.sbatch")
  say "1/6 CIFAR-5M baseline      -> $CIF   (~2.5 GB)"
fi

# --- Stage 2: sigma=0 correctness gate --------------------------------------
# With a correct baseline, noiseless extraction is EXACT (864/864 in both modes).
# If this is not 1.0000, the regenerated activations are wrong and nothing
# downstream means anything.
G_SEQ=$(sbatch --parsable --dependency=afterok:"$IMG" -J seq_N0 \
        --export=ALL,NOISE=0 "$SRC/slurm/extract_l1_seq.sbatch")
G_BS=$(sbatch --parsable --dependency=afterok:"$IMG" -J BS_N0 \
        --export=ALL,NOISE=0 "$SRC/slurm/extract_l1_BS.sbatch")
say "2/6 sigma=0 gate           -> $G_SEQ (seq) + $G_BS (oracle)"

# --- Stage 3: sigma sweep, fed in waves under the submit-job cap ------------
( "$SRC/slurm/submit_waves.sh" >> "$RUN/waves.log" 2>&1 ) &
say "3/6 sigma sweep 100..850   -> feeder started (see $RUN/waves.log)"

# --- Stage 4: chained layer 2 (needs the sigma=850 sequential layer-1 result) -
# submit_waves.sh submits sigma=850 last, so gate the sim-input build on the
# whole sweep having been submitted and drained; simplest robust form is to let
# the feeder finish, then chain. Done inline in the background subshell below.
(
  wait                                    # feeder has submitted everything
  while squeue -u "${SLURM_USER:-$USER}" -h -n seq_N850 -r 2>/dev/null | grep -q .; do
    sleep 120
  done
  SIM=$(sbatch --parsable "$SRC/slurm/gen_sim_l2.sbatch")
  L2=$(sbatch --parsable --dependency=afterok:"$SIM" "$SRC/slurm/extract_l2.sbatch")
  echo "$(date +%H:%M:%S) [run_all] 4/6 layer-2 chain -> $SIM (sim inputs) -> $L2 (extract)"
) &

# --- Stage 5: Experiment A + CIFAR extraction -------------------------------
EXPA=$(sbatch --parsable --dependency=afterok:"$RND" "$SRC/slurm/extract_random_L1L2_BS.sbatch")
say "5/6 Experiment A (random)  -> $EXPA"

if [ "${SKIP_CIFAR:-0}" != "1" ]; then
  CIFX=$(cd "$RUN/cifar" && RUN="$RUN/cifar" sbatch --parsable \
         --dependency=afterok:"$CIF" "$SRC/slurm/extract_conv1_mp.sbatch")
  say "5/6 CIFAR conv1 extraction -> $CIFX"
fi

# --- Stage 6: how to score --------------------------------------------------
cat <<EOF

$(date +%H:%M:%S) [run_all] 6/6 all stages submitted.

When the queue drains, score and plot:

    $PY $SRC/tools/compare_to_paper.py --repro "$RUN"
    $PY $SRC/tools/make_plot_data.py --root "$RUN"
    $PY $SRC/tools/make_plots.py --outdir "$RUN/plots"

compare_to_paper.py refuses to score any sigma point with fewer than 32 filters
(reported as INCOMPLETE) -- a partial average looks plausible and is wrong.

Expected: sigma=0 exactly 1.0000 in both modes; sigma=850 sequential 0.3391;
layer 2 0.0174; random-model layer 1 0.8576; CIFAR conv1 0.3831. The random-model
layer-2 value is 0.7639, NOT the 0.8056 printed in the paper -- see README.md.
EOF
