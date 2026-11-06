#!/bin/bash
# Verify the zero-noise sweep finished cleanly, then auto-launch the
# random-model experiments. Intended to run as a tiny SLURM job that depends
# on the sweep's last job (sbatch --dependency=afterany:<that job id>).
#
# Pass criteria: every expected .pth from the zero-noise sweep exists.
# Fail behaviour: print a summary of missing configs and exit 1 WITHOUT
# launching anything -- so the random-model jobs only run when the prior
# data is complete.

set -u

cd "$(dirname "$0")"

missing=()

for L in HD HW; do
  for K in 2 3 4 5 6 7 10 20 30 40 50 70; do
    p="N0/S8/model_extract/cifar6m/${L}/${K}k/seed8ModelN0${K}k.pth"
    [ -f "$p" ] || missing+=("$p")
  done
done

for L in HD HW; do
  for D in cifar10 cifar6m random c6m_low_acc c6m_high_acc; do
    p="N0/S8/inp_dis_comp/${D}/${L}/10k/seed8ModelN010k.pth"
    [ -f "$p" ] || missing+=("$p")
  done
done

if [ "${#missing[@]}" -gt 0 ]; then
  echo "[verify] FAIL: ${#missing[@]} expected .pth files missing:"
  printf '  %s\n' "${missing[@]}"
  echo "[verify] Not launching random-model experiments."
  exit 1
fi

echo "[verify] OK: all 34 zero-noise .pth files present."
echo "[verify] Launching random-model experiments via run_expers_rnd_models.sh"
./run_expers_rnd_models.sh
