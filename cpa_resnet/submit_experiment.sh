#!/bin/bash
# Usage:
# ./submit_experiment.sh -e ExperName -l LEAKAGE_MODEL -n NOISE_LEVEL -k kTraces -d Dataset_name [-s ScriptName] [-r seed] [-j dependency_job_id]
#
# Example without dependency:
# ./submit_experiment.sh -e myExp -l HD -n 24 -k 800 -d cifar1m
#
# Example with dependency only:
# ./submit_experiment.sh -e myExp -l HD -n 24 -k 800 -d cifar1m -j 12345
#
# Example with custom seed + dependency:
# ./submit_experiment.sh -e myExp -l HD -n 24 -k 800 -d cifar1m -r 42 -j 12345

set -euo pipefail

# Defaults
ScriptName="demo"
seed=8
dependency_job_id=0

# Parse flags
while getopts ":e:l:n:k:d:s:r:j:" opt; do
  case "${opt}" in
    e) ExperName="${OPTARG}" ;;
    l) LEAKAGE_MODEL="${OPTARG}" ;;
    n) NOISE_LEVEL="${OPTARG}" ;;
    k) kTraces="${OPTARG}" ;;
    d) Dataset_name="${OPTARG}" ;;
    s) ScriptName="${OPTARG}" ;;
    r) seed="${OPTARG}" ;;
    j) dependency_job_id="${OPTARG}" ;;
    \?)
      echo "Invalid option: -${OPTARG}" >&2
      exit 1
      ;;
    :)
      echo "Option -${OPTARG} requires an argument." >&2
      exit 1
      ;;
  esac
done

# Validate required params
: "${ExperName:?Missing -e ExperName}"
: "${LEAKAGE_MODEL:?Missing -l LEAKAGE_MODEL}"
: "${NOISE_LEVEL:?Missing -n NOISE_LEVEL}"
: "${kTraces:?Missing -k kTraces}"
: "${Dataset_name:?Missing -d Dataset_name}"

NUMTRACES="${kTraces}000"

Out_path="./N${NOISE_LEVEL}/S${seed}/${ExperName}/${Dataset_name}/${LEAKAGE_MODEL}/${kTraces}k"
# Use chosen seed in model path
MODEL_PATH="${Out_path}/seed${seed}ModelN${NOISE_LEVEL}${kTraces}k.pth"

mkdir -p "${Out_path}"

TMP_SCRIPT="temp_${NOISE_LEVEL}_${kTraces}_${LEAKAGE_MODEL}.sbatch"

cat > "${TMP_SCRIPT}" <<EOT
#!/bin/bash
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
# Set --nodelist / --partition / --constraint to match your cluster, e.g.:
# #SBATCH --nodelist=<your-nodes>
# #SBATCH --partition=<your-partition>
# #SBATCH --constraint=geforce_rtx_3090
EOT

if [ "${dependency_job_id}" != "0" ]; then
  echo "#SBATCH --dependency=afterany:${dependency_job_id}" >> "${TMP_SCRIPT}"
fi

cat >> "${TMP_SCRIPT}" <<EOT

export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64/

python ${ScriptName}.py --model_save_path ${MODEL_PATH} \\
               --num_traces ${NUMTRACES} \\
               --noise_level ${NOISE_LEVEL} \\
               --leakage_model ${LEAKAGE_MODEL} \\
               --exper_name ${ExperName} \\
               --dataset_name ${Dataset_name} \\
               --seed ${seed}
EOT

sbatch --parsable "${TMP_SCRIPT}"