#!/bin/bash
# Precompute every text embedding that a RoboTwin randomized eval can sample.
#
# Example:
#   sbatch --export=ALL,TASKCFG=robotwin_place_empty_cup_scene_only_1x4,TASK_NAME=place_empty_cup \
#     slurm/precompute_eval_text_embeds.sbatch

#SBATCH --job-name=fw-eval-text
#SBATCH --partition=b200-batch
#SBATCH --account=chuangg_umass
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/hzhen_umass/code/fastwam-actionimage
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo_dir="/home/hzhen_umass/code/fastwam-actionimage"
cd "${repo_dir}"
mkdir -p logs

: "${TASKCFG:?TASKCFG must name the matching RoboTwin task config}"
: "${TASK_NAME:?TASK_NAME must name the RoboTwin environment task}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
source ./env_setup.sh

echo "[eval-text] host=$(hostname) job=${SLURM_JOB_ID} task_cfg=${TASKCFG} task=${TASK_NAME}"
exec python -u scripts/precompute_eval_text_embeds.py \
  --config-name sim_robotwin \
  "task=${TASKCFG}" \
  "EVALUATION.task_name=${TASK_NAME}"
