#!/bin/bash
# Closed-loop RoboTwin success-rate eval for one poker checkpoint.
#
#   sbatch --export=ALL,CKPT=/path/step_002500.pt,TASKCFG=robotwin_poker_scene_only_1x4 \
#     slurm/eval_robotwin_poker.sbatch
#
# The policy auto-detects which arm the checkpoint is by reading
# `video_attention_mask_mode` off the instantiated model, so the SAME launcher
# evaluates the control and the treatment. TASKCFG must still name the matching
# task config, because that is what supplies the data config the policy reads its
# render geometry and text-embedding cache from.
#
# One GPU: RoboTwin's simulator is single-process, so the arms are run as two
# independent jobs rather than one multi-GPU job.
#SBATCH --job-name=fw-eval
#SBATCH --partition=b200-batch
#SBATCH --account=chuangg_umass
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/hzhen_umass/code/fastwam-actionimage
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/hzhen_umass/code/fastwam-actionimage
mkdir -p logs

: "${CKPT:?CKPT must be set to a .pt weights file}"
: "${TASKCFG:?TASKCFG must be set, e.g. robotwin_poker_scene_only_1x4}"
TASK_NAME="${TASK_NAME:-move_playingcard_away}"
NUM_EPISODES="${NUM_EPISODES:-50}"
TASK_CONFIG="${TASK_CONFIG:-demo_randomized}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[fatal] checkpoint not found: ${CKPT}" >&2
  exit 2
fi

source /etc/profile.d/modules.sh 2>/dev/null || true
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
source ./env_setup.sh

# RoboTwin renders with the ray-tracing shader by default. The rasterizer is the
# shader this project standardized on, because the RT path produced a wedge
# artifact in rendered observations. Set it for eval too so the closed-loop
# observations match what the policy was trained on.
export ROBOTWIN_CAMERA_SHADER="${ROBOTWIN_CAMERA_SHADER:-default}"

# Vulkan/EGL offscreen rendering needs an explicit device on this cluster.
export SAPIEN_HEADLESS="${SAPIEN_HEADLESS:-1}"

echo "[eval] host=$(hostname) job=${SLURM_JOB_ID}"
echo "[eval] ckpt=${CKPT}"
echo "[eval] task_cfg=${TASKCFG} task=${TASK_NAME} episodes=${NUM_EPISODES} config=${TASK_CONFIG}"
echo "[eval] shader=${ROBOTWIN_CAMERA_SHADER}"

exec python -u experiments/robotwin/eval_robotwin_single.py \
  --config-name sim_robotwin \
  "task=${TASKCFG}" \
  "ckpt=${CKPT}" \
  "EVALUATION.task_name=${TASK_NAME}" \
  "EVALUATION.task_config=${TASK_CONFIG}" \
  "EVALUATION.eval_num_episodes=${NUM_EPISODES}" \
  gpu_id=0
