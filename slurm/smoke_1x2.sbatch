#!/bin/bash
# 3-step smoke test for a task config on 2 GPUs. Not for real training.
#   sbatch --export=ALL,TASK=robotwin_poker_scene_only_1x4 slurm/smoke_1x2.sbatch
#SBATCH --job-name=fw-smoke
#SBATCH --partition=b200-batch
#SBATCH --account=chuangg_umass
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --chdir=/home/hzhen_umass/code/fastwam-actionimage
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /home/hzhen_umass/code/fastwam-actionimage
mkdir -p logs
: "${TASK:?TASK must be set}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
source ./env_setup.sh
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((21000 + SLURM_JOB_ID % 9000))
export RUN_ID="smoke_${TASK}"
exec bash scripts/train_zero1.sh 2 "task=${TASK}" \
  max_steps=3 save_every=100000 eval_every=2 log_every=1 \
  batch_size=1 gradient_accumulation_steps=1 num_workers=2 \
  "output_dir=./runs/smoke/${TASK}"
