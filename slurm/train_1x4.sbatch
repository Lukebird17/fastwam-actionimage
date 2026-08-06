#!/bin/bash
# FastWAM single-node training: 1 node × 4 GPUs (AICR Slurm + accelerate).
#
# Same launcher as train_2x8.sbatch, resized for the 4-GPU poker experiments.
# TASK has no default here on purpose: the two arms of that comparison must be
# named explicitly so neither can be submitted by accident.
#
# From repo root:
#   sbatch --export=ALL,TASK=robotwin_poker_scene_only_1x4   slurm/train_1x4.sbatch -- wandb.enabled=true
#   sbatch --export=ALL,TASK=robotwin_poker_action_image_1x4 slurm/train_1x4.sbatch -- wandb.enabled=true
#
# Optional env:
#   TASK          hydra task=... (default: robotwin_action_image_3cam_384_1e-4)
#   TRAIN_SCRIPT  scripts/train_zero1.sh | scripts/train_zero2.sh

#SBATCH --job-name=fastwam-1x4
#SBATCH --partition=b200-batch
#SBATCH --account=chuangg_umass
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/hzhen_umass/code/fastwam-actionimage
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/hzhen_umass/code/fastwam-actionimage}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

if [[ -z "${TASK:-}" ]]; then
  echo "[fatal] TASK must be set explicitly, e.g. --export=ALL,TASK=robotwin_poker_scene_only_1x4" >&2
  exit 2
fi
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_zero1.sh}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"

# Rendezvous: avoid loopback MASTER_ADDR (hangs multi-node).
_MASTER_HOST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
_MASTER_ADDR="$(scontrol show node "${_MASTER_HOST}" -o 2>/dev/null | sed -n 's/.* NodeAddr=\([^ ]*\).*/\1/p')"
if [[ -z "${_MASTER_ADDR}" || "${_MASTER_ADDR}" == "127.0.0.1" || "${_MASTER_ADDR}" == "::1" ]]; then
  _MASTER_ADDR="$(getent ahostsv4 "${_MASTER_HOST}" 2>/dev/null | awk '$1 !~ /^(127\.|0\.)/ {print $1; exit}')"
fi
if [[ -z "${_MASTER_ADDR}" || "${_MASTER_ADDR}" == "127.0.0.1" || "${_MASTER_ADDR}" == "::1" ]]; then
  _MASTER_ADDR="${_MASTER_HOST}"
fi
export MASTER_ADDR="${_MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
export NNODES="${SLURM_NNODES}"

# NCCL: TCP over bond; do not hardcode eth0.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker,virbr,ibs}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"

echo "[sbatch] host=$(hostname) job=${SLURM_JOB_ID}"
echo "[sbatch] nodes=${SLURM_NNODES} nodelist=${SLURM_JOB_NODELIST}"
echo "[sbatch] gpus_per_node=${GPUS_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT} (host=${_MASTER_HOST})"
printf -v _EXTRA_ARGS '%q ' "$@"
echo "[sbatch] task=${TASK} train_script=${TRAIN_SCRIPT}"
echo "[sbatch] extra_args: $*"

if [[ "${SLURM_NNODES}" -gt 1 && ( "${MASTER_ADDR}" == "::1" || "${MASTER_ADDR}" == "127.0.0.1" ) ]]; then
  echo "[fatal] MASTER_ADDR=${MASTER_ADDR} is loopback; multi-node rendezvous cannot work." >&2
  exit 2
fi

# One accelerate launcher per node. Re-source env on each rank (sbatch is non-interactive).
srun --export=ALL --cpu-bind=none --kill-on-bad-exit=1 \
  bash -lc "
    set -euo pipefail
    cd '${REPO_ROOT}'
    if [[ -f /etc/profile.d/modules.sh ]]; then
      source /etc/profile.d/modules.sh
    fi
    module load conda/latest
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    source '${REPO_ROOT}/env_setup.sh'
    export NNODES='${SLURM_NNODES}'
    export NODE_RANK=\"\${SLURM_NODEID}\"
    export MASTER_ADDR='${MASTER_ADDR}'
    export MASTER_PORT='${MASTER_PORT}'
    echo \"[rank] host=\$(hostname) node_rank=\${NODE_RANK} python=\$(command -v python)\"
    exec bash '${REPO_ROOT}/${TRAIN_SCRIPT}' '${GPUS_PER_NODE}' 'task=${TASK}' ${_EXTRA_ARGS}
  "
