#!/usr/bin/env bash
#SBATCH --job-name=action-tube-probe
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/action-tube-probe-%j.out
#SBATCH --error=logs/action-tube-probe-%j.err

set -euo pipefail

repo_dir="/home/hzhen_umass/code/fastwam-actionimage"
python_bin="/home/hzhen_umass/.conda/envs/curobo/bin/python"
run_id="${SLURM_JOB_ID:-local}"

cd "${repo_dir}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

"${python_bin}" scripts/probe_robotwin_action_tube.py \
  --dataset-root data/RoboTwin2.0/dataset \
  --tasks move_playingcard_away place_empty_cup stack_blocks_two \
  --embodiments aloha-agilex arx-x5 piper \
  --max-episodes 20 \
  --output-dir "outputs/action_tube_probe/${run_id}"
