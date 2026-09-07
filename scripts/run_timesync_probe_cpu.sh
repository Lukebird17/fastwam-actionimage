#!/usr/bin/env bash
#SBATCH --job-name=timesync-probe
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/timesync-probe-%j.out
#SBATCH --error=logs/timesync-probe-%j.err

set -euo pipefail

repo_dir="/home/hzhen_umass/code/fastwam-actionimage"
python_bin="/home/hzhen_umass/.conda/envs/curobo/bin/python"
run_id="${SLURM_JOB_ID:-local}"

cd "${repo_dir}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

"${python_bin}" scripts/probe_robotwin_timesync.py \
  --dataset-root data/RoboTwin2.0/dataset \
  --tasks handover_block pick_dual_bottles place_empty_cup stack_blocks_two \
  --embodiments aloha-agilex arx-x5 piper \
  --max-episodes 20 \
  --output-dir "outputs/timesync_probe/${run_id}"
