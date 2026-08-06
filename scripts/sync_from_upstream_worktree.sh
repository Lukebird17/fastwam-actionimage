#!/usr/bin/env bash
# Sync the senior student's working tree (~/code/FastWAM, uncommitted) into this fork.
#
# The fork was branched by `cp -a` at a point in time, so it silently goes stale
# every time the shared tree advances. That is not cosmetic: the snapshot taken on
# 2026-08-06 predated the `ee_from_action_frame` calibration, so this fork rendered
# action images with the WRONG gripper axes (green/blue channels swapped onto the
# wrong EE directions) while claiming to test the same method.
#
# Only files that are genuinely shared are copied. Everything this fork owns --
# the poker experiment configs, the 1x4/smoke launchers, the relative
# third_party symlink, and the sbatch --chdir rewrites -- is deliberately NOT in
# the list and survives a sync.
set -euo pipefail

UPSTREAM="${UPSTREAM:-/home/hzhen_umass/code/FastWAM}"
FORK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "$UPSTREAM" ]]; then
  echo "[fatal] upstream worktree not found: $UPSTREAM" >&2
  exit 2
fi
if [[ "$UPSTREAM" == "$FORK" ]]; then
  echo "[fatal] upstream and fork are the same tree" >&2
  exit 2
fi

# Shared implementation. Keep this list explicit: a blanket rsync would clobber
# the fork-owned files above and re-point the sbatch --chdir at the old tree.
FILES=(
  src/fastwam/datasets/robotwin/__init__.py
  src/fastwam/datasets/robotwin/action_image.py
  src/fastwam/datasets/robotwin/conditioning.py
  src/fastwam/datasets/robotwin/obs_utils.py
  src/fastwam/datasets/robotwin/raw_dataset.py
  src/fastwam/datasets/robotwin/wrist_mounts.py
  src/fastwam/models/wan22/fastwam.py
  src/fastwam/models/wan22/fastwam_idm.py
  src/fastwam/models/wan22/fastwam_joint.py
  src/fastwam/runtime.py
  src/fastwam/trainer.py
  src/fastwam/utils/precision.py
  configs/data/robotwin_action_image.yaml
  configs/model/fastwam_action_image.yaml
  configs/sim_robotwin.yaml
  configs/task/robotwin_action_image_3cam_384_1e-4.yaml
  experiments/libero/eval_libero_single.py
  experiments/robotwin/eval_robotwin_single.py
  experiments/robotwin/fastwam_policy/deploy_policy.py
  experiments/robotwin/fastwam_policy/deploy_policy.yml
  scripts/compute_robotwin_action_stats.py
  scripts/precompute_text_embeds.py
  vis/visualize_robotwin_action_3d.py
  vis/visualize_robotwin_action_episode.py
)

changed=0
for rel in "${FILES[@]}"; do
  src="$UPSTREAM/$rel"
  dst="$FORK/$rel"
  if [[ ! -e "$src" ]]; then
    echo "  skip (absent upstream): $rel"
    continue
  fi
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" ]] && cmp -s "$src" "$dst"; then
    continue
  fi
  cp -a "$src" "$dst"
  echo "  synced: $rel"
  changed=$((changed + 1))
done

echo "[sync] $changed file(s) updated from $UPSTREAM"
echo "[sync] now run: python scripts/check_poker_configs.py"
