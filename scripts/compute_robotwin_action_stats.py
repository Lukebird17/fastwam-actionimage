#!/usr/bin/env python3
"""Compute 16D Aloha end-effector normalization statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from fastwam.datasets.robotwin.raw_dataset import ALOHA_EMBODIMENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="./data/RoboTwin2.0/dataset")
    parser.add_argument("--output", default="./data/RoboTwin2.0/aloha_action_16d_stats.json")
    parser.add_argument("--task", action="append", dest="tasks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    task_filter = set(args.tasks or [])
    paths = [
        path
        for path in sorted(root.glob("*/*/data/episode*.hdf5"))
        if path.parents[1].name in ALOHA_EMBODIMENTS
        and (not task_filter or path.parents[2].name in task_filter)
    ]
    if not paths:
        raise FileNotFoundError(f"No matching Aloha episodes under {root}")

    count = 0
    total = np.zeros(16, dtype=np.float64)
    total_squared = np.zeros(16, dtype=np.float64)
    for path in tqdm(paths, desc="Computing 16D action stats"):
        with h5py.File(path, "r") as episode:
            endpose = episode["endpose"]
            action = np.concatenate(
                (
                    endpose["left_endpose"][:],
                    endpose["left_gripper"][:][:, None],
                    endpose["right_endpose"][:],
                    endpose["right_gripper"][:][:, None],
                ),
                axis=-1,
            ).astype(np.float64)
        count += action.shape[0]
        total += action.sum(axis=0)
        total_squared += np.square(action).sum(axis=0)

    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 1e-12)
    payload = {
        "count": count,
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "layout": "left_xyz_quat_wxyz_gripper__right_xyz_quat_wxyz_gripper",
        "embodiments": list(ALOHA_EMBODIMENTS),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump(payload, file, indent=2)
    print(f"Wrote {count:,} action rows from {len(paths):,} episodes to {output}")


if __name__ == "__main__":
    main()
