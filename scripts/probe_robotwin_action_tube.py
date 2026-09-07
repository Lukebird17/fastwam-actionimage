#!/usr/bin/env python3
"""Screen whether future action tubes localize future scene change in RoboTwin.

This is a cheap hypothesis probe rather than paper evidence. For each sampled
window, it measures future grayscale change in a shell around the projected
end-effector trajectory. The shell excludes the immediate end-effector core and
is compared with a temporally shuffled trajectory from the same episode and a
current-position-only shell.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np


@dataclass(frozen=True)
class WindowResult:
    task: str
    embodiment: str
    episode: str
    anchor: int
    control_anchor: int
    actual_energy: float
    shuffled_energy: float
    position_energy: float
    global_energy: float
    actual_pixels: int
    shuffled_pixels: int
    position_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/RoboTwin2.0/dataset"))
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=("move_playingcard_away", "place_empty_cup", "stack_blocks_two"),
    )
    parser.add_argument(
        "--embodiments",
        nargs="+",
        default=("aloha-agilex", "arx-x5", "piper"),
    )
    parser.add_argument("--dataset-suffix", default="clean_50")
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--max-windows", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--future-stride", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--inner-radius", type=int, default=7)
    parser.add_argument("--outer-radius", type=int, default=20)
    parser.add_argument("--max-control-iou", type=float, default=0.25)
    parser.add_argument("--min-mask-pixels", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def decode_gray(encoded: object, width: int) -> tuple[np.ndarray, float]:
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to decode RoboTwin JPEG frame")
    scale = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0, scale


def project_positions(
    xyz_world: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    camera = np.einsum("tij,tj->ti", extrinsics[:, :, :3], xyz_world) + extrinsics[:, :, 3]
    homogeneous = np.einsum("tij,tj->ti", intrinsics, camera)
    depth = homogeneous[:, 2]
    safe_depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    pixels = homogeneous[:, :2] / safe_depth[:, None]
    return pixels.astype(np.float32) * scale, depth > 1e-6


def trajectory_shell(
    paths: list[np.ndarray],
    valid_paths: list[np.ndarray],
    height: int,
    width: int,
    inner_radius: int,
    outer_radius: int,
) -> np.ndarray:
    outer = np.zeros((height, width), dtype=np.uint8)
    inner = np.zeros_like(outer)
    for points, valid in zip(paths, valid_paths):
        valid_points = points[valid]
        if len(valid_points) < 2:
            continue
        rounded = np.rint(valid_points).astype(np.int32)
        cv2.polylines(
            outer,
            [rounded],
            isClosed=False,
            color=1,
            thickness=2 * outer_radius + 1,
            lineType=cv2.LINE_8,
        )
        cv2.polylines(
            inner,
            [rounded],
            isClosed=False,
            color=1,
            thickness=2 * inner_radius + 1,
            lineType=cv2.LINE_8,
        )
    return (outer.astype(bool) & ~inner.astype(bool))


def position_shell(
    centers: list[np.ndarray],
    valid: list[bool],
    height: int,
    width: int,
    inner_radius: int,
    outer_radius: int,
) -> np.ndarray:
    outer = np.zeros((height, width), dtype=np.uint8)
    inner = np.zeros_like(outer)
    for center, is_valid in zip(centers, valid):
        if not is_valid:
            continue
        xy = tuple(np.rint(center).astype(int))
        cv2.circle(outer, xy, outer_radius, 1, thickness=-1, lineType=cv2.LINE_8)
        cv2.circle(inner, xy, inner_radius, 1, thickness=-1, lineType=cv2.LINE_8)
    return outer.astype(bool) & ~inner.astype(bool)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.logical_or(first, second).sum())
    return float(np.logical_and(first, second).sum() / max(union, 1))


def energy(change: np.ndarray, mask: np.ndarray) -> float:
    return float(change[mask].mean())


def episode_results(
    path: Path,
    task: str,
    embodiment: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[WindowResult]:
    with h5py.File(path, "r") as episode:
        camera = episode[f"observation/{args.camera}"]
        encoded = camera["rgb"]
        frames = []
        scale = None
        for item in encoded:
            frame, current_scale = decode_gray(item, args.image_width)
            if scale is None:
                scale = current_scale
            elif not np.isclose(scale, current_scale):
                raise ValueError(f"Frame scale changed within {path}")
            frames.append(frame)
        images = np.stack(frames)
        extrinsics = camera["extrinsic_cv"][:].astype(np.float32)
        intrinsics = camera["intrinsic_cv"][:].astype(np.float32)
        arm_pixels = {}
        for arm in ("left", "right"):
            pose = episode[f"endpose/{arm}_endpose"][:].astype(np.float32)
            arm_pixels[arm] = project_positions(pose[:, :3], extrinsics, intrinsics, float(scale))

    height, width = images.shape[1:]
    last_anchor = len(images) - args.horizon - 1
    if last_anchor < 0:
        return []
    control_anchors = np.arange(0, last_anchor + 1, args.window_stride)
    anchors = control_anchors
    if len(anchors) > args.max_windows:
        anchors = np.sort(rng.choice(anchors, size=args.max_windows, replace=False))

    results = []
    future_offsets = np.arange(0, args.horizon + 1, args.future_stride)
    for anchor_value in anchors:
        anchor = int(anchor_value)
        indices = anchor + future_offsets
        actual = trajectory_shell(
            [arm_pixels[arm][0][indices] for arm in arm_pixels],
            [arm_pixels[arm][1][indices] for arm in arm_pixels],
            height,
            width,
            args.inner_radius,
            args.outer_radius,
        )
        if int(actual.sum()) < args.min_mask_pixels:
            continue

        candidate_controls = control_anchors[np.abs(control_anchors - anchor) >= args.horizon]
        if not len(candidate_controls):
            continue
        candidate_controls = rng.permutation(candidate_controls)
        control_anchor = None
        shuffled = None
        for candidate_value in candidate_controls:
            candidate = int(candidate_value)
            control_indices = candidate + future_offsets
            candidate_mask = trajectory_shell(
                [arm_pixels[arm][0][control_indices] for arm in arm_pixels],
                [arm_pixels[arm][1][control_indices] for arm in arm_pixels],
                height,
                width,
                args.inner_radius,
                args.outer_radius,
            )
            if int(candidate_mask.sum()) < args.min_mask_pixels:
                continue
            if mask_iou(actual, candidate_mask) <= args.max_control_iou:
                control_anchor = candidate
                shuffled = candidate_mask
                break
        if shuffled is None or control_anchor is None:
            continue

        position = position_shell(
            [arm_pixels[arm][0][anchor] for arm in arm_pixels],
            [bool(arm_pixels[arm][1][anchor]) for arm in arm_pixels],
            height,
            width,
            args.inner_radius,
            args.outer_radius,
        )
        if int(position.sum()) < args.min_mask_pixels:
            continue

        future = images[indices[1:]]
        change = np.max(np.abs(future - images[anchor][None, ...]), axis=0)
        results.append(
            WindowResult(
                task=task,
                embodiment=embodiment,
                episode=path.stem,
                anchor=anchor,
                control_anchor=control_anchor,
                actual_energy=energy(change, actual),
                shuffled_energy=energy(change, shuffled),
                position_energy=energy(change, position),
                global_energy=float(change.mean()),
                actual_pixels=int(actual.sum()),
                shuffled_pixels=int(shuffled.sum()),
                position_pixels=int(position.sum()),
            )
        )
    return results


def aggregate(rows: list[WindowResult]) -> dict[str, float | int]:
    actual = np.asarray([row.actual_energy for row in rows])
    shuffled = np.asarray([row.shuffled_energy for row in rows])
    position = np.asarray([row.position_energy for row in rows])
    global_energy = np.asarray([row.global_energy for row in rows])
    epsilon = 1e-8
    return {
        "windows": len(rows),
        "episodes": len({f"{row.task}/{row.embodiment}/{row.episode}" for row in rows}),
        "pooled_actual_energy": float(actual.mean()),
        "pooled_shuffled_energy": float(shuffled.mean()),
        "pooled_position_energy": float(position.mean()),
        "pooled_global_energy": float(global_energy.mean()),
        "actual_over_shuffled": float(actual.sum() / max(shuffled.sum(), epsilon)),
        "actual_over_position": float(actual.sum() / max(position.sum(), epsilon)),
        "actual_over_global": float(actual.sum() / max(global_energy.sum(), epsilon)),
        "fraction_actual_beats_shuffled": float(np.mean(actual > shuffled)),
        "fraction_actual_beats_position": float(np.mean(actual > position)),
    }


def summarize(rows: list[WindowResult], elapsed_seconds: float, args: argparse.Namespace) -> dict[str, object]:
    groups = {}
    for task in sorted({row.task for row in rows}):
        groups[f"task/{task}"] = aggregate([row for row in rows if row.task == task])
    for embodiment in sorted({row.embodiment for row in rows}):
        groups[f"embodiment/{embodiment}"] = aggregate(
            [row for row in rows if row.embodiment == embodiment]
        )
    return {
        "warning": (
            "Hypothesis screen only. The inner shell removes the immediate end-effector path, "
            "not all visible robot pixels, and the metric does not establish causality."
        ),
        "overall": aggregate(rows),
        "grouped": groups,
        "elapsed_seconds": elapsed_seconds,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "results": [asdict(row) for row in rows],
    }


def write_summary(metrics: dict[str, object], output_path: Path) -> None:
    overall = metrics["overall"]
    lines = [
        "# Action-Tube Scene-Change Probe",
        "",
        "Future image change is measured in a shell around the projected future end-effector",
        "path. The shell excludes the immediate path but cannot remove every visible robot pixel.",
        "This is a hypothesis screen, not causal or policy evidence.",
        "",
        f"- Episodes: {overall['episodes']}",
        f"- Windows: {overall['windows']}",
        f"- Actual / temporally shuffled tube energy: {overall['actual_over_shuffled']}",
        f"- Actual tube / current-position shell energy: {overall['actual_over_position']}",
        f"- Actual tube / global energy: {overall['actual_over_global']}",
        f"- Fraction actual > shuffled: {overall['fraction_actual_beats_shuffled']}",
        f"- Fraction actual > position: {overall['fraction_actual_beats_position']}",
        "",
        "Go criterion: actual/shuffled >= 2.0 and actual/position >= 1.25 overall,",
        "with actual/shuffled >= 1.5 for every task. Passing only motivates a loss-weighting run.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if not 0 <= args.inner_radius < args.outer_radius:
        raise ValueError("Require 0 <= inner radius < outer radius")
    if args.horizon < args.future_stride or args.window_stride < 1:
        raise ValueError("Invalid horizon or stride")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(0)
    rng = np.random.default_rng(args.seed)
    started = time.time()
    rows = []
    for task in args.tasks:
        for embodiment in args.embodiments:
            data_dir = args.dataset_root / task / f"{embodiment}_{args.dataset_suffix}" / "data"
            if not data_dir.is_dir():
                print(f"[skip] missing {data_dir}", flush=True)
                continue
            paths = sorted(data_dir.glob("episode*.hdf5"))[: args.max_episodes]
            for index, path in enumerate(paths, start=1):
                try:
                    episode_rows = episode_results(path, task, embodiment, args, rng)
                except Exception as error:
                    print(f"[error] {path}: {error!r}", flush=True)
                    continue
                rows.extend(episode_rows)
                print(
                    f"[done] {task}/{embodiment} {index}/{len(paths)} "
                    f"windows={len(episode_rows)} total={len(rows)}",
                    flush=True,
                )
    if not rows:
        raise RuntimeError("No valid action-tube windows were found")
    metrics = summarize(rows, time.time() - started, args)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_summary(metrics, args.output_dir / "summary.md")
    print((args.output_dir / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
