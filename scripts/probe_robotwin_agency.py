#!/usr/bin/env python3
"""Screen for body-world motion coupling in RoboTwin demonstrations.

This is a hypothesis probe, not a paper benchmark. Stable closed-gripper phases
in successful demonstrations are used as a weak proxy for object ownership. The
probe asks whether scene flow around an end effector is better explained by the
true end-effector motion than by RGB motion and end-effector speed alone.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASELINE_FEATURES = ("ee_speed", "flow_energy", "moving_fraction")
COUPLING_FEATURES = (
    "weighted_cosine",
    "aligned_fraction",
    "velocity_match",
)


@dataclass(frozen=True)
class FeatureRow:
    task: str
    embodiment: str
    episode: str
    arm: str
    frame: int
    label: int
    features: tuple[float, ...]
    shuffled_features: tuple[float, ...]

    @property
    def group(self) -> str:
        return f"{self.task}/{self.embodiment}/{self.episode}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/RoboTwin2.0/dataset"))
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=("handover_block", "pick_dual_bottles", "place_empty_cup", "stack_blocks_two"),
    )
    parser.add_argument(
        "--embodiments",
        nargs="+",
        default=("aloha-agilex", "arx-x5", "piper"),
    )
    parser.add_argument("--dataset-suffix", default="clean_50")
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--flow-width", type=int, default=160)
    parser.add_argument("--inner-radius", type=float, default=10.0)
    parser.add_argument("--outer-radius", type=float, default=52.0)
    parser.add_argument("--min-ee-speed", type=float, default=0.5)
    parser.add_argument("--min-flow", type=float, default=0.2)
    parser.add_argument("--closed-threshold", type=float, default=0.2)
    parser.add_argument("--transition-margin", type=int, default=6)
    parser.add_argument("--shuffle-shift", type=int, default=11)
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
    return resized, scale


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


def stable_binary_labels(values: np.ndarray, threshold: float, margin: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(values) < threshold
    stable = np.ones_like(labels, dtype=bool)
    transitions = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    for transition in transitions:
        start = max(0, transition - margin)
        end = min(len(labels), transition + margin + 1)
        stable[start:end] = False
    return labels.astype(np.int64), stable


def annulus_mask(
    height: int,
    width: int,
    center_xy: np.ndarray,
    inner_radius: float,
    outer_radius: float,
) -> np.ndarray:
    x0, y0 = float(center_xy[0]), float(center_xy[1])
    x_min = max(0, int(math.floor(x0 - outer_radius)))
    x_max = min(width, int(math.ceil(x0 + outer_radius + 1)))
    y_min = max(0, int(math.floor(y0 - outer_radius)))
    y_max = min(height, int(math.ceil(y0 + outer_radius + 1)))
    mask = np.zeros((height, width), dtype=bool)
    if x_min >= x_max or y_min >= y_max:
        return mask
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    radius_sq = (xx - x0) ** 2 + (yy - y0) ** 2
    local = (radius_sq >= inner_radius**2) & (radius_sq <= outer_radius**2)
    mask[y_min:y_max, x_min:x_max] = local
    return mask


def flow_features(
    flow: np.ndarray,
    center_xy: np.ndarray,
    ee_delta: np.ndarray,
    *,
    inner_radius: float,
    outer_radius: float,
    min_flow: float,
) -> tuple[float, ...] | None:
    height, width = flow.shape[:2]
    mask = annulus_mask(height, width, center_xy, inner_radius, outer_radius)
    vectors = flow[mask]
    if len(vectors) < 32:
        return None

    magnitudes = np.linalg.norm(vectors, axis=1)
    moving = magnitudes >= min_flow
    if not np.any(moving):
        moving_vectors = vectors
        moving_magnitudes = magnitudes
    else:
        moving_vectors = vectors[moving]
        moving_magnitudes = magnitudes[moving]

    speed = float(np.linalg.norm(ee_delta))
    direction = ee_delta / max(speed, 1e-6)
    cosine = (moving_vectors @ direction) / np.maximum(moving_magnitudes, 1e-6)
    weight_sum = max(float(moving_magnitudes.sum()), 1e-6)
    weighted_cosine = float((cosine * moving_magnitudes).sum() / weight_sum)
    aligned_fraction = float(np.mean(cosine > 0.7))
    residual = np.linalg.norm(moving_vectors - ee_delta[None, :], axis=1)
    velocity_match = float(np.mean(np.exp(-residual / max(speed, 1.0))))

    return (
        speed,
        float(np.mean(magnitudes)),
        float(np.mean(moving)),
        weighted_cosine,
        aligned_fraction,
        velocity_match,
    )


def episode_rows(
    path: Path,
    task: str,
    embodiment: str,
    args: argparse.Namespace,
) -> list[FeatureRow]:
    rows: list[FeatureRow] = []
    with h5py.File(path, "r") as episode:
        camera = episode[f"observation/{args.camera}"]
        encoded = camera["rgb"]
        first_gray, scale = decode_gray(encoded[0], args.flow_width)
        extrinsics = camera["extrinsic_cv"][:].astype(np.float32)
        intrinsics = camera["intrinsic_cv"][:].astype(np.float32)

        arm_data: dict[str, dict[str, np.ndarray]] = {}
        for arm in ("left", "right"):
            pose = episode[f"endpose/{arm}_endpose"][:].astype(np.float32)
            pixels, valid = project_positions(pose[:, :3], extrinsics, intrinsics, scale)
            labels, stable = stable_binary_labels(
                episode[f"endpose/{arm}_gripper"][:],
                args.closed_threshold,
                args.transition_margin,
            )
            arm_data[arm] = {"pixels": pixels, "valid": valid, "labels": labels, "stable": stable}

        previous_index = 0
        previous_gray = first_gray
        for frame in range(args.frame_stride, len(encoded), args.frame_stride):
            current_gray, current_scale = decode_gray(encoded[frame], args.flow_width)
            if not np.isclose(scale, current_scale):
                raise ValueError(f"Frame scale changed within {path}")
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                current_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            for arm, data in arm_data.items():
                if not data["stable"][frame] or not data["valid"][previous_index] or not data["valid"][frame]:
                    continue
                pixels = data["pixels"]
                true_delta = pixels[frame] - pixels[previous_index]
                speed = float(np.linalg.norm(true_delta))
                if speed < args.min_ee_speed:
                    continue

                shuffled_start = (previous_index + args.shuffle_shift) % len(pixels)
                shuffled_end = (frame + args.shuffle_shift) % len(pixels)
                shuffled_delta = pixels[shuffled_end] - pixels[shuffled_start]
                shuffled_norm = float(np.linalg.norm(shuffled_delta))
                if shuffled_norm < 1e-6:
                    shuffled_delta = -true_delta
                else:
                    shuffled_delta = shuffled_delta * (speed / shuffled_norm)

                features = flow_features(
                    flow,
                    pixels[previous_index],
                    true_delta,
                    inner_radius=args.inner_radius,
                    outer_radius=args.outer_radius,
                    min_flow=args.min_flow,
                )
                shuffled_features = flow_features(
                    flow,
                    pixels[previous_index],
                    shuffled_delta,
                    inner_radius=args.inner_radius,
                    outer_radius=args.outer_radius,
                    min_flow=args.min_flow,
                )
                if features is None or shuffled_features is None:
                    continue
                rows.append(
                    FeatureRow(
                        task=task,
                        embodiment=embodiment,
                        episode=path.stem,
                        arm=arm,
                        frame=frame,
                        label=int(data["labels"][frame]),
                        features=features,
                        shuffled_features=shuffled_features,
                    )
                )

            previous_index = frame
            previous_gray = current_gray
    return rows


def fit_predict_auc(
    train_rows: list[FeatureRow],
    test_rows: list[FeatureRow],
    feature_indices: tuple[int, ...],
    *,
    shuffled: bool = False,
    seed: int,
) -> float | None:
    y_train = np.asarray([row.label for row in train_rows])
    y_test = np.asarray([row.label for row in test_rows])
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None
    source = "shuffled_features" if shuffled else "features"
    x_train = np.asarray(
        [[getattr(row, source)[index] for index in feature_indices] for row in train_rows],
        dtype=np.float32,
    )
    x_test = np.asarray(
        [[getattr(row, source)[index] for index in feature_indices] for row in test_rows],
        dtype=np.float32,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, scores))


def leave_one_group_out(rows: list[FeatureRow], seed: int) -> dict[str, object]:
    all_indices = tuple(range(len(BASELINE_FEATURES) + len(COUPLING_FEATURES)))
    baseline_indices = tuple(range(len(BASELINE_FEATURES)))
    tasks = sorted({row.task for row in rows})
    per_task: dict[str, object] = {}
    pooled: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "baseline": [],
        "coupling": [],
        "time_shuffled": [],
    }

    for held_out in tasks:
        train_rows = [row for row in rows if row.task != held_out]
        test_rows = [row for row in rows if row.task == held_out]
        result = {
            "samples": len(test_rows),
            "positives": int(sum(row.label for row in test_rows)),
            "episodes": len({row.group for row in test_rows}),
            "baseline_auc": fit_predict_auc(
                train_rows, test_rows, baseline_indices, seed=seed
            ),
            "coupling_auc": fit_predict_auc(
                train_rows, test_rows, all_indices, seed=seed
            ),
            "time_shuffled_auc": fit_predict_auc(
                train_rows, test_rows, all_indices, shuffled=True, seed=seed
            ),
        }
        per_task[held_out] = result

    valid_results = [value for value in per_task.values() if value["coupling_auc"] is not None]
    macro = {}
    for name in ("baseline_auc", "coupling_auc", "time_shuffled_auc"):
        values = [float(value[name]) for value in valid_results if value[name] is not None]
        macro[name] = float(np.mean(values)) if values else None
    if macro.get("coupling_auc") is not None and macro.get("baseline_auc") is not None:
        macro["coupling_minus_baseline"] = macro["coupling_auc"] - macro["baseline_auc"]
    if macro.get("coupling_auc") is not None and macro.get("time_shuffled_auc") is not None:
        macro["coupling_minus_time_shuffled"] = macro["coupling_auc"] - macro["time_shuffled_auc"]
    return {"per_task": per_task, "macro": macro}


def save_rows(rows: list[FeatureRow], output_path: Path) -> None:
    np.savez_compressed(
        output_path,
        task=np.asarray([row.task for row in rows]),
        embodiment=np.asarray([row.embodiment for row in rows]),
        episode=np.asarray([row.episode for row in rows]),
        arm=np.asarray([row.arm for row in rows]),
        frame=np.asarray([row.frame for row in rows]),
        label=np.asarray([row.label for row in rows]),
        features=np.asarray([row.features for row in rows], dtype=np.float32),
        shuffled_features=np.asarray([row.shuffled_features for row in rows], dtype=np.float32),
        feature_names=np.asarray(BASELINE_FEATURES + COUPLING_FEATURES),
    )


def write_summary(metrics: dict[str, object], output_path: Path) -> None:
    macro = metrics["leave_one_task_out"]["macro"]
    lines = [
        "# Agency Probe Summary",
        "",
        "This is a weak-label screening experiment. Stable closed-gripper phases in successful",
        "demonstrations proxy object ownership; they are not contact or ownership ground truth.",
        "",
        f"- Samples: {metrics['samples']}",
        f"- Episodes: {metrics['episodes']}",
        f"- Positive proxy samples: {metrics['positives']}",
        f"- LOTO baseline AUC: {macro.get('baseline_auc')}",
        f"- LOTO body-world coupling AUC: {macro.get('coupling_auc')}",
        f"- LOTO time-shuffled AUC: {macro.get('time_shuffled_auc')}",
        f"- Coupling gain over baseline: {macro.get('coupling_minus_baseline')}",
        f"- Coupling gain over time-shuffled: {macro.get('coupling_minus_time_shuffled')}",
        "",
        "Go criterion: coupling gain >= 0.10 over baseline and >= 0.10 over time-shuffled.",
        "Passing only motivates an oracle-labeled simulator probe; it does not validate the paper claim.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(0)
    start_time = time.time()
    rows: list[FeatureRow] = []
    processed: list[str] = []

    for task in args.tasks:
        for embodiment in args.embodiments:
            data_dir = args.dataset_root / task / f"{embodiment}_{args.dataset_suffix}" / "data"
            if not data_dir.is_dir():
                print(f"[skip] missing {data_dir}", flush=True)
                continue
            paths = sorted(data_dir.glob("*.hdf5"))[: args.max_episodes]
            for index, path in enumerate(paths, start=1):
                try:
                    new_rows = episode_rows(path, task, embodiment, args)
                except Exception as error:
                    print(f"[error] {path}: {error!r}", flush=True)
                    continue
                rows.extend(new_rows)
                processed.append(str(path))
                print(
                    f"[done] {task}/{embodiment} {index}/{len(paths)} "
                    f"rows={len(new_rows)} total={len(rows)}",
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No valid feature rows were produced")
    save_rows(rows, args.output_dir / "features.npz")
    metrics: dict[str, object] = {
        "warning": (
            "Weak-label screening only: closed-gripper phases in successful demonstrations "
            "are not contact or ownership ground truth."
        ),
        "samples": len(rows),
        "positives": int(sum(row.label for row in rows)),
        "episodes": len({row.group for row in rows}),
        "tasks": sorted({row.task for row in rows}),
        "embodiments": sorted({row.embodiment for row in rows}),
        "processed_files": processed,
        "feature_names": list(BASELINE_FEATURES + COUPLING_FEATURES),
        "leave_one_task_out": leave_one_group_out(rows, args.seed),
        "elapsed_seconds": time.time() - start_time,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_summary(metrics, args.output_dir / "summary.md")
    print((args.output_dir / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
