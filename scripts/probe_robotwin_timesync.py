#!/usr/bin/env python3
"""Probe whether rendered actions can recover video-proprio timestamp offsets.

The probe injects a constant timestamp offset into each RoboTwin episode and
estimates its correction from the agreement between projected end-effector
motion and local image optical flow. It is a data-quality screen, not a policy
result: success only shows that automatic temporal calibration is observable.
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


@dataclass(frozen=True)
class EpisodeResult:
    task: str
    embodiment: str
    episode: str
    injected_offset: int
    target_correction: int
    predicted_correction: int
    baseline_correction: int
    confidence_margin: float
    valid_pairs: int
    scores: tuple[float, ...]
    baseline_scores: tuple[float, ...]

    @property
    def error(self) -> int:
        return self.predicted_correction - self.target_correction

    @property
    def baseline_error(self) -> int:
        return self.baseline_correction - self.target_correction


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
    parser.add_argument("--radius", type=float, default=30.0)
    parser.add_argument("--min-ee-speed", type=float, default=0.4)
    parser.add_argument("--min-flow", type=float, default=0.15)
    parser.add_argument("--max-offset", type=int, default=8)
    parser.add_argument("--min-injected-offset", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def decode_gray(encoded: object, width: int) -> tuple[np.ndarray, float]:
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to decode RoboTwin JPEG frame")
    scale = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), scale


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


def disk_vectors(flow: np.ndarray, center_xy: np.ndarray, radius: float) -> np.ndarray:
    height, width = flow.shape[:2]
    x0, y0 = float(center_xy[0]), float(center_xy[1])
    x_min = max(0, int(math.floor(x0 - radius)))
    x_max = min(width, int(math.ceil(x0 + radius + 1)))
    y_min = max(0, int(math.floor(y0 - radius)))
    y_max = min(height, int(math.ceil(y0 + radius + 1)))
    if x_min >= x_max or y_min >= y_max:
        return np.empty((0, 2), dtype=np.float32)
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius**2
    return flow[y_min:y_max, x_min:x_max][mask]


def agreement_features(
    flow: np.ndarray,
    center_xy: np.ndarray,
    ee_delta: np.ndarray,
    *,
    radius: float,
    min_flow: float,
) -> tuple[float, float, float, float] | None:
    vectors = disk_vectors(flow, center_xy, radius)
    if len(vectors) < 32:
        return None
    magnitudes = np.linalg.norm(vectors, axis=1)
    moving = magnitudes >= min_flow
    if int(moving.sum()) < 8:
        return None
    vectors = vectors[moving]
    magnitudes = magnitudes[moving]

    speed = float(np.linalg.norm(ee_delta))
    if speed < 1e-6:
        return None
    direction = ee_delta / speed
    cosine = (vectors @ direction) / np.maximum(magnitudes, 1e-6)
    weighted_cosine = float(np.average(cosine, weights=magnitudes))
    aligned_fraction = float(np.mean(cosine > 0.7))
    residual = np.linalg.norm(vectors - ee_delta[None, :], axis=1)
    velocity_match = float(np.mean(np.exp(-residual / max(speed, 1.0))))
    return weighted_cosine, aligned_fraction, velocity_match, min(speed, 8.0)


def score_correction(
    flows: list[tuple[int, int, np.ndarray]],
    arm_pixels: dict[str, tuple[np.ndarray, np.ndarray]],
    injected_offset: int,
    correction: int,
    args: argparse.Namespace,
) -> tuple[float, int]:
    observations: list[tuple[float, float, float, float]] = []
    action_shift = injected_offset + correction
    for previous, frame, flow in flows:
        action_previous = previous + action_shift
        action_frame = frame + action_shift
        for pixels, valid in arm_pixels.values():
            if action_previous < 0 or action_frame >= len(pixels):
                continue
            if not valid[action_previous] or not valid[action_frame]:
                continue
            ee_delta = pixels[action_frame] - pixels[action_previous]
            if float(np.linalg.norm(ee_delta)) < args.min_ee_speed:
                continue
            features = agreement_features(
                flow,
                pixels[action_previous],
                ee_delta,
                radius=args.radius,
                min_flow=args.min_flow,
            )
            if features is not None:
                observations.append(features)

    if len(observations) < 12:
        return float("-inf"), len(observations)
    values = np.asarray(observations, dtype=np.float32)
    weights = values[:, 3]
    directional = np.average(values[:, 0], weights=weights)
    aligned = np.average(values[:, 1], weights=weights)
    velocity = np.average(values[:, 2], weights=weights)
    return float(directional + 0.25 * aligned + 0.25 * velocity), len(observations)


def score_speed_baseline(
    flows: list[tuple[int, int, np.ndarray]],
    arm_pixels: dict[str, tuple[np.ndarray, np.ndarray]],
    injected_offset: int,
    correction: int,
) -> float:
    flow_energy: list[float] = []
    action_speed: list[float] = []
    action_shift = injected_offset + correction
    for previous, frame, flow in flows:
        action_previous = previous + action_shift
        action_frame = frame + action_shift
        speeds = []
        for pixels, valid in arm_pixels.values():
            if action_previous < 0 or action_frame >= len(pixels):
                continue
            if not valid[action_previous] or not valid[action_frame]:
                continue
            speeds.append(float(np.linalg.norm(pixels[action_frame] - pixels[action_previous])))
        if not speeds:
            continue
        flow_energy.append(float(np.mean(np.linalg.norm(flow, axis=2))))
        action_speed.append(float(np.sum(speeds)))
    if len(action_speed) < 12 or np.std(action_speed) < 1e-6 or np.std(flow_energy) < 1e-6:
        return float("-inf")
    return float(np.corrcoef(action_speed, flow_energy)[0, 1])


def choose_injected_offset(rng: np.random.Generator, args: argparse.Namespace) -> int:
    choices = np.asarray(
        [
            value
            for value in range(-args.max_offset, args.max_offset + 1)
            if abs(value) >= args.min_injected_offset
        ]
    )
    if not len(choices):
        raise ValueError("No injected offsets remain after applying the configured limits")
    return int(rng.choice(choices))


def evaluate_episode(
    path: Path,
    task: str,
    embodiment: str,
    injected_offset: int,
    args: argparse.Namespace,
) -> EpisodeResult | None:
    with h5py.File(path, "r") as episode:
        camera = episode[f"observation/{args.camera}"]
        encoded = camera["rgb"]
        first_gray, scale = decode_gray(encoded[0], args.flow_width)
        extrinsics = camera["extrinsic_cv"][:].astype(np.float32)
        intrinsics = camera["intrinsic_cv"][:].astype(np.float32)

        arm_pixels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for arm in ("left", "right"):
            pose = episode[f"endpose/{arm}_endpose"][:].astype(np.float32)
            arm_pixels[arm] = project_positions(pose[:, :3], extrinsics, intrinsics, scale)

        flows: list[tuple[int, int, np.ndarray]] = []
        previous = 0
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
            flows.append((previous, frame, flow))
            previous = frame
            previous_gray = current_gray

    candidates = tuple(range(-args.max_offset, args.max_offset + 1))
    scored = [
        score_correction(flows, arm_pixels, injected_offset, correction, args)
        for correction in candidates
    ]
    baseline_scored = [
        score_speed_baseline(flows, arm_pixels, injected_offset, correction)
        for correction in candidates
    ]
    finite = [(score, correction, count) for correction, (score, count) in zip(candidates, scored) if np.isfinite(score)]
    if len(finite) < 2:
        return None
    ranked = sorted(finite, reverse=True)
    best_score, predicted, valid_pairs = ranked[0]
    baseline_finite = [
        (score, correction)
        for correction, score in zip(candidates, baseline_scored)
        if np.isfinite(score)
    ]
    if not baseline_finite:
        return None
    baseline_predicted = max(baseline_finite)[1]
    margin = best_score - ranked[1][0]
    return EpisodeResult(
        task=task,
        embodiment=embodiment,
        episode=path.stem,
        injected_offset=injected_offset,
        target_correction=-injected_offset,
        predicted_correction=predicted,
        baseline_correction=baseline_predicted,
        confidence_margin=float(margin),
        valid_pairs=valid_pairs,
        scores=tuple(float(score) for score, _ in scored),
        baseline_scores=tuple(float(score) for score in baseline_scored),
    )


def summarize(results: list[EpisodeResult], elapsed_seconds: float, args: argparse.Namespace) -> dict[str, object]:
    errors = np.asarray([result.error for result in results])

    def metrics(subset: list[EpisodeResult], *, baseline: bool = False) -> dict[str, object]:
        subset_errors = np.asarray(
            [result.baseline_error if baseline else result.error for result in subset]
        )
        return {
            "episodes": len(subset),
            "exact_accuracy": float(np.mean(subset_errors == 0)),
            "within_one_accuracy": float(np.mean(np.abs(subset_errors) <= 1)),
            "within_two_accuracy": float(np.mean(np.abs(subset_errors) <= 2)),
            "median_absolute_error": float(np.median(np.abs(subset_errors))),
        }

    grouped: dict[str, object] = {}
    for task in sorted({result.task for result in results}):
        grouped[f"task/{task}"] = metrics([result for result in results if result.task == task])
    for embodiment in sorted({result.embodiment for result in results}):
        grouped[f"embodiment/{embodiment}"] = metrics(
            [result for result in results if result.embodiment == embodiment]
        )

    return {
        "warning": (
            "Synthetic constant-offset observability probe only. It does not establish that real datasets "
            "are misaligned or that correcting timestamps improves policy learning."
        ),
        "overall": metrics(results),
        "speed_baseline": metrics(results, baseline=True),
        "grouped": grouped,
        "random_exact_accuracy": 1.0 / (2 * args.max_offset + 1),
        "mean_signed_error": float(np.mean(errors)),
        "elapsed_seconds": elapsed_seconds,
        "correction_candidates": list(range(-args.max_offset, args.max_offset + 1)),
        "results": [
            {
                "task": result.task,
                "embodiment": result.embodiment,
                "episode": result.episode,
                "injected_offset": result.injected_offset,
                "target_correction": result.target_correction,
                "predicted_correction": result.predicted_correction,
                "error": result.error,
                "baseline_correction": result.baseline_correction,
                "baseline_error": result.baseline_error,
                "confidence_margin": result.confidence_margin,
                "valid_pairs": result.valid_pairs,
                "scores": list(result.scores),
                "baseline_scores": list(result.baseline_scores),
            }
            for result in results
        ],
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def write_summary(metrics: dict[str, object], output_path: Path) -> None:
    overall = metrics["overall"]
    baseline = metrics["speed_baseline"]
    lines = [
        "# Video-Action Time-Sync Probe",
        "",
        "Synthetic constant offsets were injected per episode and recovered by matching projected",
        "end-effector motion to local optical flow. This tests observability, not policy benefit.",
        "",
        f"- Episodes: {overall['episodes']}",
        f"- Exact correction accuracy: {overall['exact_accuracy']}",
        f"- Within +/-1 frame: {overall['within_one_accuracy']}",
        f"- Within +/-2 frames: {overall['within_two_accuracy']}",
        f"- Median absolute error: {overall['median_absolute_error']}",
        f"- Speed-only baseline exact: {baseline['exact_accuracy']}",
        f"- Speed-only baseline within +/-1: {baseline['within_one_accuracy']}",
        f"- Random exact accuracy: {metrics['random_exact_accuracy']}",
        "",
        "Go criterion: >= 0.70 within +/-1 frame across all episodes and >= 0.55 in every embodiment.",
        "Passing motivates a timestamp-corruption policy experiment; it is not itself an ICLR claim.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1 or args.max_offset < 1:
        raise ValueError("frame stride and maximum offset must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(0)
    rng = np.random.default_rng(args.seed)
    start_time = time.time()
    results: list[EpisodeResult] = []

    for task in args.tasks:
        for embodiment in args.embodiments:
            data_dir = args.dataset_root / task / f"{embodiment}_{args.dataset_suffix}" / "data"
            if not data_dir.is_dir():
                print(f"[skip] missing {data_dir}", flush=True)
                continue
            paths = sorted(data_dir.glob("*.hdf5"))[: args.max_episodes]
            for index, path in enumerate(paths, start=1):
                injected_offset = choose_injected_offset(rng, args)
                try:
                    result = evaluate_episode(path, task, embodiment, injected_offset, args)
                except Exception as error:
                    print(f"[error] {path}: {error!r}", flush=True)
                    continue
                if result is None:
                    print(f"[skip] insufficient motion {path}", flush=True)
                    continue
                results.append(result)
                print(
                    f"[done] {task}/{embodiment} {index}/{len(paths)} "
                    f"target={result.target_correction:+d} predicted={result.predicted_correction:+d} "
                    f"error={result.error:+d} baseline_error={result.baseline_error:+d}",
                    flush=True,
                )

    if not results:
        raise RuntimeError("No episodes produced valid time-sync estimates")
    metrics = summarize(results, time.time() - start_time, args)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_summary(metrics, args.output_dir / "summary.md")
    print((args.output_dir / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
