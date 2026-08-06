#!/usr/bin/env python3
"""Visualize one raw RoboTwin2.0 sample and its first-frame action images.

Run from repo root:
  PYTHONPATH=src python vis/visualize_robotwin_action_images.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from fastwam.datasets.robotwin import RoboTwinActionImageDataset


def _uint8_video(video) -> np.ndarray:
    video = video.detach().cpu().permute(1, 2, 3, 0).numpy()
    return np.clip((video + 1) * 127.5, 0, 255).astype(np.uint8)


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 8 * len(text) + 12), 22), fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255))
    return np.asarray(image)


def _overlay(reference: np.ndarray, action_image: np.ndarray) -> np.ndarray:
    baseline = np.median(action_image.reshape(-1, 3), axis=0)
    signal = np.clip(action_image.astype(np.float32) - baseline, 0, 255)
    alpha = np.clip(signal.max(axis=-1, keepdims=True) / 180, 0, 0.8)
    return (reference * (1 - alpha) + action_image * alpha).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="./data/RoboTwin2.0/dataset")
    parser.add_argument("--output-dir", default="./outputs/robotwin_action_image_vis")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--embodiment", action="append", dest="embodiments")
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--global-sample-stride", type=int, default=1)
    parser.add_argument("--fov-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = RoboTwinActionImageDataset(
        args.dataset_root,
        tasks=args.tasks,
        embodiments=args.embodiments,
        window_stride=args.window_stride,
        global_sample_stride=args.global_sample_stride,
        action_fov_scale=args.fov_scale,
        return_video_components=True,
    )
    sample = dataset[args.index]
    observation = _uint8_video(sample["observation_video"])
    action_image = _uint8_video(sample["action_video"])
    combined = _uint8_video(sample["video"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{args.index:06d}"

    paired = [
        np.concatenate(
            (
                _label(observation[t], f"observation t={t}"),
                _label(
                    _overlay(observation[t], action_image[t]),
                    f"observation ⊕ action image t={t}",
                ),
                _label(action_image[t], f"action image t={t} (camera@window start)"),
            ),
            axis=1,
        )
        for t in range(len(observation))
    ]
    imageio.mimsave(output_dir / f"{stem}_paired.mp4", paired, fps=args.fps, codec="libx264")

    target = [
        _label(frame, f"{'observation' if t < len(observation) else 'action image'} {t % len(observation)}")
        for t, frame in enumerate(combined)
    ]
    imageio.mimsave(output_dir / f"{stem}_target.mp4", target, fps=args.fps, codec="libx264")

    selected = sorted({0, len(observation) // 2, len(observation) - 1})
    contact_sheet = np.concatenate([paired[t] for t in selected], axis=0)
    imageio.imwrite(output_dir / f"{stem}_contact_sheet.png", contact_sheet)

    print(f"dataset samples: {len(dataset):,}")
    print(f"episode: {sample['episode_path']}")
    print(f"window start: {sample['window_start']}")
    print(f"video: {tuple(sample['video'].shape)}  # [C, observation+action-image T, H, W]")
    print(f"action: {tuple(sample['action'].shape)}  # [T, 16]")
    print(f"prompt: {sample['prompt']}")
    print(f"wrote: {output_dir / f'{stem}_paired.mp4'}")
    print(f"wrote: {output_dir / f'{stem}_target.mp4'}")
    print(f"wrote: {output_dir / f'{stem}_contact_sheet.png'}")


if __name__ == "__main__":
    main()
