#!/usr/bin/env python3
"""Render a complete RoboTwin episode as RGB/action-image paired video."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from fastwam.datasets.robotwin.action_image import render_action_images_for_camera
from fastwam.datasets.robotwin.obs_utils import (
    CAMERA_NAMES,
    ActionImageRenderConfig,
    compose_robotwin_views,
    decode_jpeg,
    load_action_16d_from_hdf5,
)
from fastwam.datasets.robotwin.wrist_mounts import resolve_action_image_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--wrist-look-distance", type=float, default=0.25)
    return parser.parse_args()


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 8 * len(text) + 12), 22), fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255))
    return np.asarray(image)


def _overlay(reference: np.ndarray, action_image: np.ndarray) -> np.ndarray:
    signal = action_image.astype(np.float32)
    alpha = np.clip(signal.max(axis=-1, keepdims=True) / 180.0, 0.0, 0.8)
    return (reference * (1.0 - alpha) + action_image * alpha).astype(np.uint8)


def main() -> None:
    args = parse_args()
    episode = Path(args.episode)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    geometry = resolve_action_image_geometry(episode.parents[1].name)
    render_cfg = ActionImageRenderConfig(wrist_look_distance=args.wrist_look_distance)

    with h5py.File(episode, "r") as file:
        episode_length = int(file["endpose/left_endpose"].shape[0])
        indices = np.arange(args.start, episode_length, args.stride, dtype=np.int64)
        if len(indices) == 0:
            raise ValueError(
                f"No frames: start={args.start}, length={episode_length}, stride={args.stride}"
            )
        action = load_action_16d_from_hdf5(file, indices)
        reference_action = action[:1]

        camera_parameters = {}
        for camera_name in CAMERA_NAMES:
            camera = file["observation"][camera_name]
            camera_parameters[camera_name] = (
                np.asarray(camera["extrinsic_cv"][indices[0]], dtype=np.float32),
                np.asarray(camera["intrinsic_cv"][indices[0]], dtype=np.float32),
            )

        with imageio.get_writer(output, fps=args.fps, codec="libx264") as writer:
            for chunk_start in range(0, len(indices), args.chunk_size):
                chunk_end = min(chunk_start + args.chunk_size, len(indices))
                chunk_indices = indices[chunk_start:chunk_end]
                chunk_action = action[chunk_start:chunk_end]
                # Prefix every chunk with the episode-start action so wrist
                # relative motion stays anchored at t0 across chunk boundaries.
                render_action = np.concatenate((reference_action, chunk_action), axis=0)

                observation_views = []
                action_views = []
                for camera_name in CAMERA_NAMES:
                    camera = file["observation"][camera_name]
                    rgb = np.stack([decode_jpeg(camera["rgb"][i]) for i in chunk_indices])
                    height, width = rgb.shape[1:3]
                    extrinsic, intrinsic = camera_parameters[camera_name]
                    action_rgb = render_action_images_for_camera(
                        camera_name,
                        render_action,
                        extrinsic,
                        intrinsic,
                        height,
                        width,
                        wrist_cam_from_ee=geometry.wrist_cam_from_ee,
                        ee_from_action_frame=geometry.ee_from_action_frame,
                        wrist_look_distance=render_cfg.wrist_look_distance,
                        fov_scale=render_cfg.fov_scale,
                        axis_length=render_cfg.axis_length,
                        sigma=render_cfg.sigma,
                    )[1:]
                    observation_views.append(torch.from_numpy(rgb).permute(0, 3, 1, 2))
                    action_views.append(
                        torch.from_numpy((np.clip(action_rgb, 0, 1) * 255).astype(np.uint8)).permute(
                            0, 3, 1, 2
                        )
                    )

                observation = (
                    compose_robotwin_views(torch.stack(observation_views))
                    .permute(0, 2, 3, 1)
                    .numpy()
                )
                action_image = (
                    compose_robotwin_views(torch.stack(action_views))
                    .permute(0, 2, 3, 1)
                    .numpy()
                )

                for local_index, raw_index in enumerate(chunk_indices):
                    frame = np.concatenate(
                        (
                            _label(observation[local_index], f"observation raw_t={raw_index}"),
                            _label(
                                _overlay(observation[local_index], action_image[local_index]),
                                f"observation + action raw_t={raw_index}",
                            ),
                            _label(
                                action_image[local_index],
                                f"action trajectory from raw_t={indices[0]}",
                            ),
                        ),
                        axis=1,
                    )
                    writer.append_data(frame)
                print(f"rendered {chunk_end}/{len(indices)} frames", flush=True)

    print(f"episode: {episode}")
    print(f"raw frames: {episode_length}; rendered: {len(indices)}; stride: {args.stride}")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
