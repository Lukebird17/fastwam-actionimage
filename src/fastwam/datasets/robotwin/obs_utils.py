"""Observation helpers shared by RoboTwin training visualization and deploy."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from .action_image import render_action_images_for_camera
from .raw_dataset import CAMERA_NAMES, _compose_robotwin_views
from .wrist_mounts import resolve_wrist_cam_from_ee


def endpose_dict_to_action_16d(endpose: dict) -> np.ndarray:
    """Pack RoboTwin ``observation['endpose']`` into the training 16D layout."""
    return np.concatenate(
        (
            np.asarray(endpose["left_endpose"], dtype=np.float32),
            np.asarray([endpose["left_gripper"]], dtype=np.float32),
            np.asarray(endpose["right_endpose"], dtype=np.float32),
            np.asarray([endpose["right_gripper"]], dtype=np.float32),
        ),
        axis=0,
    )


def render_action_image_tensor(
    observation: dict,
    action_16d: np.ndarray,
    *,
    action_fov_scale: float = 1.0,
    action_axis_length: float = 0.1,
    action_sigma: float = 0.05,
    wrist_look_distance: float = 0.25,
    wrist_cam_from_ee: Mapping[str, object] | None = None,
    embodiment: str | None = "aloha-agilex",
) -> torch.Tensor:
    """Render the current 16D pose into the training action-image layout.

    Returns ``[1, 3, 384, 320]`` in ``[-1, 1]``.
    Head uses true projection; wrist uses per-side calibrated ``T_cam←ee``.
    """
    action = np.asarray(action_16d, dtype=np.float32).reshape(1, 16)
    mounts = resolve_wrist_cam_from_ee(wrist_cam_from_ee, embodiment=embodiment)
    obs = observation["observation"] if "observation" in observation else observation
    views = []
    for camera_name in CAMERA_NAMES:
        camera = obs[camera_name]
        rgb = np.asarray(camera["rgb"])
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        height, width = rgb.shape[:2]
        action_rgb = render_action_images_for_camera(
            camera_name,
            action,
            np.asarray(camera["extrinsic_cv"], dtype=np.float32),
            np.asarray(camera["intrinsic_cv"], dtype=np.float32),
            height,
            width,
            fov_scale=action_fov_scale,
            axis_length=action_axis_length,
            sigma=action_sigma,
            wrist_look_distance=wrist_look_distance,
            wrist_cam_from_ee=mounts,
        )
        views.append(torch.from_numpy(action_rgb).permute(0, 3, 1, 2))
    composed = _compose_robotwin_views(torch.stack(views)).float()  # [1, C, H, W]
    return composed.mul(2).sub(1)


def compose_robotwin_rgb_tensor(observation: dict) -> torch.Tensor:
    """Compose head/left/right RGB into ``[1, 3, 384, 320]`` in ``[-1, 1]``."""
    obs = observation["observation"] if "observation" in observation else observation
    views = []
    for camera_name in CAMERA_NAMES:
        rgb = np.asarray(obs[camera_name]["rgb"])
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        views.append(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0))
    composed = _compose_robotwin_views(torch.stack(views)).float() / 255.0  # [1, C, H, W]
    return composed.mul(2).sub(1)
