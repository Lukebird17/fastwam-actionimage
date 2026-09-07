"""Shared RoboTwin observation packing and action-image rendering.

Training, closed-loop deploy, trainer eval, and visualization all go through
these helpers so layout / geometry / dtype conversions stay identical.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as transforms_f
from PIL import Image

from .action_image import render_action_images_for_camera
from .wrist_mounts import ActionImageGeometry, resolve_action_image_geometry


CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")


def decode_jpeg(value: np.bytes_ | bytes) -> np.ndarray:
    """Decode a RoboTwin JPEG byte payload to ``uint8[H,W,3]`` RGB."""
    with Image.open(io.BytesIO(bytes(value))) as image:
        return np.asarray(image.convert("RGB"))


@dataclass(frozen=True)
class ActionImageRenderConfig:
    """Action-image knobs shared by dataset construction and closed-loop render."""

    fov_scale: float = 1.0
    axis_length: float = 0.1
    sigma: float = 0.05
    wrist_look_distance: float = 0.25

    @classmethod
    def from_data_cfg(cls, train_cfg: Mapping[str, Any] | Any) -> "ActionImageRenderConfig":
        return cls(
            fov_scale=float(train_cfg.get("action_fov_scale", 1.0)),
            axis_length=float(train_cfg.get("action_axis_length", 0.1)),
            sigma=float(train_cfg.get("action_sigma", 0.05)),
            wrist_look_distance=float(train_cfg.get("wrist_look_distance", 0.25)),
        )


def resolve_geometry_from_data_cfg(train_cfg: Mapping[str, Any] | Any) -> ActionImageGeometry:
    """Resolve wrist/action transforms from a Hydra data.train config node."""
    geometry_embodiment = train_cfg.get("geometry_embodiment")
    if geometry_embodiment is None or str(geometry_embodiment).strip().lower() in {"", "none", "null"}:
        embodiments = list(train_cfg.get("embodiments") or [])
        if not embodiments:
            raise ValueError(
                "Configure data.train.geometry_embodiment or at least one embodiment."
            )
        geometry_embodiment = embodiments[0]
    return resolve_action_image_geometry(
        str(geometry_embodiment),
        wrist_cam_from_ee=train_cfg.get("wrist_cam_from_ee"),
        ee_from_action_frame=train_cfg.get("ee_from_action_frame"),
    )


def dual_segment_video_frames(num_frames: int, action_video_freq_ratio: int) -> tuple[int, int]:
    """Return ``(segment_frames, num_video_frames)`` for dual-segment checkpoints."""
    segment_frames = len(range(0, num_frames, action_video_freq_ratio))
    return segment_frames, 2 * segment_frames


def unwrap_observation(observation: dict) -> dict:
    """Accept either nested ``{'observation': ...}`` or a flat camera dict."""
    return observation["observation"] if "observation" in observation else observation


def as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    """Normalize common RoboTwin RGB encodings to ``uint8[H,W,3]``."""
    array = np.asarray(rgb)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0 + 1e-3:
            array = array * 255.0
        return np.clip(array, 0, 255).astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


def compose_robotwin_views(views: torch.Tensor) -> torch.Tensor:
    """Compose ``[3, T, C, H, W]`` as head over left/right wrist tiles → ``[T,C,384,320]``."""
    if views.ndim != 5 or views.shape[0] != 3:
        raise ValueError(f"Expected three camera videos [3, T, C, H, W], got {tuple(views.shape)}")
    head = transforms_f.resize(views[0], [256, 320], antialias=True)
    left = transforms_f.resize(views[1], [128, 160], antialias=True)
    right = transforms_f.resize(views[2], [128, 160], antialias=True)
    return torch.cat((head, torch.cat((left, right), dim=-1)), dim=-2)


def endpose_dict_to_action_16d(endpose: Mapping[str, Any]) -> np.ndarray:
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


def load_action_16d_from_hdf5(episode_file: h5py.File, indices: np.ndarray) -> np.ndarray:
    """Load ``[T, 16]`` endpose actions from a RoboTwin episode file."""
    endpose = episode_file["endpose"]
    return np.concatenate(
        (
            endpose["left_endpose"][:][indices],
            endpose["left_gripper"][:][indices, None],
            endpose["right_endpose"][:][indices],
            endpose["right_gripper"][:][indices, None],
        ),
        axis=-1,
    ).astype(np.float32)


def render_camera_action_images(
    cameras: Mapping[str, Mapping[str, Any]],
    action_16d: np.ndarray,
    *,
    geometry: ActionImageGeometry,
    render_cfg: ActionImageRenderConfig,
) -> list[torch.Tensor]:
    """Render per-camera action images as ``uint8/float [T,C,H,W]`` tensors in ``[0,1]``."""
    action = np.asarray(action_16d, dtype=np.float32)
    if action.ndim == 1:
        action = action.reshape(1, 16)
    if action.ndim != 2 or action.shape[1] != 16:
        raise ValueError(f"Expected action [T,16] or [16], got {action.shape}")

    views: list[torch.Tensor] = []
    for camera_name in CAMERA_NAMES:
        camera = cameras[camera_name]
        rgb = as_uint8_rgb(camera["rgb"]) if "rgb" in camera else None
        if rgb is not None and rgb.ndim == 3:
            height, width = rgb.shape[:2]
        elif "height" in camera and "width" in camera:
            height, width = int(camera["height"]), int(camera["width"])
        else:
            # Fall back to one decoded frame shape for HDF5 episode renderers.
            raise KeyError(
                f"Camera {camera_name} must provide rgb or explicit height/width for action-image render."
            )
        action_rgb = render_action_images_for_camera(
            camera_name,
            action,
            np.asarray(camera["extrinsic_cv"], dtype=np.float32),
            np.asarray(camera["intrinsic_cv"], dtype=np.float32),
            height,
            width,
            fov_scale=render_cfg.fov_scale,
            axis_length=render_cfg.axis_length,
            sigma=render_cfg.sigma,
            wrist_look_distance=render_cfg.wrist_look_distance,
            wrist_cam_from_ee=geometry.wrist_cam_from_ee,
            ee_from_action_frame=geometry.ee_from_action_frame,
        )
        views.append(torch.from_numpy(action_rgb).permute(0, 3, 1, 2))
    return views


def render_action_image_tensor(
    observation: dict,
    action_16d: np.ndarray,
    *,
    geometry: ActionImageGeometry,
    render_cfg: ActionImageRenderConfig | None = None,
    action_fov_scale: float | None = None,
    action_axis_length: float | None = None,
    action_sigma: float | None = None,
    wrist_look_distance: float | None = None,
) -> torch.Tensor:
    """Render the current 16D pose into training layout ``[1, 3, 384, 320]`` in ``[-1, 1]``."""
    if render_cfg is None:
        render_cfg = ActionImageRenderConfig(
            fov_scale=1.0 if action_fov_scale is None else float(action_fov_scale),
            axis_length=0.1 if action_axis_length is None else float(action_axis_length),
            sigma=0.05 if action_sigma is None else float(action_sigma),
            wrist_look_distance=(
                0.25 if wrist_look_distance is None else float(wrist_look_distance)
            ),
        )
    views = render_camera_action_images(
        unwrap_observation(observation),
        action_16d,
        geometry=geometry,
        render_cfg=render_cfg,
    )
    composed = compose_robotwin_views(torch.stack(views)).float()
    return composed.mul(2).sub(1)


def compose_robotwin_rgb_tensor(observation: dict) -> torch.Tensor:
    """Compose head/left/right RGB into ``[1, 3, 384, 320]`` in ``[-1, 1]``."""
    obs = unwrap_observation(observation)
    views = []
    for camera_name in CAMERA_NAMES:
        rgb = as_uint8_rgb(obs[camera_name]["rgb"])
        views.append(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0))
    composed = compose_robotwin_views(torch.stack(views)).float() / 255.0
    return composed.mul(2).sub(1)


def build_closed_loop_visuals(
    observation: dict,
    *,
    geometry: ActionImageGeometry,
    render_cfg: ActionImageRenderConfig,
    include_action_image: bool,
    blank_action_conditioning: bool = False,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor | None]:
    """Pack proprio + RGB (+ optional action-image) exactly as closed-loop infer expects.

    Returns:
        action_16d: raw ``[16]`` endpose
        input_image: ``[1,3,384,320]`` in ``[-1,1]``
        input_action_image: same layout, or ``None`` for scene-only models
    """
    obs = unwrap_observation(observation)
    if "endpose" not in observation and "endpose" not in obs:
        raise KeyError(
            "Observation is missing `endpose`. Enable data_type.endpose in the RoboTwin task config."
        )
    endpose = observation["endpose"] if "endpose" in observation else obs["endpose"]
    action_16d = endpose_dict_to_action_16d(endpose)
    input_image = compose_robotwin_rgb_tensor(observation)
    input_action_image = None
    if include_action_image:
        if blank_action_conditioning:
            input_action_image = torch.full_like(input_image, -1.0)
        else:
            input_action_image = render_action_image_tensor(
                observation,
                action_16d,
                geometry=geometry,
                render_cfg=render_cfg,
            )
    return action_16d, input_image, input_action_image
