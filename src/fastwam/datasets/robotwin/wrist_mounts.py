"""Per-embodiment wrist-camera and canonical action-frame transforms.

``T_cam_from_ee`` is a 4×4 matrix expressing the EE pose in the wrist OpenCV
camera frame (``p_cam = T_cam_from_ee @ p_ee``).  Left and right are stored
separately so new embodiments can swap in their own calibration without
touching render code.

``T_ee_from_action`` maps the canonical Action Images frame into the robot's EE
frame. In the canonical frame, +X is the gripper-plane normal (green) and -Z is
the in-plane up direction (blue).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

# Aloha-agilex (RoboTwin): measured rigid mount, identical to numerical precision
# on left/right for this robot.  Kept as two entries for embodiment generality.
_ALOHA_AGILEX_LEFT = [
    [0.0, -1.0, 0.0, 0.032],
    [-0.38941823, 0.0, -0.92106107, 0.08712835],
    [0.92106108, 0.0, -0.38941822, -0.03916204],
    [0.0, 0.0, 0.0, 1.0],
]
_ALOHA_AGILEX_RIGHT = [
    [0.0, -1.0, 0.0, 0.032],
    [-0.38941837, 0.0, -0.92106097, 0.08712839],
    [0.92106118, 0.0, -0.38941837, -0.03916218],
    [0.0, 0.0, 0.0, 1.0],
]

# Map RoboTwin embodiment name prefixes / exact keys -> left/right mounts.
WRIST_CAM_FROM_EE: dict[str, dict[str, list[list[float]]]] = {
    "aloha-agilex": {
        "left": _ALOHA_AGILEX_LEFT,
        "right": _ALOHA_AGILEX_RIGHT,
    },
    "aloha-agilex_clean_50": {
        "left": _ALOHA_AGILEX_LEFT,
        "right": _ALOHA_AGILEX_RIGHT,
    },
    "aloha-agilex_randomized_500": {
        "left": _ALOHA_AGILEX_LEFT,
        "right": _ALOHA_AGILEX_RIGHT,
    },
}

# RoboTwin Aloha endpose axes differ from the canonical Action Images tool frame:
#   canonical +X (normal / green) -> EE -Z
#   canonical -Z (up / blue)      -> EE -X
_ALOHA_EE_FROM_ACTION = [
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

EE_FROM_ACTION_FRAME: dict[str, dict[str, list[list[float]]]] = {
    "aloha-agilex": {
        "left": _ALOHA_EE_FROM_ACTION,
        "right": _ALOHA_EE_FROM_ACTION,
    },
    "aloha-agilex_clean_50": {
        "left": _ALOHA_EE_FROM_ACTION,
        "right": _ALOHA_EE_FROM_ACTION,
    },
    "aloha-agilex_randomized_500": {
        "left": _ALOHA_EE_FROM_ACTION,
        "right": _ALOHA_EE_FROM_ACTION,
    },
}


def _as_matrix4(value: object, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got shape {matrix.shape}")
    return matrix


def _normalize_side_matrices(
    values: Mapping[str, object] | None,
    *,
    name: str,
) -> dict[str, np.ndarray] | None:
    """Validate ``{left: 4x4, right: 4x4}`` (or None)."""
    if values is None:
        return None
    if "left" not in values or "right" not in values:
        raise ValueError(f"{name} must contain both 'left' and 'right' 4x4 matrices.")
    return {
        "left": _as_matrix4(values["left"], name=f"{name}['left']"),
        "right": _as_matrix4(values["right"], name=f"{name}['right']"),
    }


def resolve_wrist_cam_from_ee(
    wrist_cam_from_ee: Mapping[str, object] | None = None,
    *,
    embodiment: str | None = None,
) -> dict[str, np.ndarray]:
    """Return left/right ``T_cam←ee`` matrices.

    Priority:
    1. explicit ``wrist_cam_from_ee`` dict from config;
    2. lookup ``embodiment`` in ``WRIST_CAM_FROM_EE`` (exact, then prefix ``aloha-agilex``);
    3. error — callers must not silently invent mounts for unknown robots.
    """
    explicit = _normalize_side_matrices(wrist_cam_from_ee, name="wrist_cam_from_ee")
    if explicit is not None:
        return explicit
    if embodiment is None:
        raise ValueError(
            "wrist_cam_from_ee is required when embodiment is not provided. "
            "Pass config wrist_cam_from_ee.{left,right} or a known embodiment name."
        )
    if embodiment in WRIST_CAM_FROM_EE:
        return _normalize_side_matrices(WRIST_CAM_FROM_EE[embodiment], name="wrist_cam_from_ee")  # type: ignore[return-value]
    # Prefer longer / more specific prefixes (e.g. aloha-agilex_clean_50 before aloha-agilex).
    for key in sorted(WRIST_CAM_FROM_EE, key=len, reverse=True):
        if embodiment.startswith(key):
            return _normalize_side_matrices(WRIST_CAM_FROM_EE[key], name="wrist_cam_from_ee")  # type: ignore[return-value]
    raise KeyError(
        f"No wrist_cam_from_ee mount for embodiment={embodiment!r}. "
        f"Known keys: {sorted(WRIST_CAM_FROM_EE)}. "
        "Add a calibrated left/right T_cam←ee pair."
    )


def resolve_ee_from_action_frame(
    ee_from_action_frame: Mapping[str, object] | None = None,
    *,
    embodiment: str | None = None,
) -> dict[str, np.ndarray]:
    """Return per-side ``T_ee←action`` matrices."""
    explicit = _normalize_side_matrices(ee_from_action_frame, name="ee_from_action_frame")
    if explicit is not None:
        return explicit
    if embodiment is None:
        raise ValueError(
            "ee_from_action_frame is required when embodiment is not provided. "
            "Pass config ee_from_action_frame.{left,right} or a known embodiment name."
        )
    if embodiment in EE_FROM_ACTION_FRAME:
        return _normalize_side_matrices(EE_FROM_ACTION_FRAME[embodiment], name="ee_from_action_frame")  # type: ignore[return-value]
    for key in sorted(EE_FROM_ACTION_FRAME, key=len, reverse=True):
        if embodiment.startswith(key):
            return _normalize_side_matrices(EE_FROM_ACTION_FRAME[key], name="ee_from_action_frame")  # type: ignore[return-value]
    raise KeyError(
        f"No ee_from_action_frame for embodiment={embodiment!r}. "
        f"Known keys: {sorted(EE_FROM_ACTION_FRAME)}. "
        "Add a calibrated left/right T_ee←action pair."
    )


@dataclass(frozen=True)
class ActionImageGeometry:
    """Resolved per-side geometry used by every train/inference/eval renderer."""

    embodiment: str
    wrist_cam_from_ee: dict[str, np.ndarray]
    ee_from_action_frame: dict[str, np.ndarray]


def resolve_action_image_geometry(
    embodiment: str,
    *,
    wrist_cam_from_ee: Mapping[str, object] | None = None,
    ee_from_action_frame: Mapping[str, object] | None = None,
) -> ActionImageGeometry:
    """Resolve both transform families from one embodiment key."""
    return ActionImageGeometry(
        embodiment=embodiment,
        wrist_cam_from_ee=resolve_wrist_cam_from_ee(
            wrist_cam_from_ee,
            embodiment=embodiment,
        ),
        ee_from_action_frame=resolve_ee_from_action_frame(
            ee_from_action_frame,
            embodiment=embodiment,
        ),
    )
