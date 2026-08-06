"""Per-embodiment rigid wrist-camera ↔ end-effector mounts.

``T_cam_from_ee`` is a 4×4 matrix expressing the EE pose in the wrist OpenCV
camera frame (``p_cam = T_cam_from_ee @ p_ee``).  Left and right are stored
separately so new embodiments can swap in their own calibration without
touching render code.
"""

from __future__ import annotations

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


def _as_matrix4(value: object, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got shape {matrix.shape}")
    return matrix


def normalize_wrist_cam_from_ee(
    wrist_cam_from_ee: Mapping[str, object] | None,
) -> dict[str, np.ndarray] | None:
    """Validate ``{left: 4x4, right: 4x4}`` (or None)."""
    if wrist_cam_from_ee is None:
        return None
    if "left" not in wrist_cam_from_ee or "right" not in wrist_cam_from_ee:
        raise ValueError("wrist_cam_from_ee must contain both 'left' and 'right' 4x4 matrices.")
    return {
        "left": _as_matrix4(wrist_cam_from_ee["left"], name="wrist_cam_from_ee['left']"),
        "right": _as_matrix4(wrist_cam_from_ee["right"], name="wrist_cam_from_ee['right']"),
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
    explicit = normalize_wrist_cam_from_ee(wrist_cam_from_ee)
    if explicit is not None:
        return explicit
    if embodiment is None:
        raise ValueError(
            "wrist_cam_from_ee is required when embodiment is not provided. "
            "Pass config wrist_cam_from_ee.{left,right} or a known embodiment name."
        )
    if embodiment in WRIST_CAM_FROM_EE:
        return normalize_wrist_cam_from_ee(WRIST_CAM_FROM_EE[embodiment])  # type: ignore[return-value]
    # Prefer longer / more specific prefixes (e.g. aloha-agilex_clean_50 before aloha-agilex).
    for key in sorted(WRIST_CAM_FROM_EE, key=len, reverse=True):
        if embodiment.startswith(key):
            return normalize_wrist_cam_from_ee(WRIST_CAM_FROM_EE[key])  # type: ignore[return-value]
    raise KeyError(
        f"No wrist_cam_from_ee mount for embodiment={embodiment!r}. "
        f"Known keys: {sorted(WRIST_CAM_FROM_EE)}. "
        "Add a calibrated left/right T_cam←ee pair."
    )
