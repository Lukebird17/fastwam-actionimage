"""Geometry and rendering for RoboTwin action images.

Head camera: true world→camera pinhole projection (no mount transform).

Wrist cameras: use a calibrated rigid ``T_cam←ee`` matrix per side
(``wrist_cam_from_ee['left'|'right']``).  Future EE motion is relative to the
window-start EE and viewed along that mount baseline.  Left/right mounts are
distinct so new embodiments only need to supply new 4×4 pairs.
"""

from __future__ import annotations

import numpy as np


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized-or-unnormalized ``[..., w, x, y, z]`` to rotation matrices."""
    quaternion = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    quaternion = quaternion / np.clip(norm, 1e-8, None)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)

    matrix = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def pose7_to_matrix(pose_wxyz: np.ndarray) -> np.ndarray:
    """``[x,y,z,qw,qx,qy,qz] -> 4x4`` world-from-EE."""
    pose = np.asarray(pose_wxyz, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_wxyz_to_matrix(pose[3:7].astype(np.float32)).astype(np.float64)
    matrix[:3, 3] = pose[:3]
    return matrix


def extrinsic_cv_to_world_to_camera(extrinsic_cv: np.ndarray) -> np.ndarray:
    """RoboTwin ``[R|t]`` (3×4 or 4×4) -> 4×4 world-to-camera."""
    extrinsic = np.asarray(extrinsic_cv, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = extrinsic[:3, :4]
    return matrix


def camera_from_ee_transform(
    world_to_camera_cv: np.ndarray,
    ee_pose_wxyz: np.ndarray,
) -> np.ndarray:
    """Return rigid ``T_cam←ee`` (EE pose expressed in the camera frame)."""
    world_to_camera = extrinsic_cv_to_world_to_camera(world_to_camera_cv)
    world_from_ee = pose7_to_matrix(ee_pose_wxyz)
    return world_to_camera @ world_from_ee


def widen_intrinsics(intrinsics: np.ndarray, scale: float, height: int, width: int) -> np.ndarray:
    """Widen the rendered action-image FOV without changing its pixel grid."""
    if scale <= 0:
        raise ValueError(f"FOV scale must be positive, got {scale}")
    intrinsics = np.asarray(intrinsics, dtype=np.float32).copy()
    if scale == 1:
        return intrinsics
    center_x, center_y = (width - 1) / 2, (height - 1) / 2
    intrinsics[0, 0] /= scale
    intrinsics[1, 1] /= scale
    intrinsics[0, 2] = center_x + (intrinsics[0, 2] - center_x) / scale
    intrinsics[1, 2] = center_y + (intrinsics[1, 2] - center_y) / scale
    return intrinsics


def _look_at_world_to_camera(camera_center: np.ndarray) -> np.ndarray:
    """OpenCV world-to-camera that sits at ``camera_center`` and looks at the origin."""
    center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    distance = np.linalg.norm(center)
    if distance < 1e-8:
        raise ValueError("camera_center must be away from the origin")
    z_axis = -center / distance
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(z_axis @ up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation_c2w = np.stack((x_axis, y_axis, z_axis), axis=1)
    rotation_w2c = rotation_c2w.T
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_w2c
    matrix[:3, 3] = -rotation_w2c @ center
    return matrix


def project_points_camera_frame(
    points_camera: np.ndarray,
    intrinsics_cv: np.ndarray,
    height: int | None = None,
    width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project points already expressed in the OpenCV camera frame."""
    points = np.asarray(points_camera, dtype=np.float32)
    intrinsics = np.asarray(intrinsics_cv, dtype=np.float32)
    homogeneous = np.einsum("ij,...j->...i", intrinsics, points)
    depth = homogeneous[..., 2]
    safe_depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    pixels = homogeneous[..., :2] / safe_depth[..., None]
    valid = depth > 1e-6
    if height is not None and width is not None:
        valid = (
            valid
            & (pixels[..., 0] >= 0)
            & (pixels[..., 0] < width)
            & (pixels[..., 1] >= 0)
            & (pixels[..., 1] < height)
        )
    return pixels, valid


def project_world_points(
    points_world: np.ndarray,
    world_to_camera_cv: np.ndarray,
    intrinsics_cv: np.ndarray,
    height: int | None = None,
    width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with RoboTwin's OpenCV ``[R|t]`` extrinsic."""
    points_world = np.asarray(points_world, dtype=np.float32)
    extrinsic = np.asarray(world_to_camera_cv, dtype=np.float32)
    camera = np.einsum("ij,...j->...i", extrinsic[:, :3], points_world) + extrinsic[:, 3]
    return project_points_camera_frame(camera, intrinsics_cv, height=height, width=width)


def _gaussian(points: np.ndarray, valid: np.ndarray, height: int, width: int, sigma: float) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    sigma_pixels = min(height, width) * sigma
    dx = x[None] - points[:, 0, None, None]
    dy = y[None] - points[:, 1, None, None]
    heatmap = np.exp(-(dx * dx + dy * dy) / (2 * sigma_pixels**2)).astype(np.float32)
    heatmap[~valid] = 0
    return heatmap


def _render_from_camera_points(
    position_cam: np.ndarray,
    forward_cam: np.ndarray,
    up_cam: np.ndarray,
    gripper: np.ndarray,
    intrinsics_cv: np.ndarray,
    height: int,
    width: int,
    sigma: float,
) -> np.ndarray:
    position_2d, position_valid = project_points_camera_frame(
        position_cam, intrinsics_cv, height=height, width=width
    )
    forward_2d, forward_valid = project_points_camera_frame(
        forward_cam, intrinsics_cv, height=height, width=width
    )
    up_2d, up_valid = project_points_camera_frame(
        up_cam, intrinsics_cv, height=height, width=width
    )
    red = _gaussian(position_2d, position_valid, height, width, sigma)
    green = _gaussian(forward_2d, forward_valid, height, width, sigma)
    blue = _gaussian(up_2d, up_valid, height, width, sigma)
    gripper_floor = ((gripper > 0.5) & position_valid)[:, None, None] * 0.25
    blue = np.where(blue <= 0.25, gripper_floor, blue)
    return np.stack((red, green, blue), axis=-1)


def render_arm_action_images(
    poses_wxyz: np.ndarray,
    gripper: np.ndarray,
    world_to_camera_cv: np.ndarray,
    intrinsics_cv: np.ndarray,
    height: int,
    width: int,
    *,
    axis_length: float = 0.1,
    sigma: float = 0.05,
    fov_scale: float = 1.0,
) -> np.ndarray:
    """Render one arm via true world→camera projection (head-camera path)."""
    poses = np.asarray(poses_wxyz, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Expected poses [T, 7], got {poses.shape}")
    if gripper.shape != (poses.shape[0],):
        raise ValueError(f"Expected gripper [T], got {gripper.shape}")

    rotation = quaternion_wxyz_to_matrix(poses[:, 3:7])
    position = poses[:, :3]
    forward = position + rotation[..., :, 0] * axis_length
    up = position - rotation[..., :, 2] * axis_length
    intrinsics = widen_intrinsics(intrinsics_cv, fov_scale, height, width)

    extrinsic = np.asarray(world_to_camera_cv, dtype=np.float32)
    position_cam = np.einsum("ij,...j->...i", extrinsic[:, :3], position) + extrinsic[:, 3]
    forward_cam = np.einsum("ij,...j->...i", extrinsic[:, :3], forward) + extrinsic[:, 3]
    up_cam = np.einsum("ij,...j->...i", extrinsic[:, :3], up) + extrinsic[:, 3]
    return _render_from_camera_points(
        position_cam, forward_cam, up_cam, gripper, intrinsics, height, width, sigma
    )


def render_wrist_arm_action_images(
    poses_wxyz: np.ndarray,
    gripper: np.ndarray,
    cam_from_ee: np.ndarray,
    intrinsics_cv: np.ndarray,
    height: int,
    width: int,
    *,
    axis_length: float = 0.1,
    sigma: float = 0.05,
    fov_scale: float = 1.0,
    wrist_look_distance: float = 0.25,
) -> np.ndarray:
    """Render ipsilateral wrist action images with a calibrated ``T_cam←ee``.

    ``cam_from_ee`` is the rigid wrist mount (left and right are distinct matrices).
    Future EE poses are expressed relative to the window-start EE, then viewed by a
    virtual camera on that mount baseline looking at the EE.
    """
    poses = np.asarray(poses_wxyz, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32)
    cam_from_ee = np.asarray(cam_from_ee, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Expected poses [T, 7], got {poses.shape}")
    if gripper.shape != (poses.shape[0],):
        raise ValueError(f"Expected gripper [T], got {gripper.shape}")
    if cam_from_ee.shape != (4, 4):
        raise ValueError(f"cam_from_ee must be 4x4, got {cam_from_ee.shape}")
    if wrist_look_distance <= 0:
        raise ValueError(f"wrist_look_distance must be positive, got {wrist_look_distance}")

    ee_from_cam = np.linalg.inv(cam_from_ee)
    camera_center_ee = ee_from_cam[:3, 3]
    baseline = np.linalg.norm(camera_center_ee)
    if baseline < 1e-6:
        raise ValueError("Degenerate cam←ee transform (zero baseline).")
    virtual_center = camera_center_ee / baseline * wrist_look_distance
    world_to_virtual = _look_at_world_to_camera(virtual_center)
    axis = min(float(axis_length), 0.35 * float(wrist_look_distance))

    ee0 = pose7_to_matrix(poses[0])
    ee0_inv = np.linalg.inv(ee0)
    position = np.empty((len(poses), 3), dtype=np.float64)
    forward = np.empty_like(position)
    up = np.empty_like(position)
    for index, pose in enumerate(poses):
        # Relative EE motion, then place through the mount matrix conceptually:
        # T_cam_ee_t = cam_from_ee @ inv(ee0) @ ee_t  (used via EE0-frame points).
        relative = ee0_inv @ pose7_to_matrix(pose)
        origin = relative[:3, 3]
        rotation = relative[:3, :3]
        position[index] = origin
        forward[index] = origin + rotation[:, 0] * axis
        up[index] = origin - rotation[:, 2] * axis

    rotation_w2c = world_to_virtual[:3, :3]
    translation = world_to_virtual[:3, 3]
    position_cam = (rotation_w2c @ position.T).T + translation
    forward_cam = (rotation_w2c @ forward.T).T + translation
    up_cam = (rotation_w2c @ up.T).T + translation
    intrinsics = widen_intrinsics(intrinsics_cv, fov_scale, height, width)
    return _render_from_camera_points(
        position_cam.astype(np.float32),
        forward_cam.astype(np.float32),
        up_cam.astype(np.float32),
        gripper,
        intrinsics,
        height,
        width,
        sigma,
    )


def render_bimanual_action_images(
    action_16d: np.ndarray,
    world_to_camera_cv: np.ndarray,
    intrinsics_cv: np.ndarray,
    height: int,
    width: int,
    **render_kwargs,
) -> np.ndarray:
    """Render both arms with true world projection (head path)."""
    action = np.asarray(action_16d, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 16:
        raise ValueError(f"Expected bimanual action [T, 16], got {action.shape}")
    # Wrist-only kwargs — ignore on head path.
    render_kwargs.pop("wrist_look_distance", None)
    render_kwargs.pop("wrist_cam_from_ee", None)
    left = render_arm_action_images(
        action[:, :7],
        action[:, 7],
        world_to_camera_cv,
        intrinsics_cv,
        height,
        width,
        **render_kwargs,
    )
    right = render_arm_action_images(
        action[:, 8:15],
        action[:, 15],
        world_to_camera_cv,
        intrinsics_cv,
        height,
        width,
        **render_kwargs,
    )
    return np.clip(left + right, 0, 1)


def render_action_images_for_camera(
    camera_name: str,
    action_16d: np.ndarray,
    world_to_camera_cv: np.ndarray,
    intrinsics_cv: np.ndarray,
    height: int,
    width: int,
    *,
    wrist_cam_from_ee: dict[str, np.ndarray] | None = None,
    **render_kwargs,
) -> np.ndarray:
    """Dispatch head (true projection) vs wrist (calibrated ``T_cam←ee``)."""
    action = np.asarray(action_16d, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 16:
        raise ValueError(f"Expected bimanual action [T, 16], got {action.shape}")

    if camera_name == "head_camera":
        return render_bimanual_action_images(
            action,
            world_to_camera_cv,
            intrinsics_cv,
            height,
            width,
            **render_kwargs,
        )
    if camera_name in {"left_camera", "right_camera"}:
        if wrist_cam_from_ee is None:
            raise ValueError(
                "wrist_cam_from_ee={left,right} is required for wrist action-image rendering."
            )
        side = "left" if camera_name == "left_camera" else "right"
        pose_slice = action[:, :7] if side == "left" else action[:, 8:15]
        grip_slice = action[:, 7] if side == "left" else action[:, 15]
        return render_wrist_arm_action_images(
            pose_slice,
            grip_slice,
            wrist_cam_from_ee[side],
            intrinsics_cv,
            height,
            width,
            **render_kwargs,
        )
    raise ValueError(f"Unknown camera_name: {camera_name}")
