#!/usr/bin/env python3
"""3D trajectory video + interactive HTML for RoboTwin EE poses and cameras.

Run from repo root:
  PYTHONPATH=src python vis/visualize_robotwin_action_3d.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw

from fastwam.datasets.robotwin.action_image import quaternion_wxyz_to_matrix
from fastwam.datasets.robotwin.obs_utils import decode_jpeg
from fastwam.datasets.robotwin.wrist_mounts import resolve_action_image_geometry


CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
CAMERA_COLORS = {
    "head_camera": "#1f77b4",
    "left_camera": "#2ca02c",
    "right_camera": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode",
        default="data/RoboTwin2.0/dataset/move_playingcard_away/aloha-agilex_clean_50/data/episode0.hdf5",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--stride", type=int, default=1, help="Sample every N raw frames for the video.")
    parser.add_argument("--axis-length", type=float, default=0.06)
    parser.add_argument("--camera-axis-length", type=float, default=0.045)
    parser.add_argument("--frustum-depth", type=float, default=0.08)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        default="./outputs/robotwin_action_image_vis_3d",
    )
    return parser.parse_args()


def camera_from_extrinsic_cv(extrinsic_cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    extrinsic = np.asarray(extrinsic_cv, dtype=np.float64)
    rotation_w2c = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    center = -rotation_w2c.T @ translation
    rotation_c2w = rotation_w2c.T
    return center.astype(np.float32), rotation_c2w.astype(np.float32)


def load_window(episode: Path, start: int, num_frames: int, stride: int) -> dict:
    indices = start + np.arange(0, num_frames, stride)
    with h5py.File(episode, "r") as file:
        length = int(file["endpose/left_endpose"].shape[0])
        indices = np.clip(indices, 0, length - 1)
        left_pose = np.asarray(file["endpose/left_endpose"][:][indices], dtype=np.float32)
        right_pose = np.asarray(file["endpose/right_endpose"][:][indices], dtype=np.float32)

        cameras = {}
        for name in CAMERA_NAMES:
            cam = file["observation"][name]
            extrinsic0 = np.asarray(cam["extrinsic_cv"][indices[0]], dtype=np.float32)
            intrinsic0 = np.asarray(cam["intrinsic_cv"][indices[0]], dtype=np.float32)
            rgb0 = decode_jpeg(cam["rgb"][indices[0]])
            centers, rotations = [], []
            for frame in indices:
                center, rotation = camera_from_extrinsic_cv(cam["extrinsic_cv"][frame])
                centers.append(center)
                rotations.append(rotation)
            cameras[name] = {
                "extrinsic0": extrinsic0,
                "intrinsic0": intrinsic0,
                "rgb0": rgb0,
                "centers": np.stack(centers),
                "rotations": np.stack(rotations),
                "center0": centers[0],
                "rotation0": rotations[0],
            }

    geometry = resolve_action_image_geometry(episode.parents[1].name)
    return {
        "indices": indices,
        "left_pose": left_pose,
        "right_pose": right_pose,
        "ee_from_action_frame": geometry.ee_from_action_frame,
        "cameras": cameras,
    }


def frustum_corners(
    center: np.ndarray,
    rotation_c2w: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    depth: float,
) -> np.ndarray:
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    corners_px = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float64,
    )
    corners_cam = np.stack(
        [
            (corners_px[:, 0] - cx) / fx * depth,
            (corners_px[:, 1] - cy) / fy * depth,
            np.full(4, depth),
        ],
        axis=1,
    )
    return (rotation_c2w @ corners_cam.T).T + center


def set_equal_aspect(ax, points: np.ndarray, margin: float = 1.15):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = max(0.5 * float((maxs - mins).max()) * margin, 0.15)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def draw_small_camera(
    ax,
    name: str,
    center: np.ndarray,
    rotation: np.ndarray,
    intrinsic: np.ndarray,
    rgb_shape: tuple[int, int],
    axis_length: float,
    frustum_depth: float,
    *,
    show_label: bool = False,
):
    color = CAMERA_COLORS[name]
    ax.scatter(center[0], center[1], center[2], color=color, s=18, marker="s", depthshade=False)
    # Optical axis only (keep cameras compact).
    tip = center + rotation[:, 2] * axis_length
    ax.plot(
        [center[0], tip[0]],
        [center[1], tip[1]],
        [center[2], tip[2]],
        color=color,
        linewidth=1.6,
        alpha=0.9,
    )
    height, width = rgb_shape
    corners = frustum_corners(center, rotation, intrinsic, height, width, frustum_depth)
    for corner in corners:
        ax.plot(
            [center[0], corner[0]],
            [center[1], corner[1]],
            [center[2], corner[2]],
            color=color,
            alpha=0.2,
            linewidth=0.6,
        )
    ax.add_collection3d(
        Poly3DCollection([corners], alpha=0.08, facecolor=color, edgecolor=color, linewidths=0.5)
    )
    if show_label:
        ax.text(center[0], center[1], center[2], f" {name.replace('_camera', '')}", color=color, fontsize=7)


def draw_ee_state(
    ax,
    poses: np.ndarray,
    ee_from_action_frame: np.ndarray,
    index: int,
    color: str,
    axis_length: float,
    label: str | None,
):
    rotation = quaternion_wxyz_to_matrix(poses[:, 3:7]) @ ee_from_action_frame[:3, :3]
    position = poses[:, :3]
    trail = position[: index + 1]
    ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=color, linewidth=2.0, label=label, alpha=0.95)
    if len(position) > index + 1:
        ax.plot(
            position[index:, 0],
            position[index:, 1],
            position[index:, 2],
            color=color,
            linewidth=1.0,
            alpha=0.25,
        )
    origin = position[index]
    normal = origin + rotation[index, :, 0] * axis_length
    up = origin - rotation[index, :, 2] * axis_length
    ax.scatter(origin[0], origin[1], origin[2], color="#e41a1c", s=22, depthshade=False)
    ax.plot([origin[0], normal[0]], [origin[1], normal[1]], [origin[2], normal[2]], color="#4daf4a", lw=2.0)
    ax.plot([origin[0], up[0]], [origin[1], up[1]], [origin[2], up[2]], color="#377eb8", lw=2.0)


def collect_bounds(data: dict) -> np.ndarray:
    points = [data["left_pose"][:, :3], data["right_pose"][:, :3]]
    for camera in data["cameras"].values():
        points.append(camera["centers"])
    return np.concatenate(points, axis=0)


def render_frame(
    data: dict,
    index: int,
    bounds: np.ndarray,
    axis_length: float,
    camera_axis_length: float,
    frustum_depth: float,
    elev: float,
    azim: float,
) -> np.ndarray:
    fig = plt.figure(figsize=(8.5, 7.0), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    draw_ee_state(
        ax,
        data["left_pose"],
        data["ee_from_action_frame"]["left"],
        index,
        "#ff7f0e",
        axis_length,
        "left_ee" if index == 0 else None,
    )
    draw_ee_state(
        ax,
        data["right_pose"],
        data["ee_from_action_frame"]["right"],
        index,
        "#9467bd",
        axis_length,
        "right_ee" if index == 0 else None,
    )

    for name, camera in data["cameras"].items():
        # Action-image convention: fixed window-start extrinsics.
        draw_small_camera(
            ax,
            name,
            camera["center0"],
            camera["rotation0"],
            camera["intrinsic0"],
            camera["rgb0"].shape[:2],
            camera_axis_length,
            frustum_depth,
            show_label=(index == 0),
        )
        # Light dashed path of moving wrist cams for context.
        path = camera["centers"][: index + 1]
        if len(path) > 1 and name != "head_camera":
            ax.plot(
                path[:, 0],
                path[:, 1],
                path[:, 2],
                color=CAMERA_COLORS[name],
                linestyle="--",
                linewidth=0.8,
                alpha=0.35,
            )

    set_equal_aspect(ax, bounds)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    frame_id = int(data["indices"][index])
    ax.set_title(
        f"t={index}/{len(data['indices']) - 1}  raw_frame={frame_id}\n"
        "solid trail=EE so far; green=normal, blue=up; cameras@window-start (small)",
        fontsize=9,
    )
    if index == 0:
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = np.asarray(rgba)[..., :3].copy()
    plt.close(fig)
    return frame


def write_video(data: dict, args: argparse.Namespace, output_path: Path) -> None:
    bounds = collect_bounds(data)
    frames = []
    n = len(data["indices"])
    for index in range(n):
        # Slow orbit so the trajectory is readable in 3D.
        azim = -60 + 90.0 * index / max(n - 1, 1)
        frame = render_frame(
            data,
            index,
            bounds,
            args.axis_length,
            args.camera_axis_length,
            args.frustum_depth,
            elev=22,
            azim=azim,
        )
        frames.append(frame)
        print(f"render frame {index + 1}/{n}", flush=True)
    imageio.mimsave(output_path, frames, fps=args.fps, codec="libx264")
    print(f"wrote: {output_path}")


def write_html(data: dict, output_path: Path, axis_length: float, camera_axis_length: float) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed; skipped HTML")
        return

    fig = go.Figure()
    n = len(data["indices"])

    for arm_name, poses, ee_from_action, color in (
        ("left_ee", data["left_pose"], data["ee_from_action_frame"]["left"], "#ff7f0e"),
        ("right_ee", data["right_pose"], data["ee_from_action_frame"]["right"], "#9467bd"),
    ):
        position = poses[:, :3]
        rotation = quaternion_wxyz_to_matrix(poses[:, 3:7]) @ ee_from_action[:3, :3]
        fig.add_trace(
            go.Scatter3d(
                x=position[:, 0],
                y=position[:, 1],
                z=position[:, 2],
                mode="lines+markers",
                name=arm_name,
                line=dict(color=color, width=6),
                marker=dict(size=3, color=color),
            )
        )
        # Animated current EE marker via frames below; also static axes samples.
        xs, ys, zs = [], [], []
        for index in range(0, n, max(1, n // 10)):
            origin = position[index]
            normal = origin + rotation[index, :, 0] * axis_length
            up = origin - rotation[index, :, 2] * axis_length
            for tip in (normal, up):
                xs += [origin[0], tip[0], None]
                ys += [origin[1], tip[1], None]
                zs += [origin[2], tip[2], None]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=f"{arm_name}_axes",
                line=dict(color=color, width=3),
                opacity=0.55,
                showlegend=False,
            )
        )

    for name, camera in data["cameras"].items():
        center = camera["center0"]
        rotation = camera["rotation0"]
        color = CAMERA_COLORS[name]
        tip = center + rotation[:, 2] * camera_axis_length
        fig.add_trace(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode="markers+text",
                text=[name.replace("_camera", "")],
                textposition="top center",
                marker=dict(size=4, color=color, symbol="square"),
                name=name,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[center[0], tip[0]],
                y=[center[1], tip[1]],
                z=[center[2], tip[2]],
                mode="lines",
                line=dict(color=color, width=5),
                name=f"{name}_z",
                showlegend=False,
            )
        )

    # Playable trajectory: moving EE points.
    frames = []
    for index in range(n):
        left = data["left_pose"][index, :3]
        right = data["right_pose"][index, :3]
        left_trail = data["left_pose"][: index + 1, :3]
        right_trail = data["right_pose"][: index + 1, :3]
        frames.append(
            go.Frame(
                name=str(index),
                data=[
                    go.Scatter3d(
                        x=left_trail[:, 0],
                        y=left_trail[:, 1],
                        z=left_trail[:, 2],
                        mode="lines",
                        line=dict(color="#ff7f0e", width=6),
                    ),
                    go.Scatter3d(
                        x=right_trail[:, 0],
                        y=right_trail[:, 1],
                        z=right_trail[:, 2],
                        mode="lines",
                        line=dict(color="#9467bd", width=6),
                    ),
                    go.Scatter3d(
                        x=[left[0], right[0]],
                        y=[left[1], right[1]],
                        z=[left[2], right[2]],
                        mode="markers",
                        marker=dict(size=5, color=["#e41a1c", "#e41a1c"]),
                    ),
                ],
                traces=[0, 1, len(fig.data)],  # will append marker trace
            )
        )

    # Current EE markers (animated).
    fig.add_trace(
        go.Scatter3d(
            x=[data["left_pose"][0, 0], data["right_pose"][0, 0]],
            y=[data["left_pose"][0, 1], data["right_pose"][0, 1]],
            z=[data["left_pose"][0, 2], data["right_pose"][0, 2]],
            mode="markers",
            marker=dict(size=5, color="#e41a1c"),
            name="ee_now",
        )
    )
    marker_trace = len(fig.data) - 1
    fixed_trace_count = marker_trace  # 0..marker_trace-1 stay; we rebuild frames cleanly

    frames = []
    for index in range(n):
        left_trail = data["left_pose"][: index + 1, :3]
        right_trail = data["right_pose"][: index + 1, :3]
        left = data["left_pose"][index, :3]
        right = data["right_pose"][index, :3]
        frame_data = []
        # Replace only left trail, right trail, and current markers.
        frame_data.append(
            go.Scatter3d(
                x=left_trail[:, 0],
                y=left_trail[:, 1],
                z=left_trail[:, 2],
                mode="lines+markers",
                line=dict(color="#ff7f0e", width=6),
                marker=dict(size=3, color="#ff7f0e"),
            )
        )
        frame_data.append(
            go.Scatter3d(
                x=right_trail[:, 0],
                y=right_trail[:, 1],
                z=right_trail[:, 2],
                mode="lines+markers",
                line=dict(color="#9467bd", width=6),
                marker=dict(size=3, color="#9467bd"),
            )
        )
        # Keep other traces unchanged by only updating traces 0,1,marker
        frame_data.append(
            go.Scatter3d(
                x=[left[0], right[0]],
                y=[left[1], right[1]],
                z=[left[2], right[2]],
                mode="markers",
                marker=dict(size=6, color="#e41a1c"),
            )
        )
        frames.append(go.Frame(data=frame_data, traces=[0, 1, marker_trace], name=str(index)))

    fig.frames = frames
    fig.update_layout(
        title="RoboTwin EE trajectory + cameras@window-start",
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=1100,
        height=800,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 120, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [str(i)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": str(i),
                        "method": "animate",
                    }
                    for i in range(n)
                ],
                "x": 0.1,
                "len": 0.8,
                "currentvalue": {"prefix": "frame: "},
            }
        ],
    )
    # silence unused
    _ = fixed_trace_count
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"wrote: {output_path}")


def write_cam_strip(data: dict, output_path: Path) -> None:
    rgbs = [data["cameras"][name]["rgb0"] for name in CAMERA_NAMES]
    target_h = min(img.shape[0] for img in rgbs)
    resized = []
    for img in rgbs:
        if img.shape[0] != target_h:
            new_w = int(round(img.shape[1] * target_h / img.shape[0]))
            yy = np.linspace(0, img.shape[0] - 1, target_h).astype(np.int32)
            xx = np.linspace(0, img.shape[1] - 1, new_w).astype(np.int32)
            img = img[yy][:, xx]
        canvas = img.copy()
        canvas[:18] = 0
        resized.append(canvas)
    strip = np.concatenate(resized, axis=1)
    image = Image.fromarray(strip)
    draw = ImageDraw.Draw(image)
    x = 6
    for name, img in zip(CAMERA_NAMES, resized):
        draw.text((x, 2), name, fill=(255, 255, 255))
        x += img.shape[1]
    image.save(output_path)
    print(f"wrote: {output_path}")


def print_camera_ee_geometry(data: dict) -> None:
    print("=== EE vs cameras at window start ===")
    left = data["left_pose"][0, :3]
    right = data["right_pose"][0, :3]
    for name, camera in data["cameras"].items():
        center = camera["center0"]
        rotation = camera["rotation0"]
        for arm_name, pos in (("left_ee", left), ("right_ee", right)):
            in_cam = rotation.T @ (pos - center)
            print(f"{name:13s} {arm_name}: cam_xyz={in_cam} depth(z)={in_cam[2]:+.3f}")


def main() -> None:
    args = parse_args()
    episode = Path(args.episode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_window(episode, args.start, args.num_frames, args.stride)
    print_camera_ee_geometry(data)

    stem = f"{episode.stem}_start{args.start}"
    write_video(data, args, output_dir / f"{stem}_traj.mp4")
    write_html(data, output_dir / f"{stem}_3d.html", args.axis_length, args.camera_axis_length)
    write_cam_strip(data, output_dir / f"{stem}_cams_rgb.png")


if __name__ == "__main__":
    main()
