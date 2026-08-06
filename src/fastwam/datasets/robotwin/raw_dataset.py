"""Direct RoboTwin2.0 HDF5 dataset with first-frame action-image projection."""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch

from .action_image import render_action_images_for_camera
from .conditioning import (
    DEFAULT_PROMPT,
    format_robotwin_prompt,
    load_action_stats,
    load_text_embedding,
    normalize_action_16d,
)
from .obs_utils import (
    CAMERA_NAMES,
    ActionImageRenderConfig,
    compose_robotwin_views,
    decode_jpeg,
    load_action_16d_from_hdf5,
)
from .wrist_mounts import resolve_action_image_geometry


ALOHA_EMBODIMENTS = ("aloha-agilex_clean_50", "aloha-agilex_randomized_500")
# Re-export for older imports; preferred public location is ``obs_utils.CAMERA_NAMES``.
__all__ = ["RoboTwinActionImageDataset", "CAMERA_NAMES", "DEFAULT_PROMPT", "ALOHA_EMBODIMENTS"]


@dataclass(frozen=True)
class Episode:
    path: Path
    task: str
    embodiment: str
    episode_id: str
    length: int
    num_windows: int


# Backward-compatible alias.
_decode_jpeg = decode_jpeg


class RoboTwinActionImageDataset(torch.utils.data.Dataset):
    """Read raw RoboTwin2.0 episodes without converting them to LeRobot.

    ``video`` is time-concatenated as ``[observation video, action-image video]``.
    Each half uses the same 384x320 head/left-wrist/right-wrist spatial layout.
    With ``include_action_video=False`` only the observation half is returned, which
    is the pure-FastWAM ablation of this exact dataset.
    Action supervision is 16D absolute bimanual end-effector pose:
    ``[left xyz+quat(wxyz)+gripper, right xyz+quat(wxyz)+gripper]``.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        num_frames: int = 33,
        action_video_freq_ratio: int = 4,
        global_sample_stride: int = 1,
        window_stride: int = 1,
        include_padded_tail: bool = True,
        tasks: Sequence[str] | None = None,
        embodiments: Sequence[str] | None = None,
        split: str = "all",
        val_set_proportion: float = 0.01,
        split_seed: int = 0,
        instruction_set: str = "seen",
        instruction_index: int = 0,
        override_instruction: str | None = None,
        action_fov_scale: float = 1.0,
        action_axis_length: float = 0.1,
        action_sigma: float = 0.05,
        wrist_look_distance: float = 0.25,
        geometry_embodiment: str | None = None,
        wrist_cam_from_ee: dict | None = None,
        ee_from_action_frame: dict | None = None,
        include_action_video: bool = True,
        return_video_components: bool = False,
        normalization_stats: str | Path | None = None,
        pretrained_norm_stats: str | Path | None = None,
        text_embedding_cache_dir: str | Path | None = None,
        context_len: int = 128,
        action_dim: int = 16,
        proprio_dim: int = 16,
    ) -> None:
        if action_dim != 16 or proprio_dim != 16:
            raise ValueError("RoboTwinActionImageDataset uses fixed 16D action and proprio layouts.")
        self.dataset_root = Path(dataset_root).expanduser()
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        self.global_sample_stride = global_sample_stride
        self.window_stride = window_stride
        self.include_padded_tail = include_padded_tail
        self.instruction_set = instruction_set
        self.instruction_index = instruction_index
        self.override_instruction = override_instruction
        self.render_cfg = ActionImageRenderConfig(
            fov_scale=action_fov_scale,
            axis_length=action_axis_length,
            sigma=action_sigma,
            wrist_look_distance=wrist_look_distance,
        )
        # Resolve per episode in __getitem__. This is essential for datasets that
        # mix embodiments; an explicit geometry_embodiment intentionally pins all
        # episodes to one calibration family.
        self.geometry_embodiment = geometry_embodiment
        self.wrist_cam_from_ee_override = wrist_cam_from_ee
        self.ee_from_action_frame_override = ee_from_action_frame
        # `include_action_video=False` yields the observation video alone, so a pure
        # FastWAM ablation shares this dataset -- same episodes, same window
        # indexing, same 16D action space, same normalization stats, same text
        # cache. Going through `RobotVideoDataset` instead would change the action
        # space (14D joint) and the data source at the same time as removing the
        # action image, so nothing could be attributed to the action image itself.
        self.include_action_video = include_action_video
        self.return_video_components = return_video_components
        self.text_embedding_cache_dir = (
            None if text_embedding_cache_dir is None else Path(text_embedding_cache_dir).expanduser()
        )
        self.context_len = context_len
        self.action_mean = None
        self.action_std = None
        stats_path = normalization_stats or pretrained_norm_stats
        if stats_path is not None:
            self.action_mean, self.action_std = load_action_stats(stats_path)

        if split not in {"all", "train", "val"}:
            raise ValueError(f"split must be one of all/train/val, got {split}")
        if not 0 <= val_set_proportion < 1:
            raise ValueError(f"val_set_proportion must be in [0, 1), got {val_set_proportion}")
        if min(num_frames, global_sample_stride, window_stride) <= 0:
            raise ValueError("num_frames and strides must be positive")
        if (num_frames - 1) % action_video_freq_ratio:
            raise ValueError("num_frames - 1 must be divisible by action_video_freq_ratio")

        self.video_sample_indices = np.arange(0, num_frames, action_video_freq_ratio)
        task_filter = set(tasks) if tasks else None
        embodiment_filter = set(embodiments or ALOHA_EMBODIMENTS)
        self.episodes = self._index_episodes(
            task_filter,
            embodiment_filter,
            split,
            val_set_proportion,
            split_seed,
        )
        self._cumulative_windows = np.cumsum([episode.num_windows for episode in self.episodes]).tolist()
        if not self.episodes:
            raise FileNotFoundError(
                f"No matching extracted HDF5 episodes under {self.dataset_root}. "
                "Expected {task}/{embodiment}/data/episode*.hdf5."
            )

    def _index_episodes(
        self,
        tasks: set[str] | None,
        embodiments: set[str] | None,
        split: str,
        val_fraction: float,
        split_seed: int,
    ) -> list[Episode]:
        episodes = []
        span = (self.num_frames - 1) * self.global_sample_stride + 1
        for path in sorted(self.dataset_root.glob("*/*/data/episode*.hdf5")):
            task, embodiment = path.parents[2].name, path.parents[1].name
            if tasks is not None and task not in tasks:
                continue
            if embodiments is not None and embodiment not in embodiments:
                continue
            digest = hashlib.sha256(f"{split_seed}:{path.relative_to(self.dataset_root)}".encode()).digest()
            is_val = int.from_bytes(digest[:8], "big") / 2**64 < val_fraction
            if split == "train" and is_val or split == "val" and not is_val:
                continue
            with h5py.File(path, "r") as episode_file:
                length = int(episode_file["endpose/left_endpose"].shape[0])
            if self.include_padded_tail:
                num_windows = (length + self.window_stride - 1) // self.window_stride
            else:
                num_windows = max(0, (length - span) // self.window_stride + 1)
            if num_windows:
                episodes.append(
                    Episode(path, task, embodiment, path.stem, length, num_windows)
                )
        return episodes

    def __len__(self) -> int:
        return self._cumulative_windows[-1]

    def _locate(self, index: int) -> tuple[Episode, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self._cumulative_windows, index)
        previous = self._cumulative_windows[episode_index - 1] if episode_index else 0
        start = (index - previous) * self.window_stride
        return self.episodes[episode_index], start

    def _instruction(self, episode: Episode) -> str:
        if self.override_instruction is not None:
            task = self.override_instruction
        else:
            path = episode.path.parents[1] / "instructions" / f"{episode.episode_id}.json"
            with path.open() as file:
                instructions = json.load(file)
            choices = instructions[self.instruction_set]
            task = choices[self.instruction_index % len(choices)]
        return format_robotwin_prompt(task)

    def _text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not configured.")
        return load_text_embedding(
            prompt,
            self.text_embedding_cache_dir,
            context_len=self.context_len,
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        episode, start = self._locate(index)
        raw_indices = start + np.arange(self.num_frames) * self.global_sample_stride
        indices = np.minimum(raw_indices, episode.length - 1)
        frame_is_pad = raw_indices >= episode.length
        video_indices = indices[self.video_sample_indices]

        scene_views = []
        action_views = []
        geometry_embodiment = self.geometry_embodiment or episode.embodiment
        geometry = resolve_action_image_geometry(
            geometry_embodiment,
            wrist_cam_from_ee=self.wrist_cam_from_ee_override,
            ee_from_action_frame=self.ee_from_action_frame_override,
        )
        with h5py.File(episode.path, "r") as episode_file:
            all_actions = load_action_16d_from_hdf5(episode_file, indices)
            rendered_actions = all_actions[self.video_sample_indices]
            observations = episode_file["observation"]
            for camera_name in CAMERA_NAMES:
                camera = observations[camera_name]
                rgb = np.stack([decode_jpeg(camera["rgb"][frame]) for frame in video_indices])
                height, width = rgb.shape[1:3]
                scene_views.append(torch.from_numpy(rgb).permute(0, 3, 1, 2))
                if not self.include_action_video:
                    continue
                action_rgb = render_action_images_for_camera(
                    camera_name,
                    rendered_actions,
                    camera["extrinsic_cv"][indices[0]],
                    camera["intrinsic_cv"][indices[0]],
                    height,
                    width,
                    fov_scale=self.render_cfg.fov_scale,
                    axis_length=self.render_cfg.axis_length,
                    sigma=self.render_cfg.sigma,
                    wrist_look_distance=self.render_cfg.wrist_look_distance,
                    wrist_cam_from_ee=geometry.wrist_cam_from_ee,
                    ee_from_action_frame=geometry.ee_from_action_frame,
                )
                action_views.append(torch.from_numpy(action_rgb).permute(0, 3, 1, 2))

        scene_video = compose_robotwin_views(torch.stack(scene_views)).float() / 255
        scene_video = scene_video.mul(2).sub(1).permute(1, 0, 2, 3)
        action_video = None
        if self.include_action_video:
            action_video = compose_robotwin_views(torch.stack(action_views)).float()
            action_video = action_video.mul(2).sub(1).permute(1, 0, 2, 3)
            video = torch.cat((scene_video, action_video), dim=1)
        else:
            video = scene_video

        video_is_pad = torch.from_numpy(frame_is_pad[self.video_sample_indices]).bool()
        action_is_pad = torch.from_numpy(frame_is_pad[1:]).bool()
        prompt = self._instruction(episode)
        actions = torch.from_numpy(all_actions)
        if self.action_mean is not None and self.action_std is not None:
            actions = normalize_action_16d(actions, self.action_mean, self.action_std)
        sample = {
            "video": video,
            "action": actions[1:],
            "proprio": actions[:-1],
            "prompt": prompt,
            # One entry per frame of `video`, so it must follow the same
            # single/dual-segment layout the action-image switch selects.
            "image_is_pad": (
                torch.cat((video_is_pad, video_is_pad))
                if self.include_action_video
                else video_is_pad
            ),
            "action_is_pad": action_is_pad,
            "proprio_is_pad": torch.from_numpy(frame_is_pad[:-1]).bool(),
            "episode_path": str(episode.path),
            "window_start": start,
            "camera_names": CAMERA_NAMES,
        }
        if self.return_video_components:
            sample["observation_video"] = scene_video
            if action_video is not None:
                sample["action_video"] = action_video
        if self.text_embedding_cache_dir is not None:
            sample["context"], sample["context_mask"] = self._text_context(prompt)
        return sample
