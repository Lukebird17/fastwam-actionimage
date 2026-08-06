"""FastWAM policy adapter for RoboTwin closed-loop evaluation."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.robotwin.obs_utils import (
    compose_robotwin_rgb_tensor,
    endpose_dict_to_action_16d,
    render_action_image_tensor,
)
from fastwam.datasets.robotwin.raw_dataset import DEFAULT_PROMPT
from fastwam.datasets.robotwin.wrist_mounts import resolve_wrist_cam_from_ee

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str], cfg: DictConfig) -> Path:
    candidates = []
    if not _is_none_like(dataset_stats_path):
        candidates.append(Path(str(dataset_stats_path)).expanduser())
    stats = cfg.data.train.get("normalization_stats")
    if not _is_none_like(stats):
        candidates.append(Path(str(stats)).expanduser())
    for path in candidates:
        resolved = path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        f"Could not resolve 16D action stats. Tried: {candidates}. "
        "Pass EVALUATION.dataset_stats_path or set data.train.normalization_stats."
    )


def _load_action_stats(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    import json

    with path.open() as file:
        payload = json.load(file)
    mean = torch.tensor(payload["mean"], dtype=torch.float32)
    std = torch.tensor(payload["std"], dtype=torch.float32).clamp_min(1e-6)
    if mean.shape != (16,) or std.shape != (16,):
        raise ValueError(f"Expected 16D mean/std in {path}, got {tuple(mean.shape)}/{tuple(std.shape)}")
    return mean, std


class WorldActionRobotWinPolicy:
    """Closed-loop policy for action-image FastWAM checkpoints (16D ee)."""

    def __init__(
        self,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.action_mean, self.action_std = _load_action_stats(dataset_stats_path)
        self.action_fov_scale = float(data_cfg.train.get("action_fov_scale", 1.0))
        self.action_axis_length = float(data_cfg.train.get("action_axis_length", 0.1))
        self.action_sigma = float(data_cfg.train.get("action_sigma", 0.05))
        self.wrist_look_distance = float(data_cfg.train.get("wrist_look_distance", 0.25))
        self.wrist_cam_from_ee = resolve_wrist_cam_from_ee(
            data_cfg.train.get("wrist_cam_from_ee"),
            embodiment="aloha-agilex",
        )

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized action-image RoboTwin policy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

    def _normalize_action(self, action: np.ndarray) -> torch.Tensor:
        action_t = torch.as_tensor(action, dtype=torch.float32)
        return (action_t - self.action_mean) / self.action_std

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        action = action.detach().to(dtype=torch.float32, device="cpu")
        if action.ndim == 2:
            action = action.unsqueeze(0)
        denorm = action * self.action_std + self.action_mean
        return denorm.numpy()

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        if "endpose" not in observation:
            raise KeyError(
                "Observation is missing `endpose`. Enable data_type.endpose in the RoboTwin task config."
            )
        action_16d = endpose_dict_to_action_16d(observation["endpose"])
        proprio = self._normalize_action(action_16d)

        image_tensor = compose_robotwin_rgb_tensor(observation).to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        action_image = render_action_image_tensor(
            observation,
            action_16d,
            action_fov_scale=self.action_fov_scale,
            action_axis_length=self.action_axis_length,
            action_sigma=self.action_sigma,
            wrist_look_distance=self.wrist_look_distance,
            wrist_cam_from_ee=self.wrist_cam_from_ee,
        ).to(device=self.model.device, dtype=self.model.torch_dtype)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "input_action_image": action_image,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        return self._denormalize_action(pred["action"])[0]

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="ee")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    cfg = _compose_sim_cfg(
        sim_cfg_path=usr_args.get("sim_cfg_path"),
        sim_cfg_name=usr_args.get("sim_cfg_name"),
        sim_task=usr_args.get("sim_task"),
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path") or cfg.EVALUATION.get("dataset_stats_path"),
        cfg=cfg,
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    return WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        data_cfg=cfg.data,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=_parse_optional_int(usr_args.get("seed")),
        text_cfg_scale=float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0))),
        negative_prompt=str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", ""))),
        rand_device=str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu"))),
        tiled=_parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False))),
        timing_enabled=_parse_bool(
            usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
        ),
    )


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
