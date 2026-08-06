"""Shared RoboTwin action/text conditioning utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)


def format_robotwin_prompt(instruction: str) -> str:
    """Wrap a raw task instruction with the shared training/eval prompt template."""
    return DEFAULT_PROMPT.format(task=instruction)


def load_action_stats(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and validate the shared 16D RoboTwin z-score statistics."""
    stats_path = Path(path).expanduser()
    with stats_path.open() as file:
        payload = json.load(file)
    mean = torch.tensor(payload["mean"], dtype=torch.float32)
    std = torch.tensor(payload["std"], dtype=torch.float32).clamp_min(1e-6)
    if mean.shape != (16,) or std.shape != (16,):
        raise ValueError(
            f"Expected 16D mean/std in {stats_path}, got {tuple(mean.shape)}/{tuple(std.shape)}"
        )
    return mean, std


def normalize_action_16d(
    action: np.ndarray | torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    action_tensor = torch.as_tensor(action, dtype=torch.float32)
    return (action_tensor - mean) / std


def denormalize_action_16d(
    action: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return action.detach().to(dtype=torch.float32, device="cpu") * std + mean


def load_text_embedding(
    prompt: str,
    cache_dir: str | Path,
    *,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the exact precomputed text conditioning used during training."""
    cache_path = Path(cache_dir).expanduser()
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    path = cache_path / f"{digest}.t5_len{context_len}.wan22ti2v5b.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing text embedding cache {path}. "
            "Run scripts/precompute_text_embeds.py for the evaluation instruction set."
        )
    payload = torch.load(path, map_location="cpu")
    context = payload["context"].clone()
    mask = payload["mask"].bool()
    if context.ndim != 2 or context.shape[0] != context_len:
        raise ValueError(f"Invalid cached context shape {tuple(context.shape)} in {path}")
    if mask.shape != (context_len,):
        raise ValueError(f"Invalid cached context mask shape {tuple(mask.shape)} in {path}")
    context[~mask] = 0
    # Match training dataset semantics: zero padded embeddings, then expose the
    # fixed-length context as fully valid to the model.
    return context, torch.ones_like(mask)
