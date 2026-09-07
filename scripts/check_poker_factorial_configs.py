"""Validate token-matched action-image factorial configs before submission."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
os.chdir(Path(__file__).resolve().parent.parent)

import hydra
import torch
from hydra import compose, initialize_config_dir


CONFIG_DIR = str(Path("configs").resolve())
SUITES = {
    "poker": {
        "robotwin_poker_ai_token_control_1x4": (True, 0.0),
        "robotwin_poker_ai_input_only_1x4": (False, 0.0),
        "robotwin_poker_ai_target_only_1x4": (True, 0.05),
        "robotwin_poker_action_image_1x4": (False, 0.05),
    },
    "place_empty_cup": {
        "robotwin_place_empty_cup_ai_token_control_1x4": (True, 0.0),
        "robotwin_place_empty_cup_ai_input_only_1x4": (False, 0.0),
        "robotwin_place_empty_cup_ai_target_only_1x4": (True, 0.05),
        "robotwin_place_empty_cup_action_image_1x4": (False, 0.05),
    },
}


def load(task: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="train", overrides=[f"task={task}"])


def validate_suite(name: str, arms: dict[str, tuple[bool, float]]) -> None:
    reference = None
    for task, (expected_blank, expected_loss) in arms.items():
        cfg = load(task)
        ds = hydra.utils.instantiate(cfg.data.train)
        sample = ds[0]
        segment_frames = sample["video"].shape[1] // 2
        action_condition = sample["video"][:, segment_frames]
        blank = bool(torch.all(action_condition == -1.0))
        loss = float(cfg.model.loss.lambda_action_image)
        signature = {
            "episodes": len(ds.episodes),
            "windows": len(ds),
            "video_shape": tuple(sample["video"].shape),
            "action_shape": tuple(sample["action"].shape),
            "effective_batch": int(cfg.batch_size * cfg.gradient_accumulation_steps),
            "learning_rate": float(cfg.learning_rate),
        }
        print(f"{task}: blank={blank} target_loss={loss} signature={signature}")
        assert blank == expected_blank
        assert loss == expected_loss
        assert cfg.model.video_dit_config.video_attention_mask_mode == "segment_first_frame_causal"
        if reference is None:
            reference = signature
        else:
            assert signature == reference, f"{name} arms drifted: {signature} != {reference}"
    print(f"OK: {name} arms are token/data/compute matched; only input and target factors vary")


parser = argparse.ArgumentParser()
parser.add_argument("--suite", choices=["all", *SUITES], default="all")
args = parser.parse_args()
selected = SUITES if args.suite == "all" else {args.suite: SUITES[args.suite]}
for suite_name, suite_arms in selected.items():
    validate_suite(suite_name, suite_arms)
