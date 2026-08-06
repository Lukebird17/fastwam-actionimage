"""Validate the two poker-experiment configs without touching a GPU.

Checks that (a) hydra composes both task configs, (b) the shared dataset yields
the shapes each arm's model expects, (c) the two arms differ ONLY in the
action-image half -- same episodes, same window count, same 16D action space --
and (d) the two data configs are still a one-line diff, which is what catches
the arms silently drifting apart when the shared tree is re-synced.

Run after any change to the RoboTwin dataset or to either arm's config, and
always after scripts/sync_from_upstream_worktree.sh:
    python scripts/check_poker_configs.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
os.chdir(Path(__file__).resolve().parent.parent)

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = str(Path("configs").resolve())
VAE_TEMPORAL_FACTOR = 4
ARMS = ("robotwin_poker_scene_only_1x4", "robotwin_poker_action_image_1x4")
# The control data config must stay the treatment config plus this one key. A
# `cp -a` fork goes stale every time the shared tree advances, and a renderer
# calibration landing on only one side would make the arms incomparable while
# both configs still look individually valid.
CONTROL_ONLY_KEYS = {"include_action_video"}


def check_data_configs_are_one_line_apart() -> None:
    treatment = OmegaConf.load("configs/data/robotwin_action_image.yaml")
    control = OmegaConf.load("configs/data/robotwin_scene_only.yaml")
    print("\n=== control vs treatment data config ===")
    for split in ("train", "val"):
        t = OmegaConf.to_container(treatment[split])
        c = OmegaConf.to_container(control[split])
        extra = set(c) - set(t)
        missing = set(t) - set(c)
        differing = {k for k in set(t) & set(c) if t[k] != c[k]}
        print(f"  {split}: control-only={sorted(extra)} missing={sorted(missing)} "
              f"differing={sorted(differing)}")
        assert extra == CONTROL_ONLY_KEYS, f"{split}: unexpected control-only keys {sorted(extra)}"
        assert not missing, f"{split}: control is missing {sorted(missing)} -- arms have drifted"
        assert not differing, f"{split}: shared keys differ {sorted(differing)} -- arms have drifted"
    print("  OK: the action-image switch is the only difference")


check_data_configs_are_one_line_apart()


def load(task: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="train", overrides=[f"task={task}"])


summaries = {}
for task in ARMS:
    cfg = load(task)
    print(f"\n=== {task} ===")
    print("  mask_mode      :", cfg.model.video_dit_config.video_attention_mask_mode)
    print("  fuse_vae       :", cfg.model.video_dit_config.fuse_vae_embedding_in_latents)
    print("  sep_timestep   :", cfg.model.video_dit_config.seperated_timestep)
    print("  loss           :", OmegaConf.to_container(cfg.model.loss))
    print("  tasks          :", OmegaConf.to_container(cfg.data.train.tasks))
    print("  action_fov     :", cfg.data.train.action_fov_scale)
    print("  batch/accum/lr :", cfg.batch_size, cfg.gradient_accumulation_steps, cfg.learning_rate)
    print("  incl_action_vid:", cfg.data.train.get("include_action_video", True))

    ds = hydra.utils.instantiate(cfg.data.train)
    print("  episodes       :", len(ds.episodes), " windows:", len(ds))
    sample = ds[0]
    for key in ("video", "action", "proprio", "image_is_pad", "action_is_pad", "context"):
        value = sample[key]
        print(f"  {key:14s}:", tuple(value.shape) if torch.is_tensor(value) else type(value).__name__)

    num_frames = sample["video"].shape[1]
    is_dual = num_frames % 2 == 0 and (num_frames // 2) % VAE_TEMPORAL_FACTOR == 1
    is_single = num_frames % VAE_TEMPORAL_FACTOR == 1
    if not (is_single or is_dual):
        raise AssertionError(f"T={num_frames} is neither one nor two valid VAE segments")
    segment_frames = num_frames // 2 if is_dual else num_frames
    latent_t = (2 if is_dual else 1) * ((segment_frames - 1) // VAE_TEMPORAL_FACTOR + 1)
    dual_mode = cfg.model.video_dit_config.video_attention_mask_mode == "segment_first_frame_causal"
    conditioning = (0, latent_t // 2) if dual_mode else (0,)
    print(f"  layout         : T={num_frames} dual={is_dual} seg={segment_frames} latent_t={latent_t}")
    print(f"  cond_indices   : {conditioning}  supervised_latent_steps={latent_t - len(conditioning)}")

    # The model derives its segment layout from T, so a mask mode that disagrees
    # with the dataset's video shape fails deep inside the DiT instead of here.
    assert dual_mode == is_dual, f"mask mode dual={dual_mode} but video dual={is_dual}"
    assert sample["action"].shape[0] % (segment_frames - 1) == 0, "action horizon indivisible"
    assert sample["image_is_pad"].shape[0] == num_frames, "image_is_pad does not match T"
    assert sample["video"].shape[2] % 16 == 0 and sample["video"].shape[3] % 16 == 0
    has_action_image_loss = "lambda_action_image" in OmegaConf.to_container(cfg.model.loss)
    assert has_action_image_loss == is_dual, "lambda_action_image present without a second segment"
    summaries[task] = {
        "episodes": len(ds.episodes),
        "windows": len(ds),
        "action_shape": tuple(sample["action"].shape),
        "effective_batch": cfg.batch_size * cfg.gradient_accumulation_steps,
        "lr": cfg.learning_rate,
    }
    print("  OK")

# The comparison is only interpretable if the two arms see identical data.
control, treatment = (summaries[arm] for arm in ARMS)
print("\n=== arms comparable? ===")
for field in ("episodes", "windows", "action_shape", "effective_batch", "lr"):
    same = control[field] == treatment[field]
    print(f"  {field:16s}: {control[field]} vs {treatment[field]}  {'OK' if same else 'MISMATCH'}")
    assert same, f"{field} differs between arms: {control[field]} vs {treatment[field]}"
print("  both arms share the data pipeline; action image is the only variable")
