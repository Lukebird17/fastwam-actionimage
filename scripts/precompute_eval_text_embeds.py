"""Precompute text embeddings for every instruction a RoboTwin eval can sample.

`scripts/precompute_text_embeds.py` caches the instructions found in the
dataset's `instructions/episode*.json` files. That is the right set for
training, but it is NOT the set eval draws from. At eval time
`script/eval_policy.py` calls `generate_episode_descriptions(...)` on the live
episode info and samples `np.random.choice(results[0][instruction_type])`, and
that generator fills each `{A}`-style placeholder with a RANDOMLY chosen entry
from the object's description file. So eval can produce a template/description
combination that no recorded episode happened to use, and the policy then dies
with `Missing text embedding cache ...` mid-rollout.

The reachable set is closed and small, so enumerate it exhaustively instead of
sampling it: for every episode info dict the task can emit, take every template
whose placeholders match, and expand the cartesian product of all description
choices. For `move_playingcard_away` that is 50 templates x 3 card models x 12
descriptions x 2 arms, i.e. a few thousand prompts -- cheaper to encode once
than to debug a rollout that dies at episode 37.

The episode info dicts are read from the recorded `scene_info.json` rather than
hardcoded, and the templates come from RoboTwin's own
`filter_instructions`/`extract_placeholders`, so this stays in sync with the
generator it has to agree with.

Usage:
  python scripts/precompute_eval_text_embeds.py \
      --config-name sim_robotwin task=robotwin_poker_scene_only_1x4

  # enumerate only, no GPU / no text encoder -- prints the prompt count
  python scripts/precompute_eval_text_embeds.py \
      --config-name sim_robotwin task=robotwin_poker_scene_only_1x4 dry_run=true
"""

import hashlib
import itertools
import json
import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from fastwam.datasets.robotwin.conditioning import format_robotwin_prompt
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_BATCH_SIZE = 16


def _import_robotwin_generator(robotwin_root: Path):
    """Import RoboTwin's instruction generator so we expand exactly as eval does."""
    utils_dir = robotwin_root / "description" / "utils"
    if not (utils_dir / "generate_episode_instructions.py").exists():
        raise FileNotFoundError(f"Missing RoboTwin instruction generator under {utils_dir}")
    sys.path.insert(0, str(utils_dir))
    import generate_episode_instructions as gen  # noqa: E402

    return gen


def _episode_infos(dataset_root: Path, task: str, embodiments: list[str]) -> list[dict]:
    """Every distinct episode info dict the task emitted in the recorded data.

    Eval builds its info dict from the live `play_once()`, so in principle it is
    whatever the env can produce. The recorded runs are the only ground truth we
    have for that space; for poker the 500-episode randomized set covers all
    card-model x arm combinations many times over.
    """
    infos: list[dict] = []
    seen: set[str] = set()
    for embodiment in embodiments:
        scene_info_path = dataset_root / task / embodiment / "scene_info.json"
        if not scene_info_path.exists():
            logger.warning("No scene_info.json for %s/%s; skipping.", task, embodiment)
            continue
        with scene_info_path.open() as file:
            scene_info = json.load(file)
        for episode_data in scene_info.values():
            info = episode_data.get("info", {}) if isinstance(episode_data, dict) else {}
            key = json.dumps(info, sort_keys=True)
            if key not in seen:
                seen.add(key)
                infos.append(info)
    if not infos:
        raise ValueError(f"Found no episode info dicts for task {task} in {dataset_root}")
    logger.info("Task %s emits %d distinct episode info dicts.", task, len(infos))
    return infos


def _placeholder_choices(gen, value: str, instruction_types: tuple[str, ...]) -> list[str]:
    """All strings a single placeholder value can expand to.

    Mirrors `replace_placeholders` / `replace_placeholders_unseen`: a value that
    names a description JSON expands to every entry in it, an arm value becomes
    "the <arm> arm", anything else is literal.
    """
    desc_path = (
        Path(gen.parent_directory) / ".." / "objects_description" / f"{value}.json"
    )
    if desc_path.exists():
        with desc_path.open() as file:
            payload = json.load(file)
        choices = []
        for instruction_type in instruction_types:
            choices.extend(f"the {desc}" for desc in payload.get(instruction_type, []))
        return list(dict.fromkeys(choices))
    return [value]


def _expand_template(gen, template: str, info: dict, instruction_types: tuple[str, ...]) -> list[str]:
    """Every string `template` can become under `info`, over all description choices."""
    stripped = {key.strip("{}"): value for key, value in info.items()}
    present = [key for key in gen.extract_placeholders(template) if key in stripped]

    per_key: list[list[str]] = []
    for key in present:
        value = str(stripped[key])
        if len(key) == 1 and "a" <= key <= "z":
            per_key.append([f"the {value} arm"])
        else:
            per_key.append(_placeholder_choices(gen, value, instruction_types))

    expanded = []
    for combo in itertools.product(*per_key) if per_key else [()]:
        text = template
        for key, replacement in zip(present, combo):
            text = text.replace("{" + key + "}", replacement)
        expanded.append(text)
    return expanded


def _enumerate_eval_prompts(
    gen,
    task: str,
    episode_infos: list[dict],
    instruction_types: tuple[str, ...],
) -> list[str]:
    task_data = gen.load_task_instructions(task)
    prompts: list[str] = []
    seen: set[str] = set()
    for info in episode_infos:
        for instruction_type in instruction_types:
            templates = task_data.get(instruction_type, [])
            # filter_instructions shuffles in place; it is only used here to
            # decide which templates this info dict can fill.
            for template in gen.filter_instructions(list(templates), info):
                for text in _expand_template(gen, template, info, instruction_types):
                    prompt = format_robotwin_prompt(text)
                    if prompt not in seen:
                        seen.add(prompt)
                        prompts.append(prompt)
    logger.info("Enumerated %d unique eval prompts for %s.", len(prompts), task)
    return prompts


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{os.getpid()}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


@hydra.main(config_path="../configs", config_name="sim_robotwin", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    evaluation = cfg.EVALUATION
    robotwin_root = Path(str(evaluation.robotwin_root)).expanduser().resolve()
    task_name = str(evaluation.task_name)
    dry_run = bool(cfg.get("dry_run", False))

    data_cfg = cfg.data
    cache_dir = Path(str(data_cfg.train.text_embedding_cache_dir)).expanduser()
    context_len = int(data_cfg.train.context_len)
    dataset_root = Path(str(data_cfg.train.dataset_root)).expanduser()
    embodiments = [str(name) for name in data_cfg.train.embodiments]

    # `instruction_type` in deploy_policy.yml is what eval samples; cache the
    # other type too, because it costs little and a config flip should not
    # crash a 12-hour rollout.
    instruction_types = ("seen", "unseen")

    gen = _import_robotwin_generator(robotwin_root)
    episode_infos = _episode_infos(dataset_root, task_name, embodiments)
    prompts = _enumerate_eval_prompts(gen, task_name, episode_infos, instruction_types)

    filenames = {
        prompt: f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}.t5_len{context_len}.wan22ti2v5b.pt"
        for prompt in prompts
    }
    missing = [prompt for prompt in prompts if not (cache_dir / filenames[prompt]).exists()]
    logger.info(
        "Cache %s: %d/%d eval prompts already cached, %d missing.",
        cache_dir,
        len(prompts) - len(missing),
        len(prompts),
        len(missing),
    )

    if dry_run:
        for prompt in prompts[:3]:
            logger.info("sample prompt: %s", prompt)
        return
    if not missing:
        logger.info("Nothing to encode.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = cfg.model
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID))
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=bool(model_cfg.get("redirect_common_files", True)),
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path, seq_len=context_len, clean="whitespace"
    )

    over_length = 0
    with torch.no_grad():
        for start in tqdm(
            range(0, len(missing), DEFAULT_BATCH_SIZE), desc="Encoding", unit="batch"
        ):
            batch = missing[start : start + DEFAULT_BATCH_SIZE]
            ids, mask = tokenizer(batch, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device=device, dtype=torch.bool)
            over_length += int(mask.all(dim=1).sum().item())
            context = text_encoder(ids, mask)
            for i, prompt in enumerate(batch):
                _atomic_torch_save(
                    {
                        "context": context[i].detach().to("cpu", torch.bfloat16).contiguous(),
                        "mask": mask[i].detach().to("cpu", torch.bool).contiguous(),
                    },
                    cache_dir / filenames[prompt],
                )

    logger.info("Encoded %d prompts into %s.", len(missing), cache_dir)
    logger.info("Over-length prompts (no padding left at len=%d): %d", context_len, over_length)


if __name__ == "__main__":
    main()
