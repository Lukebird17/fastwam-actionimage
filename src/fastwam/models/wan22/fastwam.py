from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_rgb: float | None = None,
        loss_lambda_action_image: float | None = None,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_rgb = float(
            self.loss_lambda_video if loss_lambda_rgb is None else loss_lambda_rgb
        )
        self.loss_lambda_action_image = float(
            self.loss_lambda_video if loss_lambda_action_image is None else loss_lambda_action_image
        )
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_rgb: float | None = None,
        loss_lambda_action_image: float | None = None,
        loss_lambda_action: float = 1.0,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_rgb=loss_lambda_rgb,
            loss_lambda_action_image=loss_lambda_action_image,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        temporal_factor = int(self.vae.temporal_downsample_factor)
        num_frames = int(video_tensor.shape[2])
        if num_frames % temporal_factor == 1:
            segments = (video_tensor,)
        elif num_frames % 2 == 0 and (num_frames // 2) % temporal_factor == 1:
            segments = video_tensor.chunk(2, dim=2)
        else:
            raise ValueError(
                f"Video must be one valid VAE segment or two equal segments, got T={num_frames} "
                f"with temporal factor {temporal_factor}."
            )
        latents = [
            self.vae.encode(
                segment,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            for segment in segments
        ]
        return torch.cat(latents, dim=2)

    def _conditioning_latent_indices(self, num_latent_frames: int) -> tuple[int, ...]:
        mode = str(getattr(self.video_expert, "video_attention_mask_mode", ""))
        if mode == "segment_first_frame_causal":
            if num_latent_frames % 2:
                raise ValueError(
                    "segment_first_frame_causal requires two equal latent segments, "
                    f"got {num_latent_frames} latent frames."
                )
            return (0, num_latent_frames // 2)
        return (0,)

    def _parse_video_frame_layout(self, num_video_frames: int) -> tuple[bool, int, int]:
        """Return ``(is_dual_segment, segment_frames, latent_frames)``."""
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if num_video_frames % temporal_factor == 1:
            segment_frames = num_video_frames
            return False, segment_frames, (segment_frames - 1) // temporal_factor + 1
        if num_video_frames % 2 == 0 and (num_video_frames // 2) % temporal_factor == 1:
            segment_frames = num_video_frames // 2
            return True, segment_frames, 2 * ((segment_frames - 1) // temporal_factor + 1)
        raise ValueError(
            f"`num_video_frames` must be one valid segment or two equal valid segments, got {num_video_frames}"
        )

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    @torch.no_grad()
    def _encode_conditioning_images(
        self,
        input_image: torch.Tensor,
        input_action_image: Optional[torch.Tensor] = None,
        tiled: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        scene_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        mode = str(getattr(self.video_expert, "video_attention_mask_mode", ""))
        if mode != "segment_first_frame_causal":
            return scene_latents, (0,)
        if input_action_image is None:
            raise ValueError(
                "`input_action_image` is required when `video_attention_mask_mode='segment_first_frame_causal'`."
            )
        action_latents = self._encode_input_image_latents_tensor(
            input_image=input_action_image, tiled=tiled
        )
        return torch.cat([scene_latents, action_latents], dim=2), (0, 1)

    @torch.no_grad()
    def _build_conditioning_latent_grid(
        self,
        input_image: torch.Tensor,
        input_action_image: Optional[torch.Tensor],
        latent_t: int,
        generator: Optional[torch.Generator] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Place the conditioning latents on the training-sized latent grid.

        The conditioning latents must sit at the temporal indices the model was
        trained with, because 3D RoPE encodes a token's frame index into its
        query/key. Training builds `cat(scene_9f, action_9f)`, which the VAE
        encodes per segment into `latent_t=6` with pins at `(0, 3)`. Encoding the
        two conditioning stills into a bare `latent_t=2` tensor instead would put
        the action-image pin at temporal position 1, i.e. a position the action
        expert never saw it at, and silently degrades closed-loop behaviour while
        every offline metric still improves.

        Non-conditioning frames are pure noise, matching the `sigma -> 1` end of
        the flow-matching schedule. They exist only to give the pins their correct
        RoPE positions and to fill out the attention neighbourhood the trained
        mask defines; `patch_size[0] == 1` means no token mixes two latent frames,
        so they cannot bleed into the pinned frames' own content.
        """
        conditioning_latents, _ = self._encode_conditioning_images(
            input_image=input_image,
            input_action_image=input_action_image,
            tiled=tiled,
        )
        conditioning_indices = self._conditioning_latent_indices(latent_t)
        if len(conditioning_indices) != conditioning_latents.shape[2]:
            raise ValueError(
                "Expected one conditioning latent per pinned frame, got "
                f"indices={conditioning_indices} and cond={tuple(conditioning_latents.shape)}"
            )
        latents_video = torch.randn(
            (
                conditioning_latents.shape[0],
                conditioning_latents.shape[1],
                latent_t,
                conditioning_latents.shape[3],
                conditioning_latents.shape[4],
            ),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=conditioning_latents.device, dtype=conditioning_latents.dtype)
        for src, dst in enumerate(conditioning_indices):
            latents_video[:, :, dst : dst + 1] = conditioning_latents[:, :, src : src + 1].clone()
        return latents_video, conditioning_indices

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        mode = str(getattr(self.video_expert, "video_attention_mask_mode", ""))
        if mode == "segment_first_frame_causal" and latents.shape[2] % 2 == 0:
            decoded = [
                self.vae.decode(
                    segment,
                    device=self.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
                for segment in latents.chunk(2, dim=2)
            ]
            video_tensor = torch.cat(decoded, dim=2)
        else:
            video_tensor = self.vae.decode(
                latents,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        temporal_factor = int(self.vae.temporal_downsample_factor)
        is_single_segment = num_frames % temporal_factor == 1
        is_two_segments = num_frames % 2 == 0 and (num_frames // 2) % temporal_factor == 1
        if not (is_single_segment or is_two_segments):
            raise ValueError(
                f"Video must contain one valid segment or two equal valid segments, got T={num_frames}"
            )
        segment_frames = num_frames // 2 if is_two_segments else num_frames
        if segment_frames <= 1:
            raise ValueError(f"Each video segment must contain at least 2 frames, got {segment_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (segment_frames - 1) != 0:
            raise ValueError(
                "`sample['action']` temporal dimension must be divisible by per-segment video "
                f"transitions ({segment_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        conditioning_latents = None
        conditioning_indices: tuple[int, ...] = ()
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            conditioning_indices = self._conditioning_latent_indices(input_latents.shape[2])
            conditioning_latents = input_latents[:, :, conditioning_indices]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "conditioning_latents": conditioning_latents,
            "conditioning_indices": conditioning_indices,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> conditioned video frames only
        num_latent_frames = video_seq_len // video_tokens_per_frame
        for frame_index in self._conditioning_latent_indices(num_latent_frames):
            start = frame_index * video_tokens_per_frame
            mask[video_seq_len:, start : start + video_tokens_per_frame] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        conditioning_indices: tuple[int, ...],
    ) -> list[torch.Tensor]:
        """Return one per-sample video loss tensor per RGB / action-image segment."""
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            if video_loss_token.shape[1] % 2 == 0 and len(conditioning_indices) == 2:
                half = video_loss_token.shape[1] // 2
                return [video_loss_token[:, :half].mean(dim=1), video_loss_token[:, half:].mean(dim=1)]
            return [video_loss_token.mean(dim=1)]

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        num_frames = image_is_pad.shape[1]
        if (num_frames - 1) % temporal_factor == 0:
            segments = (image_is_pad,)
        elif num_frames % 2 == 0 and (num_frames // 2 - 1) % temporal_factor == 0:
            segments = image_is_pad.chunk(2, dim=1)
        else:
            raise ValueError(
                "Cannot align `image_is_pad` with one or two video segments: "
                f"num_frames={num_frames}, temporal_downsample_factor={temporal_factor}."
            )
        latent_pad_segments = []
        for segment in segments:
            latent_tail_is_pad = segment[:, 1:].view(
                segment.shape[0], -1, temporal_factor
            ).all(dim=2)
            latent_pad_segments.append(torch.cat([segment[:, :1], latent_tail_is_pad], dim=1))
        video_is_pad = torch.cat(latent_pad_segments, dim=1)
        if conditioning_indices:
            keep = torch.ones(video_is_pad.shape[1], dtype=torch.bool, device=video_is_pad.device)
            keep[list(conditioning_indices)] = False
            video_is_pad = video_is_pad[:, keep]

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        if len(segments) == 1:
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            return [(video_loss_token * valid).sum(dim=1) / valid_sum]

        if video_loss_token.shape[1] % 2 != 0:
            raise ValueError(
                "Dual-segment video loss expects an even number of supervised latent steps, "
                f"got {video_loss_token.shape[1]}."
            )
        half = video_loss_token.shape[1] // 2
        segment_losses = []
        for start, end in ((0, half), (half, video_loss_token.shape[1])):
            token = video_loss_token[:, start:end]
            segment_valid = valid[:, start:end]
            segment_losses.append((token * segment_valid).sum(dim=1) / segment_valid.sum(dim=1).clamp(min=1.0))
        return segment_losses

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["conditioning_latents"] is not None:
            latents[:, :, inputs["conditioning_indices"]] = inputs["conditioning_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        if inputs["conditioning_indices"]:
            keep = torch.ones(pred_video.shape[2], dtype=torch.bool, device=pred_video.device)
            keep[list(inputs["conditioning_indices"])] = False
            pred_video = pred_video[:, :, keep]
            target_video = target_video[:, :, keep]

        segment_losses = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            conditioning_indices=inputs["conditioning_indices"],
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            segment_losses[0].device, dtype=segment_losses[0].dtype
        )
        weighted_segments = [(loss * video_weight).mean() for loss in segment_losses]
        loss_rgb = weighted_segments[0]
        loss_action_image = weighted_segments[1] if len(weighted_segments) > 1 else None

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_rgb * loss_rgb + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_rgb": self.loss_lambda_rgb * float(loss_rgb.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if loss_action_image is not None:
            loss_total = loss_total + self.loss_lambda_action_image * loss_action_image
            loss_dict["loss_action_image"] = (
                self.loss_lambda_action_image * float(loss_action_image.detach().item())
            )
            loss_dict["loss_video"] = loss_dict["loss_rgb"] + loss_dict["loss_action_image"]
        else:
            loss_dict["loss_video"] = loss_dict["loss_rgb"]
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        input_action_image: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                input_action_image=(
                    input_action_image.clone() if input_action_image is not None else None
                ),
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        is_dual_segment, segment_frames, latent_t = self._parse_video_frame_layout(num_video_frames)
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if input_action_image is not None:
            if input_action_image.ndim == 3:
                input_action_image = input_action_image.unsqueeze(0)
            input_action_image = input_action_image.to(device=self.device, dtype=self.torch_dtype)
        conditioning_latents, _ = self._encode_conditioning_images(
            input_image=input_image,
            input_action_image=input_action_image,
            tiled=tiled,
        )
        if is_dual_segment:
            conditioning_indices = self._conditioning_latent_indices(latent_t)
            if len(conditioning_indices) != conditioning_latents.shape[2]:
                raise ValueError(
                    "Dual-segment inference expects one conditioning latent per segment, "
                    f"got indices={conditioning_indices} and cond={tuple(conditioning_latents.shape)}"
                )
        else:
            conditioning_indices = (0,)
            if conditioning_latents.shape[2] != 1:
                raise ValueError(
                    f"Single-segment inference expects one conditioning latent, got {tuple(conditioning_latents.shape)}"
                )
        for src, dst in enumerate(conditioning_indices):
            latents_video[:, :, dst:dst + 1] = conditioning_latents[:, :, src:src + 1].clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            for src, dst in enumerate(conditioning_indices):
                latents_video[:, :, dst:dst + 1] = conditioning_latents[:, :, src:src + 1].clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        input_action_image: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Infer an action chunk from the segment conditioning frames.

        ``num_video_frames`` sizes the latent grid the conditioning frames are
        pinned into. It is required for ``segment_first_frame_causal``, because
        the action-image pin's temporal index -- and therefore its 3D RoPE
        position -- is defined by the training grid, not by the number of
        conditioning images.
        """
        self.eval()
        mode = str(getattr(self.video_expert, "video_attention_mask_mode", ""))
        if mode not in {"first_frame_causal", "segment_first_frame_causal"}:
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode` in "
                "{'first_frame_causal', 'segment_first_frame_causal'}."
            )
        if mode == "segment_first_frame_causal" and num_video_frames is None:
            raise ValueError(
                "`num_video_frames` is required when "
                "`video_attention_mask_mode='segment_first_frame_causal'`, because it "
                "determines the latent index the action image is pinned at."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if input_action_image is not None:
            if input_action_image.ndim == 3:
                input_action_image = input_action_image.unsqueeze(0)
            input_action_image = input_action_image.to(device=self.device, dtype=self.torch_dtype)
        if mode == "segment_first_frame_causal":
            # Separate generator so the action noise above stays bit-identical to
            # the single-stream behaviour, mirroring `infer_joint`.
            video_generator = (
                None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
            )
            _, _, latent_t = self._parse_video_frame_layout(int(num_video_frames))
            conditioning_latents, _ = self._build_conditioning_latent_grid(
                input_image=input_image,
                input_action_image=input_action_image,
                latent_t=latent_t,
                generator=video_generator,
                rand_device=rand_device,
                tiled=tiled,
            )
        else:
            # `first_frame_causal` needs no grid: its single pin is at temporal
            # index 0 in training as well, and `build_video_to_video_mask` makes
            # the first frame attend only to itself, so a bare 1-frame latent is
            # exactly equivalent to the full training grid. Building the grid
            # anyway would triple the prefill cost for an identical result.
            conditioning_latents, _ = self._encode_conditioning_images(
                input_image=input_image,
                input_action_image=input_action_image,
                tiled=tiled,
            )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        # Under `segment_first_frame_causal` the grid's non-pinned frames hold pure
        # noise, so they must be declared at the fully-noisy end of the schedule
        # rather than as clean. `pre_dit` overwrites the pinned frames' per-token
        # timestep with 0, so the pins are still announced as clean. The
        # single-pin mode has no filler frames, so it keeps its exact previous
        # value and stays bit-identical.
        timestep_video = torch.full(
            (conditioning_latents.shape[0],),
            (
                float(self.infer_video_scheduler.num_train_timesteps)
                if mode == "segment_first_frame_causal"
                else 0.0
            ),
            dtype=conditioning_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=conditioning_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        input_action_image: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            input_action_image=input_action_image,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
