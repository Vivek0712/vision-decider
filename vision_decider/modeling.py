"""VisionDeciderModel: Strands Decider's pointer readout on the full Qwen3.5 torso.

The text Decider loads `Qwen3_5ForCausalLM(...).model`, deliberately dropping the vision
tower that ships inside every Qwen3.5 checkpoint. This class loads
`Qwen3_5ForConditionalGeneration(...).model` instead -- the `Qwen3_5Model` wrapper that
holds `visual` (ViT + patch merger) and `language_model` (the same 24-layer hybrid
decoder) -- and forwards `pixel_values` / `image_grid_thw` through it. The decoder's
last hidden state comes back in the same shape as before, so `gather_options`,
`PointerHead`, `masked_log_softmax`, calibration and every confidence formula are
inherited from `StrandsDeciderModel` untouched.

What is trained: the LoRA adapter on the language model's projections (v19's target
list) and the fp32 pointer head. The vision tower is frozen in the reference recipe;
`lora_on_merger=True` additionally adapts the patch merger's two linear layers, the
cheapest place to change how image tokens enter the decoder.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import torch
from torch import nn

from strands_decider.modeling import (
    StrandsDeciderConfig,
    StrandsDeciderModel,
    build_head,
    checkpoint_dir,
    config_path,
    load_head_state,
)

VISION_CONFIG_NAME = "vision_decider.json"
# Linear layers of Qwen3.5's patch merger (`visual.merger.linear_fc1/2`).
MERGER_TARGETS = ("linear_fc1", "linear_fc2")


def mm_token_type_ids(torso: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """0 for text, 1 for image tokens: what Qwen3VLProcessor returns beside input_ids.

    transformers 5.18 refuses an image forward without it, because M-RoPE positions are
    computed from it.
    """
    return (input_ids == torso.config.image_token_id).to(torch.int32)


def qwen_base(torso: nn.Module) -> nn.Module:
    """The `Qwen3_5Model` under any PEFT wrapping (PeftModel -> LoraModel -> model)."""
    base = getattr(torso, "base_model", torso)
    return getattr(base, "model", base)


@dataclass
class VisionDeciderConfig(StrandsDeciderConfig):
    base_model: str = "Qwen/Qwen3.5-2B-Base"
    head_type: str = "pointer"
    # Longest image side after resizing; the processor's dynamic resolution then yields
    # a bounded token count (448 square -> 196 tokens at patch 16, merge 2). Fixed per
    # deployment so the window budget is predictable.
    image_long_side: int = 448
    # Upper bound on tokens one image may expand to; rows over it are dropped, never
    # truncated (the Decider rule: drop, never truncate through an option list).
    # Very wide or tall images can exceed the long side: the processor upscales to
    # its 65,536-pixel minimum.
    max_image_tokens: int = 640
    # Whether to attach LoRA to the vision patch merger as well as the decoder.
    lora_on_merger: bool = False
    lora_targets: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
        ]
    )

    def base_dict(self) -> dict[str, Any]:
        """The fields `StrandsDeciderConfig` knows, so upstream tools can read the file."""
        names = {f.name for f in fields(StrandsDeciderConfig)}
        return {k: v for k, v in asdict(self).items() if k in names}

    def vision_dict(self) -> dict[str, Any]:
        names = {f.name for f in fields(StrandsDeciderConfig)}
        return {k: v for k, v in asdict(self).items() if k not in names}

    def write(self, path: str) -> None:
        """Base fields to the upstream config file, vision fields beside it."""
        with open(config_path(path), "w", encoding="utf-8") as fh:
            json.dump(self.base_dict(), fh, indent=2)
        with open(os.path.join(path, VISION_CONFIG_NAME), "w", encoding="utf-8") as fh:
            json.dump(self.vision_dict(), fh, indent=2)

    @classmethod
    def read(cls, path: str) -> VisionDeciderConfig:
        with open(config_path(path), encoding="utf-8") as fh:
            d = json.load(fh)
        extra = os.path.join(path, VISION_CONFIG_NAME)
        if os.path.exists(extra):
            with open(extra, encoding="utf-8") as fh:
                d.update(json.load(fh))
        return cls(**d)


class VisionDeciderModel(StrandsDeciderModel):
    """Image-aware torso; readout inherited."""

    # ---- loading -------------------------------------------------------------

    @staticmethod
    def hidden_size(torso: nn.Module) -> int:
        # The multimodal Qwen3_5Config is composite; the decoder width is on text_config.
        return int(torso.config.get_text_config().hidden_size)

    @staticmethod
    def _load_torso(
        config: StrandsDeciderConfig,
        device_map: str | None,
        attn_implementation: str | None,
    ) -> nn.Module:
        import transformers

        kwargs: dict[str, Any] = {"dtype": getattr(torch, config.torch_dtype)}
        if device_map:
            kwargs["device_map"] = device_map
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        base_cfg = transformers.AutoConfig.from_pretrained(config.base_model)
        if base_cfg.model_type != "qwen3_5":
            raise ValueError(
                f"VisionDeciderModel needs a multimodal qwen3_5 checkpoint, got "
                f"{base_cfg.model_type!r}"
            )
        full = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
            config.base_model, **kwargs
        )
        torso = full.model  # Qwen3_5Model: .visual + .language_model, no lm_head
        torso.config.use_cache = True
        # Frozen before LoRA is attached, so the merger adapter (if any) stays trainable.
        for p in torso.visual.parameters():
            p.requires_grad_(False)
        return torso

    def attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        cfg: VisionDeciderConfig = self.config  # type: ignore[assignment]
        # PEFT full-matches a string `target_modules` against each module name. Scope it
        # to the language model so the ViT's own projections are never adapted.
        pattern = rf"^language_model\..*\.({'|'.join(cfg.lora_targets)})$"
        if cfg.lora_on_merger:
            pattern = rf"(?:{pattern})|(?:^visual\.merger\.({'|'.join(MERGER_TARGETS)})$)"
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=pattern,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.torso = get_peft_model(self.torso, lora_cfg)

    # ---- forward -------------------------------------------------------------

    def encode(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decoder last hidden state with image embeddings scattered into the stream.

        The parent's forward and KL reference call `encode(input_ids, attention_mask,
        past_key_values=...)` with no image arguments; `_mm` carries the batch's image
        tensors for the duration of those calls (see `forward`).
        """
        if pixel_values is None and image_grid_thw is None:
            pixel_values, image_grid_thw = self._mm
        extra: dict[str, Any] = {}
        if image_grid_thw is not None:
            extra["mm_token_type_ids"] = mm_token_type_ids(self.torso, input_ids)
        if position_ids is not None:
            extra["position_ids"] = position_ids
        out = self.torso(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
            **extra,
        )
        return out.last_hidden_state

    def __init__(self, config: StrandsDeciderConfig, torso: nn.Module, tokenizer: Any) -> None:
        super().__init__(config, torso, tokenizer)
        self._mm: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_slots: torch.Tensor,
        labels: torch.Tensor | None = None,
        label_dist: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        past_key_values: Any = None,
        temperature: Any | None = None,
        opt_idx: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._mm = (pixel_values, image_grid_thw)
        try:
            return super().forward(
                input_ids, attention_mask, n_slots,
                labels=labels, label_dist=label_dist, weights=weights,
                past_key_values=past_key_values, temperature=temperature, opt_idx=opt_idx,
            )
        finally:
            self._mm = (None, None)

    def frozen_slot_log_probs(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_slots: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The untouched multimodal torso's own option-number reading, image included."""
        self._mm = (pixel_values, image_grid_thw)
        try:
            return super().frozen_slot_log_probs(input_ids, attention_mask, n_slots)
        finally:
            self._mm = (None, None)

    # ---- persistence ---------------------------------------------------------

    def save_pretrained(self, path: str) -> None:
        super().save_pretrained(path)
        # Upstream wrote every field into strands_decider_config.json, which upstream
        # `StrandsDeciderConfig.from_json` would then refuse; split it.
        self.config.write(path)  # type: ignore[attr-defined]

    @classmethod
    def load(
        cls,
        path: str,
        *,
        device_map: str | None = None,
        attn_implementation: str | None = None,
    ) -> VisionDeciderModel:
        from peft import PeftModel
        from transformers import AutoTokenizer

        path = checkpoint_dir(path)
        config = VisionDeciderConfig.read(path)
        lora_dir = os.path.join(path, "lora")
        if config.use_lora and not os.path.isdir(lora_dir):
            raise FileNotFoundError(f"{lora_dir}: missing, but the checkpoint config sets use_lora")
        head_state = load_head_state(path)
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        torso = cls._load_torso(config, device_map, attn_implementation)
        if config.use_lora:
            torso = PeftModel.from_pretrained(torso, lora_dir, is_trainable=False)

        # __init__ would attach a second LoRA on top of the loaded one; build in place.
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.config = config
        obj.torso = torso
        obj.tokenizer = tok
        obj._mm = (None, None)
        obj.head = build_head(config, cls.hidden_size(torso))
        obj.head.load_state_dict(head_state)
        obj.head.to(torch.float32)
        obj.eval()
        return obj
