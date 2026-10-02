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
`lora_on_merger=True` additionally adapts the patch merger, the cheapest place to
change how image tokens enter the decoder.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from strands_decider.modeling import (
    CONFIG_NAME,
    StrandsDeciderConfig,
    StrandsDeciderModel,
    build_head,
)


@dataclass
class VisionDeciderConfig(StrandsDeciderConfig):
    base_model: str = "Qwen/Qwen3.5-2B-Base"
    head_type: str = "pointer"
    # Longest image side after resizing; the processor's dynamic resolution then yields
    # a bounded token count. Fixed per deployment so the window budget is predictable.
    image_long_side: int = 448
    # Upper bound on tokens one image may expand to; rows over it are dropped, never
    # truncated (the Decider rule: drop, never truncate through an option list).
    max_image_tokens: int = 640
    # Whether to attach LoRA to the vision patch merger as well as the decoder.
    lora_on_merger: bool = False
    # KL(frozen || student) also applies to image rows; disable per task family where
    # the untrained torso is near chance on images (measure first).
    kl_frozen_on_images: bool = True
    lora_targets: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
        ]
    )


class VisionDeciderModel(StrandsDeciderModel):
    """Image-aware torso; readout inherited."""

    # ---- loading -------------------------------------------------------------

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
        return torso

    def attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        cfg: VisionDeciderConfig = self.config  # type: ignore[assignment]
        # PEFT matches `target_modules` by name suffix. The vision blocks also have
        # `q_proj`-style names in some revisions, so scope the match with a regex to the
        # language model (and optionally the merger) rather than adapting the ViT.
        names = "|".join(cfg.lora_targets)
        scope = r"language_model\..*"
        if cfg.lora_on_merger:
            scope = r"(language_model\..*|visual\.merger\..*)"
        pattern = rf"^{scope}\.({names})$"
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=pattern,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.torso = get_peft_model(self.torso, lora_cfg)

    def freeze_vision_tower(self) -> None:
        base = self.torso
        base = getattr(base, "base_model", base)
        base = getattr(base, "model", base)
        for p in base.visual.parameters():
            p.requires_grad_(False)

    # ---- forward -------------------------------------------------------------

    def encode(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decoder last hidden state with image embeddings scattered into the stream.

        With a shared prefix cache the caller forwards only the question suffix, which
        holds no image placeholders, so `pixel_values` is None on that path and the
        cached recurrent/KV state already carries what the image contributed.
        """
        out = self.torso(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        return out.last_hidden_state

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
        # Route the image tensors through `encode` by stashing them for the duration of
        # the parent's forward, which calls `self.encode(input_ids, attention_mask,
        # past_key_values=...)` and knows nothing about images.
        self._mm = (pixel_values, image_grid_thw)
        try:
            return super().forward(
                input_ids, attention_mask, n_slots,
                labels=labels, label_dist=label_dist, weights=weights,
                past_key_values=past_key_values, temperature=temperature, opt_idx=opt_idx,
            )
        finally:
            self._mm = (None, None)

    # The parent calls encode(input_ids, attention_mask, past_key_values=...). Wrap it so
    # the stashed image tensors ride along without editing the parent class.
    def _encode_with_stash(self, input_ids, attention_mask, past_key_values=None):
        pv, grid = getattr(self, "_mm", (None, None))
        return VisionDeciderModel.encode(
            self, input_ids, attention_mask, past_key_values=past_key_values,
            pixel_values=pv, image_grid_thw=grid,
        )

    def __init__(self, config: StrandsDeciderConfig, torso: nn.Module, tokenizer: Any) -> None:
        super().__init__(config, torso, tokenizer)
        self._mm: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)
        # Bind the stash-aware encoder for the parent's forward and KL reference paths.
        self.encode = self._encode_with_stash  # type: ignore[method-assign]

    # ---- KL reference on image rows -----------------------------------------

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

    def _output_embedding(self) -> torch.Tensor:
        """The tied LM output matrix, reached through the multimodal wrapper.

        `Qwen3_5Model` keeps the token table on `language_model.embed_tokens`; the text
        Decider's walk (`base_model` -> `model`) stops one level short of it.
        """
        base = self.torso
        base = getattr(base, "base_model", base)
        base = getattr(base, "model", base)
        lm = getattr(base, "language_model", base)
        emb = lm.get_input_embeddings()
        if not getattr(lm.config, "tie_word_embeddings", False):
            raise RuntimeError("base model does not tie embeddings, so the input table is not the LM head")
        return emb.weight

    # ---- persistence ---------------------------------------------------------

    def save_pretrained(self, path: str) -> None:
        super().save_pretrained(path)
        with open(os.path.join(path, "vision_decider.json"), "w", encoding="utf-8") as fh:
            fh.write(self.config.to_json())

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

        vcfg = os.path.join(path, "vision_decider.json")
        config = VisionDeciderConfig.from_json(vcfg if os.path.exists(vcfg)
                                              else os.path.join(path, CONFIG_NAME))
        tok = AutoTokenizer.from_pretrained(path)
        torso = cls._load_torso(config, device_map, attn_implementation)
        lora_dir = os.path.join(path, "lora")
        if config.use_lora and os.path.isdir(lora_dir):
            torso = PeftModel.from_pretrained(torso, lora_dir, is_trainable=False)
        model = cls(config, torso, tok)
        head_path = os.path.join(path, "slot_head.pt")
        if os.path.exists(head_path):
            model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
        else:
            from safetensors.torch import load_file
            model.head.load_state_dict(load_file(os.path.join(path, "head.safetensors")))
        model.head = model.head.to(torch.float32)
        model.eval()
        return model
