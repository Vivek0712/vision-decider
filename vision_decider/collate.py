"""Batch assembly for image rows: processor for pixels, tokenizer offsets for options.

Differences from `strands_decider.data.collate.SystemOneCollator`:
  * the prompt carries one `<|image_pad|>` per image, expanded here to N tokens;
  * `pixel_values` and `image_grid_thw` are returned beside the text tensors;
  * rows whose image or whole prompt exceeds the budget are dropped, not cut, and the
    batch shrinks; a batch with nothing left comes back as None.
Everything else -- option shuffling every example, label remapping, ordinal smoothing,
score reversal, teacher distributions in slot order -- is delegated to the parent.
Pointer head only: targets are cut to the batch's widest option count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from strands_decider.data.collate import CollatorConfig, SystemOneCollator

from .data import VisionExample
from .prompting import (
    VisionState,
    build_vision_prompt,
    expand_image_tokens,
    image_tokens_for_grid,
    option_token_index,
    question_base,
)


@dataclass
class VisionCollatorConfig(CollatorConfig):
    image_long_side: int = 448
    max_image_tokens: int = 640
    # Task-name prefixes whose rows get no frozen-KL term (`kl_mask` False).
    kl_exclude_tasks: tuple[str, ...] = ()


def _load_image(src: Any, long_side: int) -> Image.Image:
    im = src if isinstance(src, Image.Image) else Image.open(src)
    im = im.convert("RGB")
    w, h = im.size
    scale = long_side / max(w, h)
    if scale < 1.0:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return im


class VisionCollator(SystemOneCollator):
    def __init__(self, processor: Any, config: VisionCollatorConfig, *, train: bool = True):
        if config.head_type != "pointer":
            raise ValueError("VisionCollator supports the pointer head only")
        super().__init__(processor.tokenizer, config, train=train)
        self.processor = processor
        self.image_processor = processor.image_processor
        self.vcfg = config
        self.dropped = 0  # rows dropped over the image or window budget, for logging

    def _prepare_images(self, exs: list[VisionExample]) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
        flat = [_load_image(p, self.vcfg.image_long_side) for ex in exs for p in ex.images]
        if not flat:
            return {}, [[] for _ in exs]
        img = self.image_processor(images=flat, return_tensors="pt")
        counts = image_tokens_for_grid(img["image_grid_thw"].tolist(), self.image_processor.merge_size)
        per_row: list[list[int]] = []
        k = 0
        for ex in exs:
            per_row.append(counts[k:k + len(ex.images)])
            k += len(ex.images)
        return {"pixel_values": img["pixel_values"], "image_grid_thw": img["image_grid_thw"]}, per_row

    def __call__(self, batch: list[VisionExample]) -> dict[str, torch.Tensor] | None:  # type: ignore[override]
        mm, per_row = self._prepare_images(batch)
        rows: list[dict[str, Any]] = []
        for i, ex in enumerate(batch):
            if any(n > self.vcfg.max_image_tokens for n in per_row[i]):
                self.dropped += 1
                continue  # drop, never truncate
            order = self._option_order(ex)
            state = VisionState(images=tuple(ex.images), text=ex.state)
            prompt, rq = build_vision_prompt(state, ex.to_question(self._instruction(ex)), option_order=order)
            expanded = expand_image_tokens(prompt, per_row[i])
            enc = self.tok(expanded, truncation=False, return_offsets_mapping=True)
            if len(enc["input_ids"]) > self.cfg.max_length:
                self.dropped += 1
                continue
            t = getattr(ex, "teacher", None)
            rows.append({
                "i": i,
                "ex": ex,
                "ids": enc["input_ids"],
                "opt": option_token_index(enc["offset_mapping"], rq.option_spans, question_base(expanded, rq)),
                "label": self._remap_label(ex.label, order),
                "dist": self._target_distribution(ex, ex.label, ex.n_options, order),
                "teacher": None if t is None else (t if order is None else [t[k] for k in order]),
            })
        if not rows:
            return None

        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        length = max(len(r["ids"]) for r in rows)
        width = max(len(r["opt"]) for r in rows)
        labels = [r["label"] for r in rows]
        out: dict[str, torch.Tensor] = {
            # Right padding: pool_last_token finds the final real token by mask length.
            "input_ids": torch.tensor([r["ids"] + [pad_id] * (length - len(r["ids"])) for r in rows]),
            "attention_mask": torch.tensor([[1] * len(r["ids"]) + [0] * (length - len(r["ids"])) for r in rows]),
            "n_slots": torch.tensor([r["ex"].n_options for r in rows], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weights": torch.tensor([r["ex"].weight for r in rows], dtype=torch.float32),
            "opt_idx": torch.tensor([r["opt"] + [-1] * (width - len(r["opt"])) for r in rows], dtype=torch.long),
            "row_index": torch.tensor([r["i"] for r in rows], dtype=torch.long),  # kept rows' batch positions
            "kl_mask": torch.tensor([not r["ex"].task.startswith(tuple(self.vcfg.kl_exclude_tasks))
                                     for r in rows], dtype=torch.float32),
        }
        if any(r["dist"] is not None for r in rows):
            out["label_dist"] = torch.stack([
                r["dist"] if r["dist"] is not None
                else torch.nn.functional.one_hot(torch.tensor(r["label"]), num_classes=self.cfg.num_slots).float()
                for r in rows
            ])[:, :width]
        if any(r["teacher"] is not None for r in rows):
            tt = torch.zeros((len(rows), width), dtype=torch.float32)
            for i, r in enumerate(rows):
                if r["teacher"] is not None:
                    tt[i, : len(r["teacher"])] = torch.tensor(r["teacher"], dtype=torch.float32)
            out["teacher"] = tt
            out["has_teacher"] = torch.tensor([r["teacher"] is not None for r in rows])
        if len(rows) != len(batch):  # images of dropped rows must not reach the model
            mm, _ = self._prepare_images([r["ex"] for r in rows])
        out.update(mm)
        return out
