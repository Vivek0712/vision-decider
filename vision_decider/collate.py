"""Batch assembly for image rows: processor for pixels, tokenizer offsets for options.

Differences from `strands_decider.data.collate.Collator`:
  * the prompt carries one `<|image_pad|>` per image, expanded here to N tokens;
  * `pixel_values` and `image_grid_thw` are returned beside the text tensors;
  * rows whose image expansion plus question exceeds the window are dropped, not cut.
Everything else -- option shuffling every example, label remapping, ordinal smoothing,
score reversal, teacher distributions in slot order -- is delegated to the parent.
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
        super().__init__(processor.tokenizer, config, train=train)
        self.processor = processor
        self.image_processor = processor.image_processor
        self.vcfg = config

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

    def __call__(self, batch: list[VisionExample]) -> dict[str, torch.Tensor]:  # type: ignore[override]
        mm, per_row = self._prepare_images(batch)
        texts: list[str] = []
        spans: list[tuple[int, Any]] = []
        labels: list[int] = []
        n_slots: list[int] = []
        dists: list[torch.Tensor | None] = []
        weights: list[float] = []
        teachers: list[list[float] | None] = []
        keep: list[int] = []

        for i, ex in enumerate(batch):
            if any(n > self.vcfg.max_image_tokens for n in per_row[i]):
                continue  # drop, never truncate
            order = self._option_order(ex)
            state = VisionState(images=tuple(ex.images), text=ex.state)
            prompt, rq = build_vision_prompt(state, ex.to_question(self._instruction(ex)), option_order=order)
            expanded = expand_image_tokens(prompt, per_row[i])
            texts.append(expanded)
            spans.append((question_base(expanded, rq), rq.option_spans))
            labels.append(self._remap_label(ex.label, order))
            n_slots.append(ex.n_options)
            dists.append(self._target_distribution(ex, ex.label, ex.n_options, order))
            weights.append(ex.weight)
            t = getattr(ex, "teacher", None)
            teachers.append(None if t is None else (t if order is None else [t[k] for k in order]))
            keep.append(i)

        if not texts:
            raise ValueError("every row in the batch exceeded max_image_tokens")

        enc = self.tok(
            texts, padding=True, truncation=False, return_tensors="pt",
            padding_side="right", return_offsets_mapping=True,
        )
        if enc["input_ids"].size(1) > self.cfg.max_length:
            raise ValueError(
                f"batch is {enc['input_ids'].size(1)} tokens, over the {self.cfg.max_length} "
                "window; lower image_long_side or drop the row upstream"
            )
        offs = enc["offset_mapping"].tolist()
        per = [option_token_index(offs[i], sp, base) for i, (base, sp) in enumerate(spans)]
        width = max(len(p) for p in per)

        out: dict[str, torch.Tensor] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "n_slots": torch.tensor(n_slots, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
            "opt_idx": torch.tensor([p + [-1] * (width - len(p)) for p in per], dtype=torch.long),
        }
        if any(d is not None for d in dists):
            out["label_dist"] = torch.stack([
                d if d is not None
                else torch.nn.functional.one_hot(torch.tensor(lab), num_classes=self.cfg.num_slots).float()
                for d, lab in zip(dists, labels, strict=True)
            ])
        if any(t is not None for t in teachers):
            tw = torch.zeros(len(teachers), self.cfg.num_slots)
            tm = torch.zeros(len(teachers), dtype=torch.bool)
            for r, t in enumerate(teachers):
                if t is not None:
                    tw[r, : len(t)] = torch.tensor(t)
                    tm[r] = True
            out["teacher"], out["teacher_mask"] = tw, tm
        if mm:
            # Images of dropped rows must not reach the model: re-run the processor on
            # the kept rows only when anything was dropped.
            if len(keep) != len(batch):
                mm, _ = self._prepare_images([batch[i] for i in keep])
            out.update(mm)
        return out
