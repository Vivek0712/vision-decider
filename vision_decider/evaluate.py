"""Evaluate and calibrate a Vision Decider on image rows.

`strands-decider calibrate` loads the text torso and feeds no pixels, so it cannot fit
temperatures for image questions. This module runs the same procedure -- one forward,
untempered logits kept, temperature fitted per question kind on ECE -- through the
vision collator and model, and reuses upstream's metric and fitting code unchanged.

    python -m vision_decider.evaluate eval CKPT rows.jsonl [--device cuda]
    python -m vision_decider.evaluate calibrate CKPT rows.jsonl [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from strands_decider.evaluate import (
    fit_temperature,
    fit_temperature_by_kind,
    predictions_from_logits,
    summarise,
)
from strands_decider.modeling import MASK_VALUE

from .collate import VisionCollator, VisionCollatorConfig
from .data import VisionExample, read_jsonl
from .modeling import VisionDeciderModel


@torch.no_grad()
def collect_logits(
    model: VisionDeciderModel, rows: list[VisionExample], *, device: str, batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[VisionExample]]:
    """Untempered logits for every row the collator keeps, padded columns masked."""
    model.eval().to(device)
    cfg = model.config
    processor = AutoProcessor.from_pretrained(cfg.base_model)
    coll = VisionCollator(processor, VisionCollatorConfig(
        max_length=cfg.max_length, num_slots=cfg.num_slots, head_type="pointer",
        image_long_side=cfg.image_long_side, max_image_tokens=cfg.max_image_tokens,
    ), train=False)  # no option shuffling: evaluation must be deterministic
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=lambda b: (b, coll(b)))

    logits_all, labels_all, slots_all, kept = [], [], [], []
    for exs, batch in loader:
        if batch is None:
            continue
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], batch["attention_mask"], batch["n_slots"],
                    opt_idx=batch["opt_idx"], temperature=1.0,
                    pixel_values=batch.get("pixel_values"), image_grid_thw=batch.get("image_grid_thw"))
        logits = out["logits"].float()
        cols = torch.arange(logits.size(1), device=logits.device)
        logits = logits.masked_fill(cols >= batch["n_slots"].unsqueeze(1), MASK_VALUE)
        logits_all.append(logits.cpu())
        labels_all.append(batch["labels"].cpu())
        slots_all.append(batch["n_slots"].cpu())
        kept.extend(exs[i] for i in batch["row_index"].tolist())
    if not kept:
        raise ValueError("no rows survived the collator's budget")
    width = max(t.size(1) for t in logits_all)
    logits_all = [F.pad(t, (0, width - t.size(1)), value=MASK_VALUE) for t in logits_all]
    if len(kept) < len(rows):
        print(f"[vision-decider] {len(rows) - len(kept):,} rows over budget, not scored")
    return torch.cat(logits_all), torch.cat(labels_all), torch.cat(slots_all), kept


def evaluate(model: VisionDeciderModel, rows: list[VisionExample], *, device: str,
             batch_size: int = 8) -> dict[str, Any]:
    logits, labels, slots, kept = collect_logits(model, rows, device=device, batch_size=batch_size)
    t: Any = model.config.temperature_by_kind or model.config.temperature
    preds = predictions_from_logits(logits, labels, slots, kept, temperature=t,
                                    ordinal_smoothing=model.config.ordinal_smoothing)
    result = summarise(preds)
    result["temperature"] = t
    return result


def calibrate(checkpoint: str, rows: list[VisionExample], *, device: str, batch_size: int = 8) -> dict[str, Any]:
    """Fit temperatures on held-out image rows and write them into the checkpoint."""
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"{checkpoint}: calibration writes into the checkpoint; pass a local directory")
    model = VisionDeciderModel.load(checkpoint)
    logits, labels, slots, kept = collect_logits(model, rows, device=device, batch_size=batch_size)
    eps = model.config.ordinal_smoothing
    before = summarise(predictions_from_logits(logits, labels, slots, kept, 1.0, eps))
    t = fit_temperature(logits, labels, slots)
    by_kind = fit_temperature_by_kind(logits, labels, slots, kept, objective="ece", ordinal_smoothing=eps)
    after = summarise(predictions_from_logits(logits, labels, slots, kept, by_kind or t, eps))
    model.config.temperature = float(t)
    model.config.temperature_by_kind = by_kind
    model.config.write(checkpoint)  # type: ignore[attr-defined]
    return {"temperature": float(t), "temperature_by_kind": by_kind,
            "before": before["overall"], "after": after["overall"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["eval", "calibrate"])
    ap.add_argument("checkpoint")
    ap.add_argument("rows")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    rows = [r for r in read_jsonl(a.rows) if not r.ablation]
    if a.command == "calibrate":
        res = calibrate(a.checkpoint, rows, device=a.device, batch_size=a.batch_size)
    else:
        res = evaluate(VisionDeciderModel.load(a.checkpoint), rows, device=a.device, batch_size=a.batch_size)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
