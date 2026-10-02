"""Train a Vision Decider: LoRA on the decoder + fp32 pointer head, vision tower frozen.

Loss = CE(gold, ordinal-smoothed)
     + kl_frozen_weight * KL(frozen multimodal torso readout || student)   [rows it covers]
     + teacher_weight   * KL(teacher || student)                            [rows with a teacher]
Ablation rows (image dropped) use the frozen torso's text-only distribution as their
target instead of the gold label, so the head learns to lower confidence without the
image rather than guess from the question.

Single-GPU reference loop; mirror `strands_decider.distributed` for torchrun.
Run: python -m vision_decider.train configs/train-vision.yaml
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from strands_decider.data.sampling import LengthGroupedBatchSampler
from strands_decider.modeling import masked_log_softmax

from .collate import VisionCollator, VisionCollatorConfig
from .data import read_jsonl, split_by_pairs
from .modeling import VisionDeciderConfig, VisionDeciderModel


def _kl(target_log_probs: torch.Tensor, student_log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(target || student) over valid slots, averaged over masked rows."""
    if not bool(mask.any()):
        return student_log_probs.new_zeros(())
    t = target_log_probs[mask]
    s = student_log_probs[mask]
    valid = torch.isfinite(t) & torch.isfinite(s)
    tp = torch.where(valid, t.exp(), torch.zeros_like(t))
    diff = torch.where(valid, t - s, torch.zeros_like(t))
    return (tp * diff).sum(-1).mean()


def main(cfg_path: str) -> None:
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    torch.manual_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mcfg = VisionDeciderConfig(**{k: v for k, v in cfg.items() if k in VisionDeciderConfig.__dataclass_fields__})
    model = VisionDeciderModel.from_pretrained_base(mcfg, attn_implementation=cfg.get("attn_implementation"))
    model.freeze_vision_tower()
    model.to(device)
    if cfg.get("gradient_checkpointing", True):
        model.torso.gradient_checkpointing_enable()
    trainable, total = model.trainable_parameters()
    print(f"trainable {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    processor = AutoProcessor.from_pretrained(mcfg.base_model)
    ccfg = VisionCollatorConfig(
        num_slots=mcfg.num_slots, max_length=mcfg.max_length, head_type="pointer",
        shuffle_options=cfg.get("shuffle_options", True),
        ordinal_smoothing=cfg.get("ordinal_smoothing", 0.1),
        reverse_score_prob=cfg.get("reverse_score_prob", 0.5),
        image_long_side=mcfg.image_long_side, max_image_tokens=mcfg.max_image_tokens,
    )
    collate = VisionCollator(processor, ccfg, train=True)

    rows = [r for f in cfg["train_files"] for r in read_jsonl(f)]
    if cfg.get("teacher_file"):
        with open(cfg["teacher_file"], encoding="utf-8") as fh:
            teacher = {json.loads(l)["id"]: json.loads(l)["dist"] for l in fh if l.strip()}
        for i, r in enumerate(rows):
            t = teacher.get(f"{r.task}:{i}")
            if t is not None:
                r.teacher = t  # type: ignore[attr-defined]
    train_rows, val_rows = split_by_pairs(rows, cfg.get("val_fraction", 0.03), seed=cfg.get("seed", 0))
    print(f"train {len(train_rows)} val {len(val_rows)}")

    # Length grouping: image rows count ~400 tokens per image on top of their text.
    lengths = [len(str(r.state)) // 4 + 400 * len(r.images) for r in train_rows]
    sampler = LengthGroupedSampler(lengths, cfg["micro_batch_size"], seed=cfg.get("seed", 0)) \
        if cfg.get("group_by_length", True) else None
    loader = DataLoader(train_rows, batch_size=cfg["micro_batch_size"], shuffle=sampler is None,
                        sampler=sampler, collate_fn=collate, num_workers=cfg.get("num_workers", 4))

    head_params = list(model.head.parameters())
    lora_params = [p for n, p in model.torso.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": cfg["lr"]},
        {"params": head_params, "lr": cfg["head_lr"]},
    ], weight_decay=cfg.get("weight_decay", 0.01))
    steps = math.ceil(len(loader) * cfg.get("epochs", 1) / cfg.get("grad_accum", 1))
    warm = int(steps * cfg.get("warmup_ratio", 0.03))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))))

    kl_w = cfg.get("kl_frozen_weight", 0.3)
    t_w = cfg.get("teacher_weight", 1.0)
    accum = cfg.get("grad_accum", 1)
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(cfg.get("epochs", 1)):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(
                    batch["input_ids"], batch["attention_mask"], batch["n_slots"],
                    labels=batch["labels"], label_dist=batch.get("label_dist"), weights=batch["weights"],
                    opt_idx=batch["opt_idx"],
                    pixel_values=batch.get("pixel_values"), image_grid_thw=batch.get("image_grid_thw"),
                )
            loss = out["loss"]
            logs = {"ce": float(loss)}

            if kl_w > 0 and mcfg.kl_frozen_on_images:
                frozen, eligible = model.frozen_slot_log_probs(
                    batch["input_ids"], batch["attention_mask"], batch["n_slots"],
                    pixel_values=batch.get("pixel_values"), image_grid_thw=batch.get("image_grid_thw"))
                if frozen.numel():
                    student = out["log_probs"][eligible]
                    k = _kl(frozen, student, torch.ones(student.size(0), dtype=torch.bool, device=device))
                    loss = loss + kl_w * k
                    logs["kl_frozen"] = float(k)

            if "teacher" in batch and t_w > 0:
                t_log = masked_log_softmax(batch["teacher"].clamp_min(1e-8).log(), batch["n_slots"])
                k = _kl(t_log, out["log_probs"], batch["teacher_mask"])
                loss = loss + t_w * k
                logs["kl_teacher"] = float(k)

            (loss / accum).backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, cfg.get("max_grad_norm", 1.0))
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.get("log_every", 20) == 0:
                    print(f"step {step}/{steps} loss {float(loss):.4f} {logs} {time.time() - t0:.0f}s")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    model.save_pretrained(cfg["output_dir"])
    with open(os.path.join(cfg["output_dir"], "train_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"yaml": cfg, "model": asdict(mcfg)}, fh, indent=2)
    print("saved", cfg["output_dir"])


if __name__ == "__main__":
    main(sys.argv[1])
