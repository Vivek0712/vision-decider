"""Train a Vision Decider: LoRA on the decoder + fp32 pointer head, vision tower frozen.

Loss = CE(gold, ordinal-smoothed)                                     [rows with weight > 0]
     + kl_frozen_weight * KL(frozen multimodal torso readout || student) [eligible rows]
     + kl_only_weight   * KL(frozen readout || student)                  [ablation rows]
     + teacher_weight   * KL(teacher || student)                         [rows with a teacher]
Ablation rows (`data.add_ablations`: the image removed, weight 0) carry no label loss;
their only target is the frozen torso's reading of the same text-only prompt, so the
head learns to lose confidence without the image rather than guess from the question.
This is upstream's KL-only mechanism (`kl_only_files`) applied to those rows.

Single-GPU loop mirroring `strands_decider.train`: same optimizer groups, schedule and
KL blocks. Run: python -m vision_decider.train configs/train-vision.yaml
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from strands_decider.data.sampling import LengthGroupedBatchSampler, example_length
from strands_decider.train import _build_optimizer, _lr_lambda

from .collate import VisionCollator, VisionCollatorConfig
from .data import MAX_KL_OPTIONS, VisionExample, read_jsonl, split_by_source
from .modeling import VisionDeciderConfig, VisionDeciderModel


def _image_chars(cfg: VisionDeciderConfig) -> int:
    """An image's weight in the sampler's character units: ~4 characters per token."""
    return 4 * (cfg.image_long_side // 32) ** 2


def _to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _forward(model: VisionDeciderModel, batch: dict[str, torch.Tensor], **kw: Any) -> dict[str, torch.Tensor]:
    return model(
        batch["input_ids"], batch["attention_mask"], batch["n_slots"],
        opt_idx=batch["opt_idx"],
        pixel_values=batch.get("pixel_values"), image_grid_thw=batch.get("image_grid_thw"),
        **kw,
    )


@torch.no_grad()
def evaluate_loss(model: VisionDeciderModel, loader: DataLoader, device: str, max_batches: int = 50) -> dict[str, float]:
    model.eval()
    total_loss, correct, n, batches = 0.0, 0, 0, 0
    for batch in loader:
        if batch is None:
            continue
        if batches >= max_batches:
            break
        batch = _to_device(batch, device)
        out = _forward(model, batch, labels=batch["labels"], label_dist=batch.get("label_dist"),
                       weights=batch["weights"])
        total_loss += float(out["loss"])
        correct += int((out["log_probs"].argmax(dim=-1) == batch["labels"]).sum())
        n += batch["labels"].numel()
        batches += 1
    model.train()
    return {"val_loss": total_loss / max(1, batches), "val_acc": correct / max(1, n)}


def _attach_teacher(rows: list[VisionExample], path: str) -> int:
    """Upstream format: {"i": row index in the concatenated train_files, "probs": [...]}."""
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            ex = rows[d["i"]]
            if len(d["probs"]) != ex.n_options:
                raise ValueError(f"teacher row {d['i']} has {len(d['probs'])} probs for "
                                 f"{ex.n_options} options -- wrong train_files?")
            ex.teacher = d["probs"]  # type: ignore[attr-defined]
            n += 1
    return n


def main(cfg_path: str) -> None:
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    kl_w = cfg.get("kl_frozen_weight", 0.3)
    kl_only_w = cfg.get("kl_only_weight", 1.0)
    t_w = cfg.get("teacher_weight", 1.0)
    accum = cfg.get("grad_accum", 1)
    exclude = tuple(cfg.get("kl_frozen_exclude_tasks", []))

    mcfg = VisionDeciderConfig(**{k: v for k, v in cfg.items() if k in VisionDeciderConfig.__dataclass_fields__})
    model = VisionDeciderModel.from_pretrained_base(mcfg, attn_implementation=cfg.get("attn_implementation"))
    model.to(device)
    if cfg.get("gradient_checkpointing", True):
        model.torso.gradient_checkpointing_enable()
    trainable, total = model.trainable_parameters()
    print(f"[vision-decider] trainable {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    # ---- data ----
    rows = [r for f in cfg["train_files"] for r in read_jsonl(f)]
    if cfg.get("teacher_file"):
        print(f"[vision-decider] teacher distributions on {_attach_teacher(rows, cfg['teacher_file']):,} rows")
    for r in rows:
        if r.ablation:
            r.weight = 0.0
        elif r.weight <= 0:
            raise ValueError(f"labelled row in {r.task!r} has weight {r.weight}: weight 0 marks KL-only rows")
    abl = [r for r in rows if r.ablation]
    if abl and kl_w <= 0:
        raise ValueError("ablation rows need kl_frozen_weight > 0: KL is their only loss")
    # An ablation row with no usable KL target would train on nothing.
    rows = [r for r in rows if not (r.ablation and (r.n_options > MAX_KL_OPTIONS or r.task.startswith(exclude)))]
    train_rows, val_rows, held = split_by_source(
        rows, cfg.get("val_fraction", 0.03), held_out_tasks=cfg.get("held_out_tasks", []), seed=seed)
    val_rows = [r for r in val_rows if not r.ablation]  # their labels would be fake
    print(f"[vision-decider] train {len(train_rows):,} ({sum(r.ablation for r in train_rows):,} ablation) "
          f"val {len(val_rows):,} held-out {len(held):,}")
    if not train_rows:
        raise SystemExit("no training rows")

    processor = AutoProcessor.from_pretrained(mcfg.base_model)
    ccfg = VisionCollatorConfig(
        num_slots=mcfg.num_slots, max_length=mcfg.max_length, head_type="pointer",
        shuffle_options=cfg.get("shuffle_options", True),
        ordinal_smoothing=cfg.get("ordinal_smoothing", 0.1),
        reverse_score_prob=cfg.get("reverse_score_prob", 0.5),
        seed=seed,
        image_long_side=mcfg.image_long_side, max_image_tokens=mcfg.max_image_tokens,
        kl_exclude_tasks=exclude,
    )
    collate = VisionCollator(processor, ccfg, train=True)
    mb = cfg["micro_batch_size"]
    # num_workers 0: the collator's option-shuffle RNG lives in this process; forked
    # workers would each replay the same permutations.
    if cfg.get("group_by_length", True):
        per_image = _image_chars(mcfg)
        lengths = [example_length(r) + per_image * len(r.images) for r in train_rows]
        sampler = LengthGroupedBatchSampler(lengths, mb, mega=cfg.get("length_group_mega", 50),
                                            seed=seed, drop_last=True)
        loader = DataLoader(train_rows, batch_sampler=sampler, collate_fn=collate, num_workers=0)
    else:
        loader = DataLoader(train_rows, batch_size=mb, shuffle=True, drop_last=True,
                            collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_rows, batch_size=mb, shuffle=False,
                            collate_fn=VisionCollator(processor, ccfg, train=False))

    epochs = cfg.get("epochs", 1)
    total_steps = cfg.get("max_steps") or max(1, len(loader) // accum) * epochs
    warmup = max(1, int(total_steps * cfg.get("warmup_ratio", 0.03)))
    optim = _build_optimizer(model, SimpleNamespace(head_lr=cfg["head_lr"], lr=cfg["lr"],
                                                    weight_decay=cfg.get("weight_decay", 0.01)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: _lr_lambda(s, warmup, total_steps))
    params = [p for p in model.parameters() if p.requires_grad]

    print(f"[vision-decider] {total_steps} optimizer steps (warmup {warmup})")
    model.train()
    step, micro, skipped, t0 = 0, 0, 0, time.time()
    logs: dict[str, float] = {}
    history: list[dict[str, float]] = []
    done = False
    for epoch in range(epochs):
        if done:
            break
        for batch in loader:
            if batch is None:
                skipped += 1
                continue
            batch = _to_device(batch, device)
            out = _forward(model, batch, labels=batch["labels"], label_dist=batch.get("label_dist"),
                           weights=batch["weights"])
            labelled = batch["weights"] > 0
            # The parent's weighted mean is 0 for an all-ablation batch, so no NaN here.
            loss = out["loss"] if bool(labelled.any()) else out["loss"] * 0
            logs = {"ce": float(loss.detach())}

            if kl_w > 0:
                ref_lp, eligible = model.frozen_slot_log_probs(
                    batch["input_ids"], batch["attention_mask"], batch["n_slots"],
                    pixel_values=batch.get("pixel_values"), image_grid_thw=batch.get("image_grid_thw"))
                if ref_lp.numel() and bool(eligible.any()):
                    ref = ref_lp[eligible]
                    stu = out["log_probs"][eligible]
                    # The frozen reference is num_slots wide; a pointer student only as
                    # wide as the batch's widest question. Dropped columns are -inf in both.
                    ref = ref[:, : stu.shape[-1]]
                    valid = torch.isfinite(ref) & torch.isfinite(stu)
                    p = ref.exp().masked_fill(~valid, 0.0)
                    per_row = (p * (ref - stu).masked_fill(~valid, 0.0)).sum(dim=-1)
                    only = batch["weights"][eligible] == 0
                    coef = torch.where(only, kl_only_w, kl_w)
                    # Families where the untrained torso is near chance give no useful target.
                    coef = coef * batch["kl_mask"][eligible]
                    loss = loss + (coef * per_row).mean()
                    logs["kl_frozen"] = float(per_row.detach().mean())

            if t_w > 0 and "has_teacher" in batch and bool(batch["has_teacher"].any()):
                has = batch["has_teacher"]
                stu = out["log_probs"][has]
                tea = batch["teacher"][has][:, : stu.shape[-1]]
                valid = (tea > 0) & torch.isfinite(stu)
                diff = (tea.clamp_min(1e-12).log() - stu).masked_fill(~valid, 0.0)
                tkl = (tea.masked_fill(~valid, 0.0) * diff).sum(dim=-1).mean()
                loss = loss + t_w * tkl
                logs["kl_teacher"] = float(tkl.detach())

            (loss / accum).backward()
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, cfg.get("max_grad_norm", 1.0))
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.get("log_every", 20) == 0:
                print(f"[vision-decider] epoch {epoch} step {step}/{total_steps} loss {float(loss.detach()):.4f} "
                      f"{logs} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
            if cfg.get("eval_every") and step % cfg["eval_every"] == 0 and val_rows:
                ev = evaluate_loss(model, val_loader, device)
                history.append({"step": step, **ev})
                print(f"[vision-decider] step {step} {ev}")
            if step >= total_steps:
                done = True
                break

    if micro % accum:  # a trailing partial accumulation still updates the weights
        torch.nn.utils.clip_grad_norm_(params, cfg.get("max_grad_norm", 1.0))
        optim.step()
        optim.zero_grad(set_to_none=True)
    if skipped or collate.dropped:
        print(f"[vision-decider] {collate.dropped:,} rows dropped over budget, {skipped} empty batches")
    if val_rows:
        ev = evaluate_loss(model, val_loader, device, max_batches=cfg.get("final_eval_batches", 200))
        history.append({"step": step, **ev})
        print(f"[vision-decider] final {ev}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    model.save_pretrained(cfg["output_dir"])
    with open(os.path.join(cfg["output_dir"], "train_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"yaml": cfg, "model": asdict(mcfg), "history": history}, fh, indent=2)
    print("[vision-decider] saved", cfg["output_dir"])


if __name__ == "__main__":
    main(sys.argv[1])
