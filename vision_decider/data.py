"""VisionExample: a Decider Example with images, plus the converters and the pairing rule.

Keeps the text Decider's `Example` fields (kind, state, instructions, options, label,
task, weight, instruction_variants) and adds `images` (paths, relative to a root) and
`pair_id`. `pair_id` ties a row to its minimal pair: same question and options, a
different image, a different label. Pairs are kept together in a split so the model is
never scored on a question whose twin it trained on.

Torch-free so the converters can run on a laptop while the GPU trains.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from strands_decider.data.format import Example


@dataclass
class VisionExample(Example):
    images: list[str] = field(default_factory=list)
    pair_id: str | None = None
    # Text-only ablation rows keep the question but drop the image; the gold label is
    # replaced by the frozen torso's text-only distribution at labelling time.
    ablation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VisionExample":
        return cls(**d)


def write_jsonl(rows: Iterable[VisionExample], path: str | Path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[VisionExample]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield VisionExample.from_dict(json.loads(line))


# ---- the pairing rule ----------------------------------------------------------------

def make_pairs(rows: list[VisionExample], *, seed: int = 0) -> list[VisionExample]:
    """Attach a minimal pair to every noul/choice row where one exists.

    Two rows pair when they share task, kind, instructions and the option list, carry
    different labels, and have different images. Rows already carrying a `pair_id` are
    left alone. Unpaired rows survive; the balance check (below) reports how many.
    """
    rng = random.Random(seed)
    by_key: dict[tuple, list[VisionExample]] = {}
    for r in rows:
        if r.kind == "score" or r.pair_id or r.ablation:
            continue
        key = (r.task, r.kind, r.instructions, tuple(tuple(o) for o in r.options))
        by_key.setdefault(key, []).append(r)
    n = 0
    for group in by_key.values():
        rng.shuffle(group)
        by_label: dict[int, list[VisionExample]] = {}
        for r in group:
            by_label.setdefault(r.label, []).append(r)
        labels = list(by_label)
        if len(labels) < 2:
            continue
        for i, a in enumerate(labels):
            b = labels[(i + 1) % len(labels)]
            for x, y in zip(by_label[a], by_label[b], strict=False):
                if x.pair_id or y.pair_id or set(x.images) == set(y.images):
                    continue
                pid = f"pair-{n:07d}"
                x.pair_id = y.pair_id = pid
                n += 1
    return rows


def add_ablations(rows: list[VisionExample], fraction: float, *, seed: int = 0) -> list[VisionExample]:
    """Duplicate `fraction` of image rows without their image, marked `ablation=True`.

    The trainer replaces their gold with the frozen torso's text-only distribution, so
    the head learns that an absent image should lower confidence rather than invent it.
    """
    rng = random.Random(seed)
    extra = []
    for r in rows:
        if r.images and not r.ablation and rng.random() < fraction:
            d = r.to_dict()
            d.update(images=[], ablation=True, pair_id=None, task=f"{r.task}/ablation")
            extra.append(VisionExample.from_dict(d))
    return rows + extra


def balance_report(rows: list[VisionExample]) -> dict[str, dict[str, Any]]:
    """Per task: rows, label histogram, paired fraction. Flag anything a prior could solve."""
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        t = out.setdefault(r.task, {"rows": 0, "labels": {}, "paired": 0})
        t["rows"] += 1
        t["labels"][r.label] = t["labels"].get(r.label, 0) + 1
        t["paired"] += bool(r.pair_id)
    for t in out.values():
        top = max(t["labels"].values()) / t["rows"]
        t["majority_share"] = round(top, 3)
        t["paired_share"] = round(t["paired"] / t["rows"], 3)
        t["warn"] = top > 0.6 or t["paired_share"] < 0.3
    return out


def split_by_pairs(rows: list[VisionExample], val_fraction: float, *, seed: int = 0):
    """Stratify by task; never let the two halves of a pair land in different splits."""
    rng = random.Random(seed)
    units: dict[str, list[VisionExample]] = {}
    for r in rows:
        units.setdefault(r.pair_id or f"solo-{id(r)}", []).append(r)
    keys = list(units)
    rng.shuffle(keys)
    cut = int(len(keys) * val_fraction)
    val = [r for k in keys[:cut] for r in units[k]]
    train = [r for k in keys[cut:] for r in units[k]]
    return train, val


# ---- converters ---------------------------------------------------------------------

def from_docvqa_yesno(records: Iterable[dict[str, Any]], image_root: str) -> list[VisionExample]:
    """DocVQA rows whose answers are yes/no become noul questions about the page.

    `records` are DocVQA-style dicts: {"image": "...png", "question": "...",
    "answers": ["Yes"]}. Anything not answered yes/no is skipped here; open answers
    belong to a choice converter that offers the gold plus hard negatives.
    """
    out = []
    for rec in records:
        ans = {a.strip().lower() for a in rec.get("answers", [])}
        if ans & {"yes", "no"} != ans or not ans:
            continue
        out.append(VisionExample(
            kind="noul",
            state="",
            instructions=rec["question"].strip(),
            options=[["false", "the statement does not hold for this page"],
                     ["true", "the statement holds for this page"]],
            label=1 if "yes" in ans else 0,
            task="docvqa/yesno",
            images=[str(Path(image_root) / rec["image"])],
        ))
    return out


def from_labelled_images(
    records: Iterable[dict[str, Any]],
    label_descriptions: dict[str, str],
    *,
    task: str,
    instructions: str,
    n_options: int,
    seed: int = 0,
) -> list[VisionExample]:
    """Any image classification set becomes choice rows with a sampled option list.

    Each row shows the gold label plus `n_options - 1` distractors drawn from the label
    space, with their descriptions, so the model learns to read option text rather than
    a fixed class index. Option order is shuffled again at collate time.
    """
    rng = random.Random(seed)
    names = list(label_descriptions)
    out = []
    for rec in records:
        gold = rec["label"]
        if gold not in label_descriptions:
            continue
        pool = [n for n in names if n != gold]
        shown = [gold] + rng.sample(pool, min(n_options - 1, len(pool)))
        rng.shuffle(shown)
        out.append(VisionExample(
            kind="choice",
            state=rec.get("state", ""),
            instructions=instructions,
            options=[[n, label_descriptions[n]] for n in shown],
            label=shown.index(gold),
            task=task,
            images=[rec["image"]],
        ))
    return out


def from_ordinal_ratings(
    records: Iterable[dict[str, Any]], levels: list[str], *, task: str, instructions: str
) -> list[VisionExample]:
    """Ratings on an ordered scale become score rows; `levels[i]` is the rubric for i."""
    out = []
    for rec in records:
        lvl = int(rec["level"])
        if not 0 <= lvl < len(levels):
            continue
        out.append(VisionExample(
            kind="score",
            state=rec.get("state", ""),
            instructions=instructions,
            options=[[str(i), d] for i, d in enumerate(levels)],
            label=lvl,
            task=task,
            images=[rec["image"]],
        ))
    return out
