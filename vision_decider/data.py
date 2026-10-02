"""VisionExample: a Decider Example with images, plus the converters and the pairing rule.

Keeps the text Decider's `Example` fields (kind, state, instructions, options, label,
task, weight, instruction_variants) and adds `images` (paths), `pair_id`, `ablation`
and `source_id`. `pair_id` ties a row to its minimal pair: same question and options, a
different image, a different answer. `source_id` names what a row was derived from, so
the split keeps a row, its pair and its text-only ablation copy on one side.

Torch-free so the converters can run on a laptop while the GPU trains.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from strands_decider.data.format import Example

# The frozen torso's readout scores option numbers 1-9, the single-token digits, so a
# KL-only row can carry at most this many options.
MAX_KL_OPTIONS = 9


@dataclass
class VisionExample(Example):
    images: list[str] = field(default_factory=list)
    pair_id: str | None = None
    # A text-only copy of an image row (`add_ablations`). Its weight is 0: no label loss,
    # trained only toward the frozen torso's reading of the same question without the
    # image, so the head learns to lose confidence rather than guess from the question.
    ablation: bool = False
    source_id: str | None = None

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

def _gold(r: VisionExample) -> str:
    return r.options[r.label][0]


def make_pairs(rows: list[VisionExample], *, seed: int = 0) -> list[VisionExample]:
    """Attach a minimal pair to every noul/choice row where one exists.

    Two rows pair when they share task, kind, instructions and the option SET (the
    collator reshuffles order anyway), have different gold options, and have different
    images. Rows already carrying a `pair_id` (e.g. from VQAv2's complementary-pairs
    file) are left alone. Pairs form only where questions recur across images:
    templated questions, or a fixed option set per task. Free-form per-image questions
    (DocVQA) and per-row sampled distractors yield almost none.
    """
    rng = random.Random(seed)
    by_key: dict[tuple, list[VisionExample]] = {}
    for r in rows:
        if r.kind == "score" or r.pair_id or r.ablation:
            continue
        key = (r.task, r.kind, r.instructions, frozenset(tuple(o) for o in r.options))
        by_key.setdefault(key, []).append(r)
    n = 0
    for group in by_key.values():
        rng.shuffle(group)
        by_gold: dict[str, list[VisionExample]] = {}
        for r in group:
            by_gold.setdefault(_gold(r), []).append(r)
        # Repeatedly pair one row from each of the two largest remaining gold groups:
        # pairs as many rows as the group sizes allow.
        pools = [g for g in by_gold.values() if g]
        while len(pools) >= 2:
            pools.sort(key=len, reverse=True)
            x, y = pools[0].pop(), pools[1].pop()
            if set(x.images) != set(y.images):
                pid = f"pair-{n:07d}"
                x.pair_id = y.pair_id = pid
                n += 1
            pools = [g for g in pools if g]
    return rows


def add_ablations(rows: list[VisionExample], fraction: float, *, seed: int = 0) -> list[VisionExample]:
    """Duplicate `fraction` of image rows without their image, as KL-only rows.

    The copy has weight 0, so it contributes no label loss; the trainer pulls it toward
    the frozen torso's distribution on the same text-only prompt. Rows with more options
    than the frozen readout can score are skipped.
    """
    rng = random.Random(seed)
    extra = []
    for r in rows:
        if (r.images and not r.ablation and r.n_options <= MAX_KL_OPTIONS
                and rng.random() < fraction):
            d = r.to_dict()
            d.update(images=[], ablation=True, pair_id=None, weight=0.0,
                     task=f"{r.task}/ablation", source_id=_source(r))
            extra.append(VisionExample.from_dict(d))
    return rows + extra


def _source(r: VisionExample) -> str:
    return r.source_id or (r.images[0] if r.images else f"row-{id(r)}")


def balance_report(rows: list[VisionExample]) -> dict[str, dict[str, Any]]:
    """Per task: rows, gold-option histogram, paired fraction. Flag what a prior could solve.

    A task warns when one gold option holds over 60% of rows, or when under 30% of a
    noul/choice task is paired. Score tasks and ablation copies are not expected to pair.
    """
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        t = out.setdefault(r.task, {"rows": 0, "labels": {}, "paired": 0,
                                    "pairable": r.kind != "score" and not r.ablation})
        t["rows"] += 1
        g = _gold(r)
        t["labels"][g] = t["labels"].get(g, 0) + 1
        t["paired"] += bool(r.pair_id)
    for t in out.values():
        top = max(t["labels"].values()) / t["rows"]
        t["majority_share"] = round(top, 3)
        t["paired_share"] = round(t["paired"] / t["rows"], 3)
        t["warn"] = top > 0.6 or (t["pairable"] and t["paired_share"] < 0.3)
    return out


def split_by_source(
    rows: list[VisionExample],
    val_fraction: float,
    *,
    held_out_tasks: Iterable[str] = (),
    seed: int = 0,
) -> tuple[list[VisionExample], list[VisionExample], list[VisionExample]]:
    """Train / val / held-out, never splitting rows that share an image, pair or source.

    Rows are joined into units through shared image paths, pair ids and source ids, so
    a pair's two halves, every question about one image, and an ablation copy all land
    on the same side. Units are assigned to val per task (by each unit's first row) so
    small tasks still appear in val. Tasks named in `held_out_tasks` (and their
    ablation copies) go to the third list whole: the test of unseen task families.
    """
    hold = set(held_out_tasks)
    held = [r for r in rows if r.task.split("/ablation")[0] in hold]
    rest = [r for r in rows if r.task.split("/ablation")[0] not in hold]

    parent = list(range(len(rest)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for i, r in enumerate(rest):
        keys = [f"img:{p}" for p in r.images]
        if r.pair_id:
            keys.append(f"pair:{r.pair_id}")
        keys.append(f"src:{_source(r)}")
        for k in keys:
            if k in owner:
                parent[find(i)] = find(owner[k])
            else:
                owner[k] = i

    units: dict[int, list[VisionExample]] = {}
    for i, r in enumerate(rest):
        units.setdefault(find(i), []).append(r)
    by_task: dict[str, list[list[VisionExample]]] = {}
    for u in units.values():
        by_task.setdefault(u[0].task.split("/ablation")[0], []).append(u)

    rng = random.Random(seed)
    train: list[VisionExample] = []
    val: list[VisionExample] = []
    for task in sorted(by_task):
        us = by_task[task]
        rng.shuffle(us)
        cut = int(len(us) * val_fraction)
        val.extend(r for u in us[:cut] for r in u)
        train.extend(r for u in us[cut:] for r in u)
    rng.shuffle(train)
    return train, val, held


# ---- converters ---------------------------------------------------------------------

def from_docvqa_yesno(records: Iterable[dict[str, Any]], image_root: str) -> list[VisionExample]:
    """DocVQA rows whose answers are yes/no become noul questions about the page.

    `records` are DocVQA-style dicts: {"image": "...png", "question": "...",
    "answers": ["Yes"]}. Anything not answered yes/no is skipped. DocVQA has very few of
    these (about 101 of 44,812 train+val questions); its open answers belong to a choice
    converter that offers the gold plus hard negatives.
    """
    out = []
    for rec in records:
        ans = {a.strip().lower() for a in rec.get("answers", [])}
        if not ans or ans - {"yes", "no"} or len(ans) != 1:
            continue
        image = str(Path(image_root) / rec["image"])
        out.append(VisionExample(
            kind="noul",
            state="",
            instructions=rec["question"].strip(),
            options=[["false", "the statement does not hold for this page"],
                     ["true", "the statement holds for this page"]],
            label=1 if "yes" in ans else 0,
            task="docvqa/yesno",
            images=[image],
            source_id=image,
        ))
    return out


def from_labelled_images(
    records: Iterable[dict[str, Any]],
    label_descriptions: dict[str, str],
    *,
    task: str,
    instructions: str,
    n_options: int | None = None,
    image_root: str = "",
    seed: int = 0,
) -> list[VisionExample]:
    """Any image classification set becomes choice rows.

    `n_options=None` shows the whole label set on every row: one fixed option set per
    task, so rows with different gold labels pair up (`make_pairs`). An integer shows
    the gold plus `n_options - 1` sampled distractors, which teaches reading option text
    over larger label spaces but forms no pairs. Option order is reshuffled at collate
    time either way.
    """
    rng = random.Random(seed)
    names = list(label_descriptions)
    out = []
    for rec in records:
        gold = rec["label"]
        if gold not in label_descriptions:
            continue
        if n_options is None or n_options >= len(names):
            shown = list(names)
        else:
            pool = [n for n in names if n != gold]
            shown = [gold] + rng.sample(pool, n_options - 1)
            rng.shuffle(shown)
        image = str(Path(image_root) / rec["image"]) if image_root else rec["image"]
        out.append(VisionExample(
            kind="choice",
            state=rec.get("state", ""),
            instructions=instructions,
            options=[[n, label_descriptions[n]] for n in shown],
            label=shown.index(gold),
            task=task,
            images=[image],
            source_id=image,
        ))
    return out


def from_ordinal_ratings(
    records: Iterable[dict[str, Any]], levels: list[str], *, task: str, instructions: str,
    image_root: str = "",
) -> list[VisionExample]:
    """Ratings on an ordered scale become score rows; `levels[i]` is the rubric for i."""
    out = []
    for rec in records:
        lvl = int(rec["level"])
        if not 0 <= lvl < len(levels):
            continue
        image = str(Path(image_root) / rec["image"]) if image_root else rec["image"]
        out.append(VisionExample(
            kind="score",
            state=rec.get("state", ""),
            instructions=instructions,
            options=[[str(i), d] for i, d in enumerate(levels)],
            label=lvl,
            task=task,
            images=[image],
            source_id=image,
        ))
    return out
