"""Phase 1 baselines: how well do existing models decide over images, before we train?

Systems (each also run "blind", with the image removed, to measure image dependence):
  qwen-untrained   Qwen3.5-2B-Base read the Decider way: the prompt rendered as Strands
                   Decider renders it, image inside <state>, scored by the frozen torso's
                   own option-number logits at <answer> (`frozen_slot_log_probs`). This is
                   the starting point a Vision Decider trains from.
  mapika           Mapika/decider-2b-vision through its own `VisionDecisionModel`
                   (letter logits at an answer slot, its own prompt, images <= 768 px).

Benchmarks: NaturalBench (2 images x 2 questions per group, built so blind models score at
chance) and POPE adversarial (object presence on COCO val2014). Image JevBench's public
items are not downloadable, so it is not run here.

Results stream to <out>/<system>.jsonl as they are produced; <out>/summary.json is
rewritten after every system, so a run cut short still leaves usable numbers.

    python -m phase1.run --out results --nb-groups 500 --pope 1000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import traceback
from typing import Any, Iterator

import torch
from PIL import Image

from phase1.metrics import image_dependence, naturalbench_paired, summarise

NB_REPO, NB_FILE = "BaiqiL/NaturalBench", "data/train-00000-of-00003.parquet"
POPE_REPO, POPE_FILE = "lmms-lab/POPE", "Full/adversarial-00000-of-00001.parquet"
MAPIKA = "Mapika/decider-2b-vision"
# NaturalBench's fixed answer pattern (its official scorer hard-codes it too):
# question k on image j -> column Image_j_Question_k.


# ---- data ------------------------------------------------------------------------------

def _parquet(repo: str, filename: str, local: str | None):
    import pandas as pd
    if local:
        return pd.read_parquet(local)
    from huggingface_hub import hf_hub_download
    return pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset"))


def _mc(question: str) -> tuple[str, list[tuple[str, str]]]:
    """Options out of a NaturalBench question. Shapes seen:
    'stem?\nOption: A:To the right; B:To the left;'
    'stem?\nOption: A: Noticeably curved. B: Slightly curved.'
    'stem? A. Nothing happens. B. Something moves.'
    Markers are taken in sequence (A, then B, ...) so a stray capital in an option
    is never read as a new one.
    """
    region_start = question.find("Option:")
    region_start = region_start + len("Option:") if region_start >= 0 else 0
    marks: list[tuple[str, int, int]] = []
    want = "A"
    for m in re.finditer(r"(?:(?<=\s)|(?<=^)|(?<=;)|(?<=:))([A-H])\s*[:.]\s*", question[region_start:]):
        if m.group(1) == want:
            marks.append((want, region_start + m.start(), region_start + m.end()))
            want = chr(ord(want) + 1)
    if len(marks) < 2:
        raise ValueError(f"cannot parse options from {question!r}")
    stem = question[: question.find("Option:")] if "Option:" in question else question[: marks[0][1]]
    opts = []
    for k, (name, _, end) in enumerate(marks):
        stop = marks[k + 1][1] if k + 1 < len(marks) else len(question)
        opts.append((name, question[end:stop].strip().rstrip(";").strip()))
    return stem.strip(), opts


def naturalbench(n_groups: int, local: str | None) -> Iterator[dict[str, Any]]:
    df = _parquet(NB_REPO, NB_FILE, local).head(n_groups)
    for _, row in df.iterrows():
        imgs = [row["Image_0"]["bytes"], row["Image_1"]["bytes"]]
        for k in (0, 1):
            q = row[f"Question_{k}"]
            mc = row["Question_Type"] == "multiple_choice"
            for j in (0, 1):
                gold = str(row[f"Image_{j}_Question_{k}"]).strip()
                if mc:
                    stem, opts = _mc(q)
                    names = [n for n, _ in opts]
                    item = {"kind": "choice", "question": stem, "options": opts, "gold": names.index(gold[0])}
                else:
                    item = {"kind": "noul", "question": q.strip(), "options": [("no", ""), ("yes", "")],
                            "gold": 1 if gold.lower().startswith("y") else 0}
                yield {**item, "id": f"nb-{row['Index']}-q{k}-i{j}", "bench": "naturalbench",
                       "group": int(row["Index"]), "q": k, "i": j, "image": imgs[j]}


def pope(n: int, local: str | None) -> Iterator[dict[str, Any]]:
    df = _parquet(POPE_REPO, POPE_FILE, local)
    # Interleave yes/no so a truncated run stays balanced.
    df = df.sort_values("question_id").head(n)
    for _, row in df.iterrows():
        yield {"id": f"pope-{row['question_id']}", "bench": "pope_adversarial", "kind": "noul",
               "question": row["question"].strip(), "options": [("no", ""), ("yes", "")],
               "gold": 1 if row["answer"].strip().lower() == "yes" else 0, "image": row["image"]["bytes"]}


def _pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


# ---- systems ---------------------------------------------------------------------------

class QwenUntrained:
    """Qwen3.5-2B-Base, no adapter, no trained head: the frozen option-number readout."""

    name = "qwen-untrained"

    def __init__(self, base: str, dtype: str, long_side: int):
        from transformers import AutoProcessor
        from vision_decider.collate import VisionCollator, VisionCollatorConfig
        from vision_decider.modeling import VisionDeciderConfig, VisionDeciderModel

        cfg = VisionDeciderConfig(base_model=base, use_lora=False, torch_dtype=dtype,
                                  image_long_side=long_side, max_length=4096)
        self.model = VisionDeciderModel.from_pretrained_base(cfg).eval()
        self.coll = VisionCollator(AutoProcessor.from_pretrained(base), VisionCollatorConfig(
            num_slots=24, max_length=4096, head_type="pointer", image_long_side=long_side,
            max_image_tokens=4096), train=False)

    @torch.no_grad()
    def probs(self, items: list[dict[str, Any]], blind: bool) -> list[list[float]]:
        from vision_decider.data import VisionExample

        rows = []
        for it in items:
            if it["kind"] == "noul":
                opts = [["false", "the answer is no"], ["true", "the answer is yes"]]
            else:
                opts = [[n, d] for n, d in it["options"]]
            rows.append(VisionExample(kind=it["kind"], state="", instructions=it["question"], options=opts,
                                      label=it["gold"], task=it["bench"],
                                      images=[] if blind else [_pil(it["image"])]))
        b = self.coll(rows)
        if b is None or b["row_index"].numel() != len(rows):
            raise RuntimeError("collator dropped rows; raise max_length")
        lp, _ = self.model.frozen_slot_log_probs(
            b["input_ids"], b["attention_mask"], b["n_slots"],
            pixel_values=b.get("pixel_values"), image_grid_thw=b.get("image_grid_thw"))
        # Slot order follows the option order given (train=False: no shuffle); noul is
        # false/true, matching the no/yes order of the gold index.
        return [lp[r, : len(it["options"])].exp().tolist() for r, it in enumerate(items)]


class Mapika:
    """Mapika/decider-2b-vision through its own code (one image per item, <= 10 options)."""

    name = "mapika"

    def __init__(self, dtype: str):
        from huggingface_hub import snapshot_download
        repo = snapshot_download(MAPIKA)
        sys.path.insert(0, repo)
        from decider.infer import Example, Q  # type: ignore
        from decider.vision import VisionDecisionModel  # type: ignore

        self.Example, self.Q = Example, Q
        self.m = VisionDecisionModel(repo, dtype=getattr(torch, dtype), grad_ckpt=False).eval()

    @torch.no_grad()
    def probs(self, items: list[dict[str, Any]], blind: bool) -> list[list[float]]:
        exs = []
        for it in items:
            img = None
            if not blind:
                img = _pil(it["image"])
                img.thumbnail((768, 768))  # its training images were <= 768 px
            if it["kind"] == "noul":
                opts = ["no", "yes"]  # the order its yes/no training rows used
            else:
                opts = [f"{n}: {d}" for n, d in it["options"]]
            exs.append((img, self.Example("This is a visual question about the image.",
                                          [self.Q(it["question"], opts, 0)])))
        inp = self.m.prepare(exs)
        lg = self.m.slot_logits(inp).float()
        return [torch.softmax(r, -1)[: len(it["options"])].tolist() for r, it in zip(lg, items)]


# ---- driver ----------------------------------------------------------------------------

def run_system(system: Any, items: list[dict[str, Any]], out: str, batch: int, log_every: int = 50) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for blind in (False, True):
        tag = f"{system.name}{'-blind' if blind else ''}"
        path = os.path.join(out, f"{tag}.jsonl")
        done = {}
        if os.path.exists(path):  # resume
            with open(path) as fh:
                done = {json.loads(l)["id"]: json.loads(l) for l in fh if l.strip()}
        todo = [it for it in items if it["id"] not in done]
        t0 = time.time()
        with open(path, "a") as fh:
            for s in range(0, len(todo), batch):
                chunk = todo[s:s + batch]
                for it, p in zip(chunk, system.probs(chunk, blind), strict=True):
                    r = {k: it[k] for k in ("id", "bench", "kind", "gold") if k in it}
                    r.update({k: it[k] for k in ("group", "q", "i", "dataset", "exact", "official_mapika_ok")
                              if k in it}, probs=p)
                    fh.write(json.dumps(r) + "\n")
                    done[it["id"]] = r
                fh.flush()
                n = s + len(chunk)
                if n % log_every < batch or n == len(todo):
                    rate = n / max(1e-6, time.time() - t0)
                    print(f"[phase1] {tag}: {n}/{len(todo)} {rate:.2f} it/s "
                          f"eta {(len(todo) - n) / max(rate, 1e-6) / 60:.0f} min", flush=True)
        res[tag] = [done[it["id"]] for it in items if it["id"] in done]
    return res


def score(all_res: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag, rs in all_res.items():
        by_bench: dict[str, list[dict[str, Any]]] = {}
        for r in rs:
            by_bench.setdefault(r["bench"], []).append(r)
        out[tag] = {b: summarise(v) for b, v in by_bench.items()}
        nb = by_bench.get("naturalbench")
        if nb:
            out[tag]["naturalbench"]["paired"] = naturalbench_paired(nb)
        ijb = by_bench.get("ijb_preview")
        if ijb:
            block = out[tag]["ijb_preview"]
            block["exact_only"] = summarise([r for r in ijb if r.get("exact")])["all"]
            per: dict[str, list[dict[str, Any]]] = {}
            for r in ijb:
                per.setdefault(r["dataset"], []).append(r)
            block["by_dataset"] = {d: summarise(v)["all"] for d, v in sorted(per.items())}
            if tag == "mapika":
                # Our harness against the official per-item log: does it reproduce them?
                pred_ok = [max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["gold"] for r in ijb]
                agree = [a == r["official_mapika_ok"] for a, r in zip(pred_ok, ijb)]
                ex = [g for g, r in zip(agree, ijb) if r.get("exact")]
                block["official_agreement"] = {
                    "ours_correct": sum(pred_ok), "official_correct": sum(r["official_mapika_ok"] for r in ijb),
                    "item_agreement": round(sum(agree) / len(agree), 4),
                    "item_agreement_exact": round(sum(ex) / len(ex), 4) if ex else None}
    for tag in list(all_res):
        if not tag.endswith("-blind") and f"{tag}-blind" in all_res:
            out[tag]["image_dependence"] = image_dependence(all_res[tag], all_res[f"{tag}-blind"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nb-groups", type=int, default=500)
    ap.add_argument("--pope", type=int, default=1000)
    ap.add_argument("--systems", default="qwen,mapika")
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--long-side", type=int, default=448)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--nb-local"), ap.add_argument("--pope-local")
    ap.add_argument("--ijb-jsonl", help="Image JevBench preview items built by phase1.ijb_preview")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"[phase1] torch {torch.__version__} threads {torch.get_num_threads()}", flush=True)

    items = (list(naturalbench(a.nb_groups, a.nb_local)) if a.nb_groups else []) + \
        (list(pope(a.pope, a.pope_local)) if a.pope else [])
    if a.ijb_jsonl:
        from phase1.ijb_preview import load as load_ijb
        items += load_ijb(a.ijb_jsonl)
    print(f"[phase1] {len(items)} items "
          f"({sum(i['bench'] == 'naturalbench' for i in items)} NaturalBench, "
          f"{sum(i['bench'] == 'pope_adversarial' for i in items)} POPE, "
          f"{sum(i['bench'] == 'ijb_preview' for i in items)} Image JevBench preview)", flush=True)
    meta = {"args": vars(a), "items": len(items), "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    all_res: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    for name in a.systems.split(","):
        try:
            t0 = time.time()
            system = QwenUntrained(a.base, a.dtype, a.long_side) if name == "qwen" else Mapika(a.dtype)
            all_res.update(run_system(system, items, a.out, a.batch))
            meta[f"{name}_minutes"] = round((time.time() - t0) / 60, 1)
            del system
        except Exception:
            errors[name] = traceback.format_exc()
            print(f"[phase1] {name} FAILED\n{errors[name]}", flush=True)
        with open(os.path.join(a.out, "summary.json"), "w") as fh:
            json.dump({"meta": meta, "errors": errors, "scores": score(all_res)}, fh, indent=2)
    print(json.dumps(score(all_res), indent=2), flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
