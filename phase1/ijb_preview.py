"""Local reconstruction of the 128 public "Image JevBench multimodal preview" items.

The item definitions (question, options, gold, source row, transformation) are public in
github.com/fstandhartinger/model-market-comparison (MIT), file
data/raw/benchmarks/jevbench/multimodal-preview/source/items-extended-real.json. The item
images (assets/items/*.webp) are not published, and neither is the code that made them, so
this module rebuilds each image from its upstream Hugging Face row:

  CLEVR-HOPE, Geometry3K, ArxivQA  the source image, unchanged             -> exact
  FinQA                            source table re-rendered as text lines   -> approximate
  ScreenSpot                       five labelled click markers drawn        -> approximate
  Multimodal-Mind2Web              positive + 4 negatives boxed, cropped    -> approximate

"exact" means image pixels, question and options all come straight from the source with no
rendering choice of ours. The official run used WebP copies of the images (assets/items/*.webp);
whether those were lossy or resized is not public, so even "exact" items may differ from what
the official systems saw by compression only.

Rendering choices for the approximate datasets were fitted to the public example images in
the same repo (public/image-jev/examples/*.webp, from a later revision of the same builder):
  * finqa-net-revenue.webp (mm-061): 640 px wide, one "cell | cell" line per table row,
    ~14 px sans font, 22 px line pitch, x = 12.
  * mobile-translate.webp (mm-118), windows-copy.webp, browser-new-tab.webp: markers are
    white discs with a red ring and a black letter; distractor points are taken in order from
    (0.12,0.16), (0.82,0.17), (0.16,0.78), (0.82,0.80), (0.50,0.50), ... skipping points too
    near the target; the point list [gold, d0..d3] is rotated so gold lands on the item's gold
    letter (fits all four public examples).
No public reference exists for the Mind2Web rendering, so those six-plus-six items use a
documented guess (see _mind2web_image).

Torch-free: needs huggingface_hub, pyarrow, Pillow.

    python -m phase1.ijb_preview --out /tmp/ijb/out --cache /tmp/ijb/cache
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import math
import os
import re
import sys
import time
import urllib.request
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ITEMS_URL = ("https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/main/"
             "data/raw/benchmarks/jevbench/multimodal-preview/source/items-extended-real.json")
LOG_URL = ("https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/main/"
           "data/raw/benchmarks/jevbench/multimodal-preview/source/results-decider-real.log")

# dataset -> (HF repo, pinned revision, file glob in split order, columns to read)
# Revisions for ScreenSpot / FinQA / Geometry3K are the ones the site repo's example manifest
# records; the others are the revisions current when this module was written.
SOURCES: dict[str, tuple[str, str, str, list[str]]] = {
    "CLEVR-HOPE": ("user9000/CLEVR-HOPE", "2d0c19f23dc5de19f83f4e9928a793c95632327a",
                   "HOP00/test_complex_ood-*.parquet", ["idx", "query", "answer", "image"]),
    "Geometry3K": ("hiyouga/geometry3k", "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7",
                   "data/test-*.parquet", ["problem", "answer", "images"]),
    "ArxivQA": ("mm-eval/ArxivQA", "e13bb29ff5b2c2ae5f05cc0b1ecc2b24a6833c81",
                "data/train-*.parquet", ["id", "messages", "answer", "media"]),
    "FinQA": ("bevaya/FinQA", "3d6a736bc67e06bc15fbf3618d88204a57c5b25e",
              "data/test-*.parquet", ["id", "table", "question", "answer"]),
    "ScreenSpot": ("bevaya/ScreenSpot", "0be08781e2e188582f6131625ae1598d443b4d5d",
                   "data/test-*.parquet", ["file_name", "bbox", "instruction", "data_type", "data_source", "image"]),
    "Multimodal-Mind2Web": ("osunlp/Multimodal-Mind2Web", "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889",
                            "data/test_task-*.parquet",
                            ["action_uid", "operation", "pos_candidates", "neg_candidates", "website",
                             "confirmed_task", "screenshot"]),
}
EXACT_DATASETS = {"CLEVR-HOPE", "Geometry3K", "ArxivQA"}
LETTERS = "ABCDE"


# ---- source rows ------------------------------------------------------------------------

class _RowReader:
    """Reads single rows out of remote parquet shards with HTTP range requests: only the
    footer of each shard and the one row group (selected columns) holding the row."""

    def __init__(self) -> None:
        from huggingface_hub import HfFileSystem
        self.fs = HfFileSystem()
        self._shards: dict[str, list[tuple[str, list[int]]]] = {}
        self._groups: dict[tuple[str, int], list[dict[str, Any]]] = {}

    def _layout(self, dataset: str) -> list[tuple[str, list[int]]]:
        if dataset not in self._shards:
            import pyarrow.parquet as pq
            repo, rev, pattern, _ = SOURCES[dataset]
            root = f"datasets/{repo}@{rev}/"
            paths = sorted(p for p in self.fs.glob(root + "**/*.parquet")
                           if fnmatch.fnmatch(p[len(root):] if p.startswith(root) else p.split(repo + "/", 1)[-1], pattern))
            layout = []
            for p in paths:
                p = p if p.startswith(root) else root + p.split(repo + "/", 1)[-1]
                with self.fs.open(p, "rb", block_size=1 << 20) as f:
                    md = pq.ParquetFile(f).metadata
                    layout.append((p, [md.row_group(i).num_rows for i in range(md.num_row_groups)]))
            self._shards[dataset] = layout
        return self._shards[dataset]

    def row(self, dataset: str, index: int) -> dict[str, Any]:
        import pyarrow.parquet as pq
        cols = SOURCES[dataset][3]
        base = 0
        for path, groups in self._layout(dataset):
            for g, n in enumerate(groups):
                if index < base + n:
                    key = (path, g)
                    if key not in self._groups:
                        with self.fs.open(path, "rb", block_size=8 << 20) as f:
                            self._groups = {k: v for k, v in self._groups.items() if k[0] == path}  # bound memory
                            self._groups[key] = pq.ParquetFile(f).read_row_group(g, columns=cols).to_pylist()
                    return self._groups[key][index - base]
                base += n
        raise IndexError(f"{dataset} row {index} out of range ({base} rows)")


def _image_bytes(dataset: str, row: dict[str, Any]) -> bytes:
    if dataset == "CLEVR-HOPE":
        return row["image"]["bytes"]
    if dataset == "Geometry3K":
        return row["images"][0]["bytes"]
    if dataset == "ArxivQA":
        return row["media"][0]["bytes"]
    if dataset == "ScreenSpot":
        return row["image"]["bytes"]
    if dataset == "Multimodal-Mind2Web":
        return row["screenshot"]["bytes"]
    raise KeyError(dataset)


def _cached_source(reader: _RowReader | None, cache_dir: str, item: dict[str, Any]) -> tuple[dict[str, Any], bytes | None]:
    """(row fields without the image, image bytes) for an item, cached on disk."""
    ds, idx = item["dataset"], int(item["source_row"])
    d = os.path.join(cache_dir, "rows", re.sub(r"[^A-Za-z0-9]+", "-", ds))
    os.makedirs(d, exist_ok=True)
    jpath, ipath = os.path.join(d, f"{idx}.json"), os.path.join(d, f"{idx}.img")
    if os.path.exists(jpath):
        with open(jpath) as f:
            fields = json.load(f)
        img = open(ipath, "rb").read() if os.path.exists(ipath) else None
        return fields, img
    if reader is None:
        raise LookupError("not cached")
    row = reader.row(ds, idx)
    img = None if ds == "FinQA" else _image_bytes(ds, row)
    fields = {k: v for k, v in row.items() if k not in ("image", "images", "media", "screenshot")}
    with open(jpath, "w") as f:
        json.dump(fields, f)
    if img is not None:
        with open(ipath, "wb") as f:
            f.write(img)
    return fields, img


# ---- rendering --------------------------------------------------------------------------

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf", "Arial.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def _cap(img: Image.Image, long_side: int = 1200) -> Image.Image:
    """Published ScreenSpot items are the source screenshot scaled to <= 1200 px on the long
    side (mm-118: 2360x1640 source -> 1200x834); smaller screenshots are left as they are."""
    w, h = img.size
    s = long_side / max(w, h)
    if s >= 1:
        return img
    return img.resize((round(w * s), round(h * s)), Image.LANCZOS)


def _finqa_image(table: list[list[str]]) -> Image.Image:
    """One 'cell | cell | ...' line per table row, black on white, 640 px wide, 22 px pitch.
    Fitted to public/image-jev/examples/finqa-net-revenue.webp (mm-061, 640x228, 9 rows).
    The original font is unknown: an Arial-like 11 px face with spaces ~0.4 of normal width
    matches the example's line widths to within a few px. Lines longer than the canvas widen
    the canvas (the original's handling of long lines is unknown)."""
    font = _font(11)
    sp = font.getlength(" ") * 0.4
    lines = [" | ".join(str(c) for c in row).split(" ") for row in table]

    def width_of(words):
        return sum(font.getlength(w) for w in words) + sp * (len(words) - 1)

    width = max([640] + [int(math.ceil(width_of(ws))) + 24 for ws in lines])
    img = Image.new("RGB", (width, 30 + 22 * len(lines)), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i, words in enumerate(lines):
        x = 12.0
        for w in words:
            if w:
                d.text((x, 22 + 22 * i), w, fill=(0, 0, 0), font=font, anchor="ls")
            x += font.getlength(w) + sp
    return img


# Distractor click points (fractions of width/height), in the order the public examples use
# them; the first five are confirmed by public/image-jev/examples, the rest are our fallback.
_MARKER_POINTS = [(0.12, 0.16), (0.82, 0.17), (0.16, 0.78), (0.82, 0.80), (0.50, 0.50),
                  (0.50, 0.12), (0.50, 0.88), (0.12, 0.50), (0.88, 0.50), (0.30, 0.30),
                  (0.70, 0.70), (0.30, 0.70), (0.70, 0.30)]


def _marker_points(w: int, h: int, bbox: list[float], gold_letter: str) -> list[tuple[float, float]]:
    """Five pixel points, index = letter. Gold = bbox centre. Distractors: the fixed list,
    skipping points inside the bbox or within 0.1 * max(w, h) of the gold point (fits all four
    public examples; browser-new-tab drops (0.82, 0.17) at 75 px, windows-copy keeps a point at
    203 px). [gold, d0, d1, d2, d3] is rotated so gold sits on its letter - this reproduces
    the letter order of all four public examples."""
    x0, y0, x1, y1 = bbox
    gx, gy = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h
    near = 0.1 * max(w, h)
    ds: list[tuple[float, float]] = []
    for fx, fy in _MARKER_POINTS:
        if x0 <= fx <= x1 and y0 <= fy <= y1:
            continue
        px, py = fx * w, fy * h
        if math.hypot(px - gx, py - gy) < near:
            continue
        ds.append((px, py))
        if len(ds) == 4:
            break
    pts = [(gx, gy)] + ds
    g = LETTERS.index(gold_letter)
    return [pts[(i - g) % 5] for i in range(5)]


def _draw_marker(d: ImageDraw.ImageDraw, x: float, y: float, r: int, letter: str, font: Any) -> None:
    stroke = 4
    d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255), outline=(178, 12, 12), width=stroke)
    d.text((x, y), letter, fill=(0, 0, 0), font=font, anchor="mm")


def _screenspot_image(src: bytes, bbox: list[float], gold_letter: str) -> Image.Image:
    """Markers drawn at source resolution, then the image is capped at 1200 px. Fitted to the
    public examples: ring radius ~2.2% of the short side (12 px on 960x540, ~36 px on the
    2360x1640 source of mm-118), fixed ~3 px red ring and ~16 px black letter at source scale
    (which is why the letters look tiny on downscaled iPad shots), white disc."""
    img = _to_rgb(Image.open(io.BytesIO(src)))
    w, h = img.size
    r = max(12, round(0.022 * min(w, h)))
    font = _font(16)
    d = ImageDraw.Draw(img)
    for letter, (x, y) in zip(LETTERS, _marker_points(w, h, bbox, gold_letter)):
        _draw_marker(d, x, y, r, letter, font)
    return _cap(img)


def _m2w_box(cand: str) -> tuple[float, float, float, float] | None:
    c = json.loads(cand)
    attrs = json.loads(c.get("attributes") or "{}")
    rect = attrs.get("bounding_box_rect")
    if not rect:
        return None
    try:
        x, y, w, h = (float(v) for v in rect.split(","))
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, x + w, y + h)


_BOX_COLORS = [(220, 20, 60), (30, 110, 230), (20, 150, 60), (230, 120, 0), (150, 40, 200)]


def _mind2web_image(src: bytes, row: dict[str, Any], gold_letter: str) -> tuple[Image.Image, dict[str, Any]]:
    """GUESS (no public reference image or code): crop a 1280-wide x <=1000-tall window of the
    full-page screenshot centred on the positive element; pick the four dataset negatives whose
    boxes lie fully in the window, are 12..(40% of window) in size, do not overlap the positive
    much and are nearest to it; draw all five as coloured outlines with a letter tag; letters
    assigned like ScreenSpot ([positive, n0..n3] rotated onto the gold letter); cap at 1200 px."""
    img = _to_rgb(Image.open(io.BytesIO(src)))
    W, H = img.size
    pos = _m2w_box(row["pos_candidates"][0])
    assert pos is not None, "positive candidate has no bbox"
    ch = min(H, 1000)
    pcx, pcy = (pos[0] + pos[2]) / 2, (pos[1] + pos[3]) / 2
    top = int(min(max(0, pcy - ch / 2), H - ch))
    win = (0, top, W, top + ch)

    def inside(b):
        return b[0] >= win[0] and b[1] >= win[1] and b[2] <= win[2] and b[3] <= win[3]

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

    negs = []
    for k, c in enumerate(row["neg_candidates"]):
        b = _m2w_box(c)
        if b is None or not inside(b):
            continue
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bw < 12 or bh < 12 or bw > 0.4 * W or bh > 0.4 * ch:
            continue
        if iou(b, pos) > 0.1 or (b[0] <= pcx <= b[2] and b[1] <= pcy <= b[3]):
            continue
        if any(iou(b, o) > 0.3 for _, o in negs):
            continue
        negs.append((k, b))
    negs.sort(key=lambda kb: (math.hypot((kb[1][0] + kb[1][2]) / 2 - pcx, (kb[1][1] + kb[1][3]) / 2 - pcy), kb[0]))
    negs = negs[:4]
    if len(negs) < 4:
        raise ValueError(f"only {len(negs)} usable negatives")
    boxes = [pos] + [b for _, b in negs]
    g = LETTERS.index(gold_letter)
    order = [boxes[(i - g) % 5] for i in range(5)]
    crop = img.crop(win)
    d = ImageDraw.Draw(crop)
    font = _font(16)
    for i, (letter, b) in enumerate(zip(LETTERS, order)):
        x0, y0, x1, y1 = b[0] - win[0], b[1] - win[1], b[2] - win[0], b[3] - win[1]
        col = _BOX_COLORS[i]
        d.rectangle((x0, y0, x1, y1), outline=col, width=3)
        tw = 18
        ty = y0 - 20 if y0 >= 20 else y0
        d.rectangle((x0, ty, x0 + tw, ty + 20), fill=col)
        d.text((x0 + tw / 2, ty + 10), letter, fill=(255, 255, 255), font=font, anchor="mm")
    meta = {"crop_xyxy": list(win), "negative_indices": [k for k, _ in negs]}
    return _cap(crop), meta


# ---- alignment checks -------------------------------------------------------------------

def _num(s: str) -> float | None:
    m = re.fullmatch(r"\s*\$?\s*(-?[\d,]*\.?\d+)\s*%?\s*", str(s))
    return float(m.group(1).replace(",", "")) if m else None


def _same_answer(a: str, b: str) -> bool:
    if str(a).strip().lower() == str(b).strip().lower():
        return True
    x, y = _num(a), _num(b)
    return x is not None and y is not None and abs(x - y) <= 1e-6 * max(1.0, abs(x))


def check(item: dict[str, Any], row: dict[str, Any]) -> list[str]:
    """Mismatches between an item and its source row (empty list = aligned)."""
    ds, rub = item["dataset"], item["rubric"]
    q, crit, gold_text = rub["instructions"], rub["criteria"], rub["criteria"][item["gold"]]
    bad: list[str] = []
    if ds == "CLEVR-HOPE":
        if row["query"].strip().rstrip("?").strip() != q.strip():
            bad.append(f"question: {row['query']!r}")
        if row["answer"].strip().lower() != item["gold"]:
            bad.append(f"answer {row['answer']!r} vs gold {item['gold']!r}")
    elif ds == "Geometry3K":
        if row["problem"].replace("<image>", "").strip() != q.strip():
            bad.append(f"question: {row['problem']!r}")
        if not _same_answer(row["answer"], gold_text):
            bad.append(f"answer {row['answer']!r} vs gold option {gold_text!r}")
    elif ds == "ArxivQA":
        m = json.loads(row["messages"])[0]
        if m["question"].strip() != q.strip():
            bad.append(f"question: {m['question']!r}")
        if list(m.get("options", {}).items()) != list(crit.items()):
            bad.append(f"options: {m.get('options')!r}")
        if row["answer"].strip() != item["gold"]:
            bad.append(f"answer {row['answer']!r} vs gold {item['gold']!r}")
    elif ds == "FinQA":
        if row["question"].strip() != q.strip():
            bad.append(f"question: {row['question']!r}")
        if not _same_answer(row["answer"], gold_text) and not _same_answer(row["answer"], item["gold"]):
            bad.append(f"answer {row['answer']!r} vs gold option {gold_text!r}")
    elif ds == "ScreenSpot":
        lab = item["source_label"]
        if row["file_name"] != lab["file_name"]:
            bad.append(f"file {row['file_name']}")
        if any(abs(a - b) > 1e-9 for a, b in zip(row["bbox"], lab["bbox_xyxy"])):
            bad.append(f"bbox {row['bbox']}")
        if row["instruction"] != lab["instruction"] or f"Goal: {row['instruction']} " not in q:
            bad.append(f"instruction {row['instruction']!r}")
    elif ds == "Multimodal-Mind2Web":
        lab = item["source_label"]
        if row["action_uid"] != lab["action_uid"]:
            bad.append(f"action_uid {row['action_uid']}")
        if json.loads(row["pos_candidates"][0])["backend_node_id"] != lab["positive_backend_node_id"]:
            bad.append("positive backend_node_id")
        if row["operation"] != lab["operation"]:
            bad.append(f"operation {row['operation']}")
        if row["confirmed_task"].strip() not in q:
            bad.append(f"task {row['confirmed_task']!r}")
    return bad


# ---- build / load -----------------------------------------------------------------------

def _fetch_text(url: str, fallback: str | None) -> str:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read().decode("utf-8")
    except Exception:
        if fallback and os.path.exists(fallback):
            with open(fallback, encoding="utf-8") as f:
                return f.read()
        raise


def build(out_dir: str, cache_dir: str, items_path: str | None = None, log_path: str | None = None,
          verbose: bool = True) -> str:
    """Rebuild all 128 items into <out_dir>/images/*.png and <out_dir>/ijb_preview.jsonl.
    items_path / log_path are local fallbacks for the GitHub raw files."""
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "hf"))
    items = json.loads(_fetch_text(ITEMS_URL, items_path))
    log = _fetch_text(LOG_URL, log_path)
    official = {m.group(1): m.group(2) == "ok" for m in re.finditer(r"^(mm-\d+) (ok|miss)\S*$", log, re.M)}
    reader: _RowReader | None = None
    out_path = os.path.join(out_dir, "ijb_preview.jsonl")
    t0, problems = time.time(), []
    with open(out_path, "w", encoding="utf-8") as out:
        for n, item in enumerate(items):
            ds = item["dataset"]
            try:
                try:
                    row, src = _cached_source(None, cache_dir, item)
                except LookupError:
                    reader = reader or _RowReader()
                    row, src = _cached_source(reader, cache_dir, item)
                mismatches = check(item, row)
                extra: dict[str, Any] = {}
                if ds in EXACT_DATASETS:
                    img = _to_rgb(Image.open(io.BytesIO(src)))
                elif ds == "FinQA":
                    img = _finqa_image(row["table"])
                elif ds == "ScreenSpot":
                    img = _screenspot_image(src, item["source_label"]["bbox_xyxy"], item["gold"])
                else:
                    img, extra = _mind2web_image(src, row, item["gold"])
            except Exception as e:  # keep going; report at the end
                problems.append((item["id"], f"{type(e).__name__}: {e}"))
                if verbose:
                    print(f"[ijb] {item['id']} FAILED: {e}", file=sys.stderr)
                continue
            if mismatches:
                problems.append((item["id"], "; ".join(mismatches)))
            ipath = os.path.join(out_dir, "images", f"{item['id']}.png")
            img.save(ipath, optimize=False)
            names = list(item["rubric"]["criteria"])
            rec = {
                "id": item["id"], "bench": "ijb_preview", "dataset": ds, "kind": "choice",
                "question": item["rubric"]["instructions"],
                "options": [[k, v] for k, v in item["rubric"]["criteria"].items()],
                "gold": names.index(item["gold"]), "image_path": os.path.abspath(ipath),
                "exact": ds in EXACT_DATASETS and not mismatches,
                "official_mapika_ok": official.get(item["id"]),
                "source_row": item["source_row"], "image_size": list(img.size),
                "alignment_issues": mismatches, **({"render": extra} if extra else {}),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if verbose and (n + 1) % 16 == 0:
                print(f"[ijb] {n + 1}/{len(items)} items, {time.time() - t0:.0f}s", file=sys.stderr)
    if verbose:
        print(f"[ijb] wrote {out_path} in {time.time() - t0:.0f}s; problems: {problems or 'none'}", file=sys.stderr)
    return out_path


def load(jsonl: str) -> list[dict[str, Any]]:
    """Items in phase1/run.py's shape: id, bench, kind, question, options [(name, text)],
    gold (int), image (PNG bytes), plus dataset, exact, official_mapika_ok."""
    items = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            path = r["image_path"]
            if not os.path.exists(path):  # copied elsewhere: images/ sits beside the jsonl
                path = os.path.join(os.path.dirname(os.path.abspath(jsonl)), "images", os.path.basename(path))
            with open(path, "rb") as g:
                img = g.read()
            items.append({"id": r["id"], "bench": r["bench"], "kind": r["kind"], "question": r["question"],
                          "options": [tuple(o) for o in r["options"]], "gold": r["gold"], "image": img,
                          "dataset": r["dataset"], "exact": r["exact"],
                          "official_mapika_ok": r.get("official_mapika_ok")})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--items", help="local items-extended-real.json fallback")
    ap.add_argument("--log", help="local results-decider-real.log fallback")
    a = ap.parse_args()
    path = build(a.out, a.cache, a.items, a.log)
    items = load(path)
    by: dict[str, list[int]] = {}
    for it in items:
        e = by.setdefault(it["dataset"], [0, 0])
        e[0 if it["exact"] else 1] += 1
    print(json.dumps({"jsonl": path, "n": len(items), "exact/approx by dataset": by}, indent=1))


if __name__ == "__main__":
    main()
