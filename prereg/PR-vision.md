**Title:** feat(vision): image input for Strands Decider (v19), vision tower kept, no retraining

**Base:** `strands-labs/strands-decider:main` ← **Head:** `Vivek0712/strands-decider:feat/vision`

---

## Human Overview
<!-- Written by the author, in their own words (50 words). -->

## Description

**strands-vision-decider**: Strands Decider (v19) runs on Qwen3.5-2B-Base, which is natively multimodal, but `_load_torso` drops the vision tower. This PR keeps it behind `--vision`, so a request can carry images alongside `state`. No new weights: the published v19 adapter maps onto the same decoder inside the multimodal model (`layers.N` → `language_model.layers.N`), and the vision tower ships in the base checkpoint already (+0.7 GB in memory).

- `serve --vision`, `ask --image`; `SystemOneRequest.images` (base64); state may be empty with images
- Images go inside `<state>` as Qwen placeholders; encoded once and shared across questions; M-RoPE positions passed explicitly
- A text-only engine refuses images (HTTP 422) instead of ignoring them; text requests are unchanged
- `vision` extra (Pillow); needs transformers ≥ 5.18

Measured with v19 as published: NaturalBench 78.2% (ECE 0.011), POPE-adversarial 87.2%, level with the image-trained Mapika/decider-2b-vision on accuracy and better calibrated on NaturalBench. Details below.

## Related Issues

Follow-up (image fine-tune, preregistered): #<fine-tune issue, opened after this PR>

## Type of Change

New feature

## Testing

- `tests/test_vision.py` (9 tests, offline; builds a tiny Qwen3.5 with a tokeniser carrying the vision tokens): text answers equal between text and vision engines; shared-prefix answers over one and two images equal a full forward to < 1e-5 (fails without the M-RoPE delta); no position state leaks from an image forward into a text request (fails without the reset); image routing and 422 at the API.
- Full suite offline (`HF_HUB_OFFLINE=1`): 252 passed, 19 skipped. `ruff check .` passes. `mypy ./src`: no new errors from this PR (28 pre-existing on `main` with torch 2.14.1 / mypy 2.4).
- Real checkpoint: v19 on the multimodal torso matches upstream's v19 on text prompts to 9.8e-4.

- [ ] I ran `pytest -q` locally (GPU tests skip automatically without CUDA)
- [ ] I ran `ruff check .` and `mypy ./src` locally and both pass

## Additional Details

**Headline.** Three systems, each scored with the image and with it removed. Bold is the best of the three.

| | NaturalBench acc | NaturalBench G-Acc | NaturalBench ECE | POPE-adv acc | POPE-adv Brier | POPE-adv ECE | Image JevBench preview, exact (60) |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B-Base, untrained readout | 0.712 | 0.167 | 0.046 | 0.865 | 0.228 | 0.111 | 0.483 |
| **Strands Decider (v19), `--vision`** | 0.782 | 0.323 | **0.011** | **0.872** | **0.203** | 0.070 | **0.633** |
| Mapika/decider-2b-vision (image-trained) | **0.793** | **0.360** | 0.081 | 0.857 | 0.204 | **0.065** | **0.633** |

<details>
<summary>NaturalBench (300 groups, 1,200 questions)</summary>

Each group is two images and two questions whose answers flip between the images, built so a model that ignores the image scores at chance. Q-Acc counts a question right on both images, I-Acc an image right on both questions, G-Acc a group with all four right.

| System | Image | Acc | Q-Acc | I-Acc | G-Acc | Brier | ECE | Mean confidence |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B-Base, untrained | with | 0.712 | 0.435 | 0.487 | 0.167 | 0.380 | 0.046 | 0.665 |
| | removed | 0.500 | 0.000 | 0.110 | 0.000 | 0.529 | 0.092 | 0.592 |
| **Strands Decider (v19)** | with | 0.782 | 0.573 | 0.605 | 0.323 | 0.307 | **0.011** | 0.777 |
| | removed | 0.500 | 0.000 | 0.103 | 0.000 | 0.561 | 0.152 | 0.652 |
| Mapika/decider-2b-vision | with | **0.793** | **0.595** | **0.630** | **0.360** | **0.303** | 0.081 | 0.867 |
| | removed | 0.500 | 0.000 | 0.223 | 0.000 | 0.568 | 0.154 | 0.654 |

By question type (with the image):

| System | Yes/no (908): acc / Brier / ECE | Two-way choice (292): acc / Brier / ECE |
|---|---|---|
| Qwen3.5-2B-Base, untrained | 0.702 / 0.395 / 0.056 | 0.743 / 0.336 / 0.028 |
| **Strands Decider (v19)** | 0.769 / 0.324 / **0.024** | 0.822 / 0.253 / **0.031** |
| Mapika/decider-2b-vision | **0.775** / **0.321** / 0.081 | **0.849** / **0.249** / 0.089 |

v19 is within 1.1 points of Mapika on accuracy and 0.004 on Brier, with about 7x lower ECE; Mapika is ahead on paired reasoning (G-Acc 0.360 against 0.323) and answers with higher confidence (0.867 against 0.777).
</details>

<details>
<summary>POPE adversarial (600 questions)</summary>

"Is there a {object} in the image?" over COCO val2014, with adversarial negatives (objects that often co-occur), balanced yes/no.

| System | Image | Acc | Brier | ECE | Mean confidence |
|---|---|---|---|---|---|
| Qwen3.5-2B-Base, untrained | with | 0.865 | 0.228 | 0.111 | 0.757 |
| | removed | 0.470 | 0.509 | 0.056 | 0.526 |
| **Strands Decider (v19)** | with | **0.872** | **0.203** | 0.070 | 0.804 |
| | removed | 0.500 | 0.599 | 0.225 | 0.725 |
| Mapika/decider-2b-vision | with | 0.857 | 0.204 | **0.065** | 0.922 |
| | removed | 0.505 | 0.579 | 0.181 | 0.686 |

v19 is the most accurate and has the lowest Brier; Mapika is slightly better on ECE. Without the image, both trained models stay confident at chance (v19 0.725, Mapika 0.686); the untrained torso does not (0.526).
</details>

<details>
<summary>Image JevBench (official reference, and the published preview items)</summary>

**Official leaderboard** ([Image JevBench v0.1.5](https://benchmarkheaven.com/image-jev-bench)): the items are sealed and the runner is private, so v19 is not on it. For reference, Mapika/decider-2b-vision's row: public 151/228, sealed 319/456, Intelligence 52.8, Calibration 81.5, ranked 10th. We would like to submit v19 with `--vision` if you agree.

**Preview items.** The site publishes 128 earlier preview items (question, options, gold, source row) and Mapika's official right/wrong on each. We rebuilt them from their source datasets. 60 are exact (CLEVR-HOPE, Geometry3K, ArxivQA: the source image unchanged); 68 need rendering whose code is not public (FinQA tables, ScreenSpot click markers, Mind2Web boxes), so they are approximate.

Harness check: on the 60 exact items, our Mapika run agrees with Mapika's official per-item results on 95% of items (38 right against 37 officially). On the approximate items it does not (ScreenSpot 0.64 against 0.08 officially; Mind2Web 0.08 against 0.58), so only the exact subset is used for comparison.

| System | Image | Exact (60) acc | Exact Brier | Exact ECE | All 128 acc |
|---|---|---|---|---|---|
| Qwen3.5-2B-Base, untrained | with | 0.483 | 0.640 | 0.138 | 0.367 |
| | removed | 0.317 | 0.754 | 0.180 | 0.281 |
| **Strands Decider (v19)** | with | **0.633** | 0.484 | **0.135** | 0.430 |
| | removed | 0.350 | 0.683 | 0.157 | 0.266 |
| Mapika/decider-2b-vision | with | **0.633** | **0.454** | 0.191 | 0.531 |
| | removed | 0.500 | 0.593 | 0.150 | 0.336 |

By source dataset (with the image; chance is 0.50 for CLEVR-HOPE, 0.25 for Geometry3K, ArxivQA and FinQA, 0.20 for ScreenSpot and Mind2Web):

| System | CLEVR-HOPE (20, exact) | Geometry3K (20, exact) | ArxivQA (20, exact) | FinQA (20, approx) | ScreenSpot (36, approx) | Mind2Web (12, approx) |
|---|---|---|---|---|---|---|
| Qwen3.5-2B-Base, untrained | 0.85 | 0.15 | 0.45 | 0.30 | 0.25 | 0.25 |
| **Strands Decider (v19)** | 0.85 | 0.45 | **0.60** | 0.15 | 0.31 | 0.25 |
| Mapika/decider-2b-vision | **0.90** | 0.45 | 0.55 | 0.30 | 0.64 | 0.08 |
| Mapika, official | 0.90 | 0.40 | 0.55 | 0.40 | 0.08 | 0.58 |

Mapika scores 0.55 on ArxivQA with or without the image, so some of those questions are answerable from the text alone. 60 items resolve differences of about 10 points at best.
</details>

<details>
<summary>Image dependence: how much the image moves the answer</summary>

The same items scored with and without the image. Total-variation distance between the two probability vectors; confidence drop = top probability with the image minus without; answer flips = share of items whose top answer changes.

| System | NaturalBench: TV / conf drop / flips | POPE-adv: TV / conf drop / flips | Image JevBench exact: TV / conf drop / flips |
|---|---|---|---|
| Qwen3.5-2B-Base, untrained | 0.179 / 0.073 / 0.512 | 0.258 / 0.231 / 0.478 | 0.220 / 0.126 / 0.383 |
| **Strands Decider (v19)** | 0.275 / 0.124 / 0.437 | 0.279 / 0.080 / 0.412 | 0.236 / 0.094 / 0.433 |
| Mapika/decider-2b-vision | 0.368 / 0.213 / 0.477 | 0.399 / 0.236 / 0.418 | 0.355 / 0.237 / 0.500 |

All three depend on the image (every system falls to chance without it). The trained models lose too little confidence when it is missing, v19 most of all on POPE (0.080).
</details>

<details>
<summary>Text: nothing changes</summary>

With `--vision`, a text-only request runs through the same decoder weights. On text prompts v19 on the multimodal torso matches upstream's own v19 load to a max |Δp| of 9.8e-4 (bf16- against fp32-loaded weights), and the test suite pins text answers equal between the text and vision engines. JevBench (231 text tasks) was not rerun with `--vision`; v19's published result stands: 167/231 (0.723), Brier 0.342, ECE 0.052.
</details>

<details>
<summary>Method</summary>

- **Models:** `StrandsAgents/strands-decider-2B-hobson-v19` with `--vision` (adapter and head unchanged, vision tower frozen, per-kind temperatures as served); `Qwen/Qwen3.5-2B-Base` untrained, scored by the frozen torso's option-number logits at `<answer>` (the readout v19's KL term uses); `Mapika/decider-2b-vision` through its own published code and prompt.
- **Images:** v19 and Qwen at 448 px on the longer side (196 tokens for a square image); Mapika at 768 px, its training size.
- **Questions:** v19 gets yes/no as `noul` with default criteria and two-way questions as `choice`; the untrained readout gets `noul` criteria "the answer is yes/no"; Mapika gets its own `(A)/(B)` prompt.
- **Data:** NaturalBench (`BaiqiL/NaturalBench`), first 300 groups of shard 0; POPE (`lmms-lab/POPE`), adversarial split, first 600 by question id; Image JevBench preview definitions from `fstandhartinger/model-market-comparison` (MIT), images from each item's source dataset.
- **Runs:** one run per system, CPU (Intel Sapphire Rapids, 4-8 vCPU), bf16, transformers 5.18.0, torch 2.14.1. Confidence = top probability; Brier summed over options; ECE with 10 bins.
- **Reproduce:** `evaluation/vision/run.py` on the branch; `evaluation/vision/results/` holds every per-item probability behind these tables. A 16-item spot check of the in-tree script against the recorded run: same answer on all 16, mean |Δp| 0.004 (fp32 against bf16).
</details>

## Checklist
- [ ] I have read the [CONTRIBUTING](../CONTRIBUTING.md) document
- [ ] I have reviewed and understand every line of code in this PR, including any generated by AI tools, and I can explain why it works
- [ ] My change is focused and reasonably small; I have split unrelated work into separate PRs
- [ ] I have added any necessary tests that prove my fix is effective or my feature works
- [ ] I have updated the documentation accordingly
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published

----

By submitting this pull request, I confirm that you can use, modify, copy, and redistribute this contribution, under the terms of your choice.
