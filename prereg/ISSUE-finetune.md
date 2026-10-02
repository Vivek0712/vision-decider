**Title:** [FEATURE] Fine-tune Strands Decider (v19) on images: paired reasoning and confidence without the image

**Area:** training

---

### Human Overview

<!-- Written by the author, in their own words (50 words). -->

### Problem Statement

#<PR> adds image input to Strands Decider (v19) with no retraining, and it already matches the image-trained Mapika/decider-2b-vision on accuracy. Its measurements show two gaps that text training cannot close. **Paired reasoning:** NaturalBench G-Acc 0.323 against Mapika's 0.360; v19 struggles to tell near-identical images apart. **Confidence without the image:** with the image removed, v19 falls to chance but still answers POPE at mean confidence 0.725 (ECE 0.225). A decider whose confidence does not fall when its evidence is missing undermines the 0.9 / 0.5 routing convention.

### Proposed Solution

A preregistered image fine-tune from v19, in the form of `research/preregistrations/` ([draft](https://github.com/Vivek0712/vision-decider/blob/phase1-baselines/prereg/PREREGISTRATION-v19-vision.md)):

- Start from v19's adapter and head; vision tower frozen; v19's objective, LoRA targets and rank unchanged
- Image rows from licence-clean sources: VQAv2 complementary pairs (same question, similar image, opposite answer), GQA, PlotQA, rendered TabFact, EuroSAT, Open Images, KonIQ
- Image-removed copies as KL-only rows toward the frozen torso, so confidence falls without the image
- A text replay so JevBench does not regress

Evaluated on exactly the items #<PR> reports, with fixed thresholds below. If any prediction fails, v19 with `--vision` stays the image default.

### Use Case

The same agent decisions #<PR> enables (screenshots before tool calls, document checks, image guardrails), with two things production needs: telling apart screens that differ in one detail ("payment confirmed" versus "payment failed"), and confidence that drops when the image is unreadable or missing, so the agent confirms or escalates instead of acting.

### Alternative Solutions

- **Keep v19 as is with `--vision`:** already competitive; the gaps remain.
- **Higher image resolution (768 px, Mapika's size):** may explain part of the G-Acc gap without training; costs tokens and latency. Measurable separately, no training.
- **Reinforcement learning, as Mapika did (PPO):** targets games and control, not calibration.

The fine-tune targets the two measured gaps directly and is falsifiable.

### Additional Context

<details>
<summary>Predictions (on the items in #<PR>)</summary>

| # | Prediction | v19 `--vision` today | Mapika |
|---|---|---|---|
| 1 | NaturalBench G-Acc ≥ 0.383 and above Mapika; accuracy ≥ 0.802 | 0.323; 0.782 | 0.360; 0.793 |
| 2 | Image removed: mean confidence ≤ 0.58 and ECE ≤ 0.10 on NaturalBench and POPE; with the image, ECE ≤ 0.04 (NaturalBench) and ≤ 0.07 (POPE) | 0.652 / 0.152; 0.725 / 0.225 | 0.654 / 0.154; 0.686 / 0.181 |
| 3 | POPE-adversarial ≥ 0.862; Image JevBench preview exact ≥ 35/60 | 0.872; 38/60 | 0.857; 38/60 |
| 4 | JevBench ≥ 163, ECE ≤ 0.07, Brier ≤ 0.36 | 167, 0.052, 0.342 | — |
</details>

<details>
<summary>Cost and plan</summary>

- Data build on CPU (2–3 days); one training run of ~2–4 h on one L40S (or ~30–45 min on 8× A100), ~$5–20.
- The preregistration is committed before training and the outcome appended without editing, as in v9–v20.
- Thresholds, the data mix and the checkpoint name (e.g. `strands-decider-2B-hobson-v19-vision`) are open for discussion here first.
</details>
