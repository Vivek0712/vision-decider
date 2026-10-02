# Pre-registration: v19-vision, images in the state

Committed before training. Same discipline as v9-v20.

> **Draft for the issue.** Fields marked **[fill before commit]** are fixed once the
> data is built and the baselines are rerun on the full evaluation sets, then this file
> is committed and frozen before the first training step. Nothing else in it changes
> after the issue agrees it.

## Why

Qwen3.5-2B-Base, the torso of every model since v13, is natively multimodal; the
loader drops its vision tower. Kept, with v19's adapter and head loaded unchanged
(no image training), v19 already decides over images about as well as the strongest
open image decider, Mapika/decider-2b-vision, which was trained on 50k image rows and
PPO. Its calibration is far better:

| model (images at 448 px; Mapika at its 768 px) | NaturalBench acc | NaturalBench G-Acc | NaturalBench ECE | POPE-adv acc | POPE-adv Brier | Image JevBench preview, exact (60) |
| --- | --- | --- | --- | --- | --- | --- |
| frozen Qwen3.5-2B-Base, option-number readout | 0.712 | 0.167 | 0.046 | 0.865 | 0.228 | 0.483 |
| **v19 + vision tower, no image training** | 0.782 | 0.323 | **0.011** | **0.872** | **0.203** | 0.633 |
| Mapika/decider-2b-vision | **0.793** | **0.360** | 0.081 | 0.857 | 0.204 | 0.633 |

(NaturalBench: first 300 groups, 1,200 questions; POPE adversarial: first 600 by
question id; Image JevBench preview: the 60 published preview items rebuilt exactly
from their source rows, on which our Mapika run agrees with Mapika's official per-item
results 95% of the time. Single run, CPU, bf16. Per-item outputs:
`evaluation/vision/results/`.)

Two failures remain, and neither is noise:

1. **Paired reasoning.** NaturalBench groups two images and two questions with
   opposite answers; G-Acc counts a group only if all four are right. v19 gets 0.323,
   below Mapika (0.360). Per question it is close (0.782 against 0.793), so the loss is
   in telling near-identical images apart, not in reading the question.
2. **Confidence without the image.** With the image removed every model falls to chance
   (NaturalBench 0.500, POPE ~0.50), as it should, but keeps answering confidently:

   | image removed | NaturalBench mean confidence / ECE | POPE-adv mean confidence / ECE |
   | --- | --- | --- |
   | v19 + vision tower | 0.652 / 0.152 | 0.725 / 0.225 |
   | Mapika/decider-2b-vision | 0.654 / 0.154 | 0.686 / 0.181 |
   | frozen Qwen3.5-2B-Base | 0.592 / 0.092 | 0.526 / 0.056 |

   A decision model whose confidence does not fall when its evidence is missing is
   exactly the failure the confidence-routing convention (act at 0.9, confirm at 0.5)
   cannot survive. The frozen torso is less overconfident than either trained model,
   so v19's training made this worse, and nothing in the corpus teaches it.

## What is being changed

Training starts from v19 (adapter and head, `init_from`), with the Qwen3.5 vision tower
and patch merger loaded and **frozen**; only v19's LoRA (same 12 targets, rank 16) and
the pointer head train, as in v19. Same objective as v19: cross-entropy (ordinal
smoothing 0.1 on score rows), KL to the frozen torso's option-number readout
(weight 0.3), option order reshuffled every example. Images enter `<state>` as Qwen
placeholders at a fixed 448 px long side (196 tokens for a square image). One setting,
fixed here; config `configs/experiments/v19-vision.yaml`. If v19-vision fails, the
mix, the weights and the image size are not tuned against the results below and rerun.

Three kinds of rows:

| kind | source | rows | labels | why |
| --- | --- | --- | --- | --- |
| image, paired yes/no | VQAv2 train2014 complementary pairs (CC BY 4.0 annotations, COCO train images): same question, a human-chosen similar image, opposite answer; both halves kept together | **[fill: target 20,000 = 10,000 pairs]** | dataset answer, majority of 10 | failure 1: the only public source of same-question, different-image pairs at this scale |
| image, yes/no and choice | GQA balanced train, binary (verify/logical/compare) and choose questions (CC BY 4.0) | **[fill: target 10,000]** | dataset | compositional questions, templated so pairs form across images |
| image, documents and charts | PlotQA (CC BY 4.0) templated yes/no and choice; TabFact statements over tables rendered as images (MIT, CC BY-SA tables), entailed/refuted | **[fill: target 6,000 + 6,000]** | dataset | text-in-image reading, held out of every evaluation below |
| image, choice with runtime labels | EuroSAT (MIT) and Open Images (CC BY 4.0) classification, one fixed option set per task | **[fill: target 8,000]** | dataset | reading option text over images |
| image, score | KonIQ-10k (CC BY 4.0; CC-licensed images only), 5-level quality | **[fill: target 5,000]** | majority level; vote histogram as soft target | the score type on images |
| image removed, KL only | a copy of 25% of the image rows above with at most 9 options, image deleted, weight 0 | **[fill: ~12,000]** | none: KL to the frozen torso's reading of the same text-only prompt (weight 1.0, upstream `kl_only_files`) | failure 2: teaches confidence to fall without the image instead of guessing from the question |
| text replay | rows drawn from v19's training files, each source in its v19 proportion | **[fill: target 20,000]** | as in v19 | holds text behaviour (prediction 4) |

Licences: every image source above permits training a model released with open
weights. Excluded deliberately: ImageNet, FUNSD, Rico, Food-101, AVA, Winoground
(non-commercial or research-only), ScreenSpot and ChartQA (evaluation sets), SEED and
ScienceQA (CC BY-NC).

Contamination: training images are deduplicated against every evaluation image by
COCO id and by perceptual hash (dHash, Hamming distance <= 4). POPE uses COCO
val2014, so no VQAv2 or GQA row whose image is a COCO val2014 image is kept (Visual
Genome includes some). NaturalBench images are Flickr30k and DOCCI; neither is a
training source. **[fill: rows removed by deduplication]**

Splits: by image, pair and source, so the two halves of a pair, every question about
one image and an image-removed copy are always on the same side. Validation 3%.

## Baselines

All on the full evaluation sets, measured before this file is committed, same code
and settings as the outcome. **[fill: rerun the three rows of the table under Why on
the full sets]**

| evaluation | v19 + vision tower | Mapika/decider-2b-vision | frozen 2B |
| --- | --- | --- | --- |
| NaturalBench (1,900 groups, 7,600 Q): acc / G-Acc / ECE | [fill] | [fill] | [fill] |
| NaturalBench, image removed: mean confidence / ECE | [fill] | [fill] | [fill] |
| POPE adversarial (3,000): acc / Brier / ECE | [fill] | [fill] | [fill] |
| POPE adversarial, image removed: mean confidence / ECE | [fill] | [fill] | [fill] |
| Image JevBench preview, exact (60) | [fill] | [fill] | [fill] |
| JevBench (231), text: score / ECE / Brier | 167 / 0.052 / 0.342 | — | — |

## Predictions

Each threshold below is written against the 300-group / 600-item numbers under Why and
is re-expressed against the full-set baseline in the same form (baseline plus the stated
margin) before commit.

1. **Paired reasoning is learned:** NaturalBench G-Acc at least v19 + 0.060 (subset:
   0.383), and above Mapika's. Per-question NaturalBench accuracy at least v19 + 0.020.
2. **Confidence falls without the image, and only then:** with the image removed, mean
   confidence at most 0.58 on both NaturalBench and POPE, and ECE at most 0.10 on both
   (v19 0.152 / 0.225; Mapika 0.154 / 0.181). With the image, NaturalBench ECE at most
   0.04 and POPE ECE at most 0.07: the confidence is lost where the evidence is, not
   everywhere.
3. **Nothing on images is lost:** POPE adversarial accuracy at least v19 - 0.010
   (subset: 0.862); Image JevBench preview exact at least v19 - 3 items.
4. **Nothing on text is lost:** JevBench at least 163, ECE at most 0.07, Brier at most
   0.36 (v19 167 / 0.052 / 0.342; the retrain noise is SD 3.2 tasks).

Exploratory, no numeric prediction: Image JevBench (official) if the maintainers agree
to submit; the 68 approximate preview items (ScreenSpot, Mind2Web, FinQA); NaturalBench
yes/no against two-way choice; score rows on held-out KonIQ (Spearman with the mean
opinion score, and score confidence); the image-removed confidence on JevBench's text
tasks with an unrelated image attached.

## What would count as failure

- **1 fails:** 10,000 human-made complementary pairs do not teach a single-pass 2B model
  to separate near-identical images; the gap to Mapika would then come from its RL stage
  or its 768 px images, not its data.
- **2 fails on the image-removed rows but holds with the image:** KL-only rows toward the
  frozen torso are too weak a target; the frozen torso is itself mildly overconfident
  without the image (POPE confidence 0.526 at chance).
- **2 holds by lowering confidence everywhere:** the with-image ECE bounds catch this; the
  model learned to hedge, not to notice the missing image.
- **1 and 2 hold, 4 fails:** image training costs text decisions at this mix; the text
  replay share was too small.

## Which model is the default afterwards

v19 stays the reference recipe for text. If predictions 1 to 4 all hold, v19-vision is
published as the image-capable model and becomes the default when a request carries
images. Otherwise v19 with its vision tower kept (no image training) is the image
default, as measured under Why.

## What this test cannot show

- **NaturalBench is natural photographs.** It says nothing about documents, charts or
  screens, where most agent decisions sit; the Image JevBench preview's 60 exact items
  (CLEVR scenes, geometry diagrams, paper figures) are the only other measure, and 60
  items resolve differences of about 10 points at best. The official Image JevBench
  is sealed; its public items are not downloadable.
- **G-Acc on 1,900 groups resolves differences of about 0.02**; prediction 1's margin is
  three times that.
- **One seed.** The text retrain noise (SD 3.2 JevBench tasks) is known; the image
  noise is not.
- **Mapika runs at 768 px and v19-vision at 448 px.** A gap could be resolution rather
  than training; the image size is fixed here, not tuned.
- **CPU bf16 evaluation** for every row; GPU numerics differ by about 1e-3 per probability.

## Outcome (added after the run)

[Nothing above this section is edited after training.]
