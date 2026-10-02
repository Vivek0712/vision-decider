# Vision Decider: Decision Models for Vision Inputs

Research document, 1 Oct 2026. Code in this repository; Claude Doc with drawn diagrams at
https://claude.ai/code/artifact/1374de98-2d9a-4acf-8be4-03544d9f9952

Contents

1. What a decision model is
2. Strands Decider architecture
3. Inference contract and serving
4. The Qwen torso already sees: what Decider throws away
5. Four ways to build a Vision Decider
6. Recommended design
7. Reference implementation
8. Evaluation, risks and next steps
9. Existing image benchmarks that work as a decider eval
10. What can be claimed at release
11. Plan, timeline and cost
12. Sources

---

## 1. What a decision model is

A decision model (also called a system-one model) picks between options or scores on a
scale in one forward pass, with no text generation and a calibrated confidence attached
to every answer. The category took off when TypeSafe AI shipped Jev in September 2026;
Strands Decider 2B (released 1 Oct 2026 by Marc Brooker, Mike Chambers and Fabio Nonato
de Paula under strands-labs) is AWS's open-source entry, with weights, training data and
scripts on GitHub and Hugging Face.

| Property | Decision model (Decider, Jev) | Generative LLM |
| --- | --- | --- |
| Output | One probability per option, read from a single pass | Free text, token by token |
| Latency | ~115 ms median on an RTX 3090, ~153 ms on an M3 MacBook for small tasks | Hundreds of ms to seconds, grows with output length |
| Confidence | Derived from the option distribution, calibrated (Brier measured) | Not exposed by frontier APIs |
| Label set | Defined in the request at runtime, no retraining | Prompted, but unconstrained output |
| Many questions on one state | Shared-prefix cache: 5 extra questions cost ~0 ms | Each question re-encodes the prompt |
| Weak at | Multi-step reasoning, coding, summarisation, anything needing generated text | Cheap rote decisions at scale |

Strands reports Decider 2B at 3rd of 33 in the 2B class on JevBench's public set (1st of
30 excluding just-over-2B models), measured on accuracy plus Brier calibration. Uses they
see working: model routing, tool selection, evals, guardrails, memory and context
management, policy classification, and hybrid agents where an LLM makes the hard calls and
the decider makes the rote ones. The shipped example is a Strands intervention that gates
a tool call on two yes/no questions ("are the arguments grounded in what the user said",
"is it premature to call this tool") before the tool runs.

This is the first open, trainable model in the category, which is the opening for a
vision variant: nobody has published a decision model over images yet, and the recipe is
small enough to retrain on one GPU.

## 2. Strands Decider architecture

The whole model is a pretrained decoder with its LM head replaced by a 1M-parameter
pointer head that scores each option's last-token hidden state against the hidden state
at the `<answer>` token. Design documented in docs/architecture.md of the repo.

```
Rendered prompt: <state> text </state> then <question type=choice> with numbered
<options>, ending at the <answer> token. One prompt per question, state shared.
        |
        v
Qwen3.5-2B-Base torso, LM head removed
  1.9B params, 24 layers: 18 Gated DeltaNet (linear attention) + 6 full attention
  causal attention kept as pretrained, bf16
  rank-16 LoRA on the projections of both layer kinds is the only torso training
        |
        v
h(option 1) ... h(option K)   (keys: last token of each option line)
h(answer)                     (query: the <answer> token)
        |
        v
Pointer head, ~1M params, fp32
  logit_k = dot(q(h_answer), k(h_option_k)) / sqrt(256)
  no per-option parameters -> no position prior, no cap on option count
        |
        v
masked softmax over the K logits
        |
        +--> noul (yes/no): returns P(true)
        +--> choice (one of N): argmax, conf = (N p_max - 1) / (N - 1)
        +--> score (ordered levels): E[i], conf = 1 - sigma / sigma_max
```

The query comes from the `<answer>` position (it has read the whole prompt under causal
attention) and the keys from each option's last token (it has read its own option).
Softmax over K logits is the entire inference path.

**Why the pointer head beat the slot head.** The first iteration (v1 to v7 era,
Qwen3-1.7B-Base torso) mapped the final hidden state to a fixed set of slots. Slots carry
per-option parameters, so the model memorised position priors ("slot 0 tends to be
positive") and capped the option count. The pointer head holds no per-option parameters:
an option's score depends on what it says, not where it sits. Measured order sensitivity
is still ~0.016 when the option list is reversed, because later options attend to
earlier ones under causal attention; removing that would need per-option attention
isolation, which is not implemented.

**Training objective, three terms.**

1. Cross-entropy on the gold label. Score targets move 10% of mass to adjacent levels
   (ordinal smoothing), so one level off is a smaller error than four off.
2. KL to the frozen torso's own readout (weight 0.3): the same torso with the adapter
   disabled, reading the option-number tokens it would have emitted after `<answer>`.
   Keeps the model near what pretraining knew at the cost of one extra forward pass.
3. KL to a frozen Qwen3.5-4B teacher (weight 1.0), on multi-step rows only.
   Distributions computed once and stored. Distilling on short classification moved
   nothing (v12); on long documents it helped (v14).

**The one rule that matters most:** option order is reshuffled on every example, every
epoch, with the label remapped to follow. Without it the heads memorise slots; with it
the only surviving strategy is reading the option text. Score questions are only ever
reversed (rubric and label together), never permuted, because their slots are ordinal.

**Data.** v14 trained on 100,449 rows from 21 public classification tasks plus 12,909
multi-step rows from ContractNLI, MuSiQue and BoardgameQA. v19 adds 3,815 generated
document questions and 6,166 answer-adequacy rows from HelpSteer2. v20 experimented with
catch-all options and teacher distributions only where the teacher agrees with gold; it
did not displace v19.

**Design decisions that carry over to a vision variant.**

| Decision | Reason given | Holds for vision? |
| --- | --- | --- |
| Keep causal attention, read at the right tokens | Converting to bidirectional breaks the pretrained weights | Yes. A VL torso is still a causal decoder over image + text tokens |
| Head in fp32, torso bf16 | A low-precision classifier is a needless calibration error | Yes |
| Remove shortcuts the data allows | One fixed question per task taught the model to ignore questions (v11) | Yes, and vision adds one: answering from the caption or question alone |
| Drop, never truncate | A prompt cut through its option list scores an option from a neighbour's state | Yes, and image tokens make the budget tighter |
| Hybrid torsos fork the recurrent state for the prefix cache | GDN layers carry conv and recurrent state, not KV | Yes, and the image prefix is the expensive part worth caching |

Checkpoint ~88 MB (adapter, readout, tokenizer); the 4.5 GB base weights download from
Hugging Face at load. The repo ships pre-registered experiments v9 through v20 in
`research/preregistrations/`.

## 3. Inference contract and serving

`POST /v1/systemone`: one state plus a map of typed questions, each answered
independently against the same state. JevBench's `typesafe` adapter runs against it
unmodified (231 of 231 tasks attempted, schema validity 1.000).

```
POST /v1/systemone
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "hobson-latest",
  "questions": {
    "is_urgent":   {"type": "noul",   "instructions": "Does this convey urgency?"},
    "department":  {"type": "choice", "instructions": "Route it.",
                    "criteria": {"billing": "money", "technical": "bugs", "sales": "pricing"}},
    "frustration": {"type": "score",  "instructions": "How frustrated?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}
```

Recorded v19 output: urgency P(true) 0.801; department -> technical at 0.748, confidence
0.622 (billing 0.234, sales 0.018); frustration expected value 1.24, confidence 0.578.
Routing convention borrowed from Jev's docs: act at 0.9 or above, confirm between 0.5 and
0.9, hand to a person below 0.5; measure your own thresholds on your own traffic.

**Shared-prefix cache.** The state is encoded once and its KV cache plus the conv and
recurrent state of the 18 GDN layers is forked across the question batch; only each
short question suffix is forwarded. v14 on an RTX 3090: 8 questions over a ~2,000-token
state take 369 ms through the shared prefix against 1,964 ms batched; 16 take 445 ms
against 4,086 ms. Numerically identical in fp32, ~2e-3 apart in bf16.

**Window handling.** Question tokens reserved first, state cut from the front (`_fit`
in `infer.py`); training and eval drop rows over the window rather than truncate.

| Environment | Setup | Measured |
| --- | --- | --- |
| Linux + CUDA | `flash-linear-attention` for the GDN layers | ~115 ms median, RTX 3090 |
| macOS Apple silicon (MPS) | torch 2.7.1, transformers 5.17.0, peft 0.21.0, `mps_kernels.py` | 153 ms warm under 300 tokens, 234 ms median across JevBench, p95 2,628 ms, M3 Pro |
| CPU | `--device cpu` | Much slower, no figure published |

Checkpoint files: `strands_decider_config.json`, `lora/`, `head.safetensors` (or
`slot_head.pt`), tokenizer. `strands-decider calibrate` fits temperatures. The server
binds to 127.0.0.1 with no auth; in production it sits behind an authenticated wrapper.

## 4. The Qwen torso already sees: what Decider throws away

The single most useful fact for this project: Qwen3.5-2B-Base is a "Causal Language
Model with Vision Encoder", trained with early fusion on multimodal tokens from the first
pretraining step. Strands Decider deliberately discards that vision tower. In
`modeling.py`, `_load_torso` detects `model_type in {"qwen3_5", "qwen3_5_text"}` and loads
the text decoder through `Qwen3_5ForCausalLM(...).model`, with the comment that
`AutoModel` "would hand back the wrapper with a vision tower". A Vision Decider does not
need a new vision head bolted on; it needs the one already in the checkpoint kept, and the
recipe re-run with image tokens in the state.

| Model | Params | Modality | Decoder layout | Why it matters here |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B-Base | 0.8B | image + text, early fusion | GDN + gated attention hybrid | Smallest torso; edge candidate |
| Qwen3.5-2B-Base | 1.9B LM + vision encoder | image + text, early fusion | 24 layers: 6 x (3 GDN -> FFN, 1 gated attention -> FFN); hidden 2048; 262K context | Decider v13-v19 torso. Keep the vision tower and the pointer head works as is |
| Qwen3.5-4B-Base | 4B | image + text | same family | Decider's frozen teacher; natural teacher for vision rows |
| Qwen3.5-27B, 35B-A3B, 122B-A10B, 397B-A17B | dense and MoE | image + text | same family | Large teachers only |
| Qwen3-VL-Embedding / Reranker | - | image + text | late-fusion Qwen3-VL lineage | Alternative path, weaker (section 5) |
| Qwen3.8-27B, Qwen3.8-Flash-Next 180B | 28B, 180B | image + text | newer generation | Teachers only; no 2B-class Qwen3.8 base listed |

**How Qwen3.5 handles images.** Earlier Qwen-VL models were late fusion: a separately
pretrained SigLIP-style ViT, a projection, then the LM. Qwen3.5 keeps ViT + MLP merger +
decoder but trains all of it jointly from step one; the vision encoder never exists as an
independent CLIP-style model. Position via interleaved multimodal RoPE. Image tokens
enter the same hybrid decoder as text, so the `<answer>` position has read the image
through the same causal attention it uses for state text. That is the property the
pointer head depends on.

**Token budget.** A compiled edge build reports 100 image tokens at a 320px bucket;
Qwen3.5's dynamic resolution scales token count with pixels (roughly 576 tokens for
512x512 on the large models). Decider v19 trained at a 3,072-token window and serves at
4,096. The image resolution bucket is a first-class design parameter.

**Licence.** Qwen3.5 base checkpoints are Apache-2.0; Strands Decider is open source.
Nothing blocks releasing a derived model.

## 5. Four ways to build a Vision Decider

Option A is the recommendation. It keeps every validated Decider decision and adds no
trainable component beyond the LoRA and the 1M-parameter head.

| Option | What changes | New trainable params | Image cost per query | Risk |
| --- | --- | --- | --- | --- |
| A. Native VL torso + pointer head | Load `Qwen3_5ForConditionalGeneration`, keep ViT + merger, image tokens inside `<state>`, LoRA on decoder projections, pointer head as is | LoRA + 1M head; optional LoRA on merger | One ViT pass + image tokens through the decoder, cached as shared prefix | Window pressure; model may answer from question text and ignore the image |
| B. Frozen vision embedding + text decider | Qwen3-VL-Embedding or SigLIP encodes the image to a vector or short token set, projected as soft prompt tokens | Projector + LoRA + head | Cheap | Pooled embeddings drop fine detail; the decoder never saw these tokens |
| C. Caption-then-decide | A VLM captions or OCRs, the text Decider decides over the caption | None | One generation call plus the decider | Loses the latency and calibration wins; the caption decides what the decider can see |
| D. Vision-only slot head | ViT features pooled, slot head over a fixed label set | Head only | Cheapest | A classifier, not a decider: fixed labels, position priors. Reproduces the v1 slot-head failure |

**Why A wins on the Decider's own criteria.** The pointer head's guarantees (runtime
labels, no option ceiling, no position prior) come from reading option text through the
same decoder that read the state. Under early fusion, the image is just more state.
Option B inserts tokens the decoder never saw, so the KL-to-frozen-torso term has nothing
to anchor to. Option C is a pipeline whose first stage is generative. Option D throws
away everything that distinguishes a decider from a fine-tuned ViT.

**What Option A still has to solve.**

1. Window budget: reserve question tokens, then image tokens, then cut state text from
   the front. Fix an image bucket per deployment (e.g. 448px long side).
2. Image-blindness shortcut: minimal pairs (same question and options, two images that
   flip the label), following the MuSiQue answerable/unanswerable pattern.
3. Prefix cache with images: the ViT plus image tokens through 24 layers belong in the
   shared prefix; `_expand_cache` already forks GDN state, the split just falls after
   the image.
4. KL reference on image rows: the untrained Qwen3.5-2B-Base can already answer image
   questions; rerun the "what the torso knows untrained" check on images first.
5. Teacher: Qwen3.5-4B-Base (or 27B) on image rows only where it beats the student.

## 6. Recommended design

Keep the Qwen3.5-2B vision tower, put the image inside `<state>`, leave the pointer
head and the three question types exactly as v19 defines them.

```
Image(s), fixed bucket (e.g. 448px)     State text (optional: caption, DOM, metadata)
        |                                         |
Qwen3.5 ViT + MLP merger (frozen)                 |
        |                                         |
        v                                         v
Qwen3.5-2B decoder, LM head removed, rank-16 LoRA on v19 targets
  image tokens and text tokens in one causal stream (early fusion, M-RoPE)
        |
        v
shared prefix: KV + GDN recurrent state after image and state, forked per question
        |
   +----+--------------------+
   v    v                    v
noul suffix   choice suffix   score suffix     (text only, no pixels)
   |    |                    |
   v    v                    v
pointer head, unchanged: q(h_answer) against k(h_option_k), fp32, ~1M params
        |
        v
P(true), argmax + confidence, expected level + confidence
```

**Prompt rendering.**

```
<state>
<|vision_start|><|image_pad|>...<|vision_end|>
Page 3 of a signed rental agreement, scanned at 300 dpi.
</state>
<question type="noul">
Decide whether the statement is true of the state.
Is the signature block filled in?
<options>
1. false -- the signature block is empty or missing
2. true -- a signature is present in the signature block
</options>
</question>
<answer>
```

Option spans come from character offsets in the question text, so `option_token_index`
works once told how many tokens the image expanded to in front of the question.

**Training objective, with the vision additions.**

1. CE on gold, ordinal smoothing 0.1 for score rows. Unchanged.
2. KL to the frozen torso's option-number readout, weight 0.3. The frozen torso now
   includes the vision tower. Measure untrained accuracy per family first; drop the term
   where it is near chance.
3. KL to a frozen teacher (Qwen3.5-4B-Base) on image rows where the teacher beats the
   student on a held-out slice. Weight 1.0.
4. New: for every image row, a paired row with a different image and flipped label, and
   a text-only ablation row whose target is the frozen torso's text-only distribution.
   The pair stops the caption shortcut; the ablation teaches confidence to fall without
   the image.
5. Trainable: LoRA rank 16 on decoder projections (v19 targets), pointer head fp32.
   Vision tower frozen in run 1; run 2 adds LoRA to the merger only.

**Data recipe.** Target 60k to 100k rows, Decider `Example` JSONL shape.

| Family | Types | Public sources | Rows | Why |
| --- | --- | --- | --- | --- |
| Document and form checks | noul, choice | DocVQA, FUNSD, RVL-CDIP, SROIE | 20k | Gating before expensive LLM extraction |
| UI and screenshot decisions | choice, noul | Rico, ScreenSpot, Mind2Web, AITW | 20k | Computer-use agents |
| Visual grounding yes/no | noul | VQAv2 yes/no, GQA binary, Winoground pairs | 15k | Calibrated "is X visible" |
| Classification with runtime labels | choice | ImageNet-1k subsets as 5-30 option lists, Food-101, EuroSAT | 15k | Reading option text, many-option routing |
| Ordinal visual rating | score | AVA aesthetics, damage severity, chart quality | 10k | Score with a visual rubric |
| Chart and table reading | choice, noul | ChartQA, PlotQA, TabFact renders | 10k | Dashboard checks |

Every family needs minimal pairs, the per-family balance check, and whole held-out task
families (Strands holds out tasks, not rows).

**Serving.** Fixed bucket at load; image encode and prefix forward once per request;
questions fork from the cache. Expect a 200 to 300 ms floor on an RTX 3090 at 448px.

## 7. Reference implementation

Package `vision_decider` (eight modules, ~1,100 lines) subclasses `strands_decider`
rather than forking it. Written against strands-decider main (cloned 1 Oct 2026) and
transformers 5.18.0 (`Qwen3_5ForConditionalGeneration`, `Qwen3VLProcessor`). Six
torch-free tests pass; GPU modules compile but have not run on a GPU.

| Module | What it does | Reuses from Decider | Tested |
| --- | --- | --- | --- |
| `prompting.py` | `VisionState`, `<state>` with one placeholder per image, placeholder expansion, option positions after expansion | `render_question`, option-span rule | 4 tests |
| `data.py` | `VisionExample` (images, pair_id, ablation), JSONL io, `make_pairs`, `add_ablations`, `balance_report`, `split_by_pairs`, converters | `Example`, `to_question` | 2 tests |
| `collate.py` | `VisionCollator`: pixels via processor, options via tokenizer offsets, over-budget rows dropped | shuffle, remap, ordinal smoothing, teacher slot order | compiles |
| `modeling.py` | `VisionDeciderModel`: loads `Qwen3_5ForConditionalGeneration(...).model`, LoRA scoped by regex to `language_model.*`, `encode` takes `pixel_values`/`image_grid_thw`, tied-embedding walk through the wrapper | `PointerHead`, `gather_options`, `masked_log_softmax`, KL reference, save/load | compiles |
| `train.py` | CE + ordinal smoothing, KL frozen (0.3), KL teacher (1.0), length-grouped batches, AdamW, cosine | `LengthGroupedBatchSampler` | compiles |
| `infer.py` | `VisionDeciderEngine`: image prefix encoded once, hybrid cache forked, `_fit_vision` keeps the image block whole | `SystemOneEngine`, `_expand_cache`, `_to_answer` | compiles |
| `server.py` | `/v1/systemone` with an `images` list of base64 bytes | `SystemOneResponse` | compiles |
| `configs/train-vision.yaml` | v19 recipe + `image_long_side: 448`, `max_image_tokens: 640`, `lora_on_merger: false` | v19 hyperparameters | - |

The one new piece of math, option position after image expansion:

```python
def option_token_index(offsets, spans, base):
    # base = len(expanded_prompt) - len(question_text)
    out = []
    for s, e in spans:
        a, b = base + s, base + e
        last = -1
        for j, (lo, hi) in enumerate(offsets):
            if hi <= lo:
                continue
            if lo >= a and hi <= b:
                last = j
        if last < 0:
            raise ValueError("option span has no tokens left; truncated through the option list")
        out.append(last)
    return out
```

How the torso is loaded, the one line that differs from Decider:

```python
full = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(config.base_model, dtype=torch.bfloat16)
torso = full.model   # Qwen3_5Model: .visual (ViT + merger) + .language_model, no lm_head
```

LoRA scope, so the ViT is never adapted by accident:

```python
scope = r"language_model\..*" if not cfg.lora_on_merger else r"(language_model\..*|visual\.merger\..*)"
LoraConfig(r=16, lora_alpha=32, target_modules=rf"^{scope}\.({'|'.join(cfg.lora_targets)})$",
           task_type="FEATURE_EXTRACTION")
```

Request shape, unchanged apart from `images`:

```
POST /v1/systemone
{
  "images": ["<base64 png>"],
  "state": "Checkout page after the user tapped Pay.",
  "questions": {
    "done":     {"type": "noul",   "instructions": "Did the payment succeed?"},
    "next":     {"type": "choice", "instructions": "Which control should the agent use next?",
                 "criteria": {"Retry": "a retry button is shown", "Back": "return to the cart", "None": "nothing to do"}},
    "severity": {"type": "score",  "instructions": "How severe is the error shown?",
                 "criteria": ["no error", "warning", "blocking error"]}
  }
}
```

Known gaps before a first run: (1) `_output_embedding` is overridden to reach the tied
table at `language_model.embed_tokens`; assert `tie_word_embeddings` on first load.
(2) `_fork_layered_cache` has not been run with an image prefix; rerun
`test_prefix_cache.py` with one to pin bf16 agreement. (3) Dynamic resolution means two
448px images of different aspect yield different token counts; fix aspect as well as
long side when measuring latency.

## 8. Evaluation, risks and next steps

| Measure | How | Target for v1 |
| --- | --- | --- |
| Accuracy, held-out families | Hold out whole families: chart reading and UI screenshots | +0.15 absolute over the untrained readout; above caption-then-text-Decider |
| Calibration | Brier and ECE after `strands-decider calibrate`, per type | ECE under 0.06 (v19 text: 0.052) |
| Image dependence | Minimal-pair twin flips the answer; image removal drops confidence | Pair accuracy above 0.8 of paired; mean confidence drops at least 0.2 |
| Option genericity | Reversed order; unseen label sets | Order shift under 0.03; unseen-label accuracy within 0.05 |
| Latency | p50/p95 at 320, 448, 640px, 1 and 8 questions | p50 under 300 ms at 448px; 8 questions under 1.6x of 1 |
| Prefix cache exactness | fp32 and bf16 with an image prefix | Identical fp32; under 3e-3 bf16 |

**Risks, in likely order.** Caption and question shortcuts (minimal pairs and the
balance warning are the guard). Window pressure (448px page ~256 tokens; multi-page
needs a smaller bucket or one image per request). Frozen-torso KL weak on images (the
`kl_frozen_on_images` switch, measure first). Hybrid cache forking untested with images
(fallback to batched encoding). Serving footprint grows with the vision tower (the 88 MB
checkpoint does not).

**Where it applies.** Tool-call gating for computer-use agents with the screen in the
state, document checks before an expensive LLM extraction, guardrails that need a
confidence score rather than generated text.

**Next steps.** Untrained readout on 500 image rows per family; convert DocVQA yes/no,
VQAv2 binary, a Rico subset; pairs and balance report; run 1; calibrate; held-out eval;
prefix-cache test with an image prefix; run 2 with `lora_on_merger`; pre-register in
the Decider style and offer upstream to strands-labs.

## 9. Existing image benchmarks that work as a decider eval

No decision-model benchmark exists for images, but several VLM benchmarks have
closed-form answers that map onto noul and choice, and two test exactly what the pointer
head claims.

| Benchmark | Format | Maps to | Why it fits |
| --- | --- | --- | --- |
| MME | ~2,800 yes/no, 14 subtasks, two questions per image (one yes, one no) | noul | Built-in minimal pairs; measures image dependence for free |
| POPE | ~9,000 yes/no object presence; random, popular, adversarial splits | noul | Hallucination and calibration; confidence should drop on adversarial |
| MMBench | ~3,000 multiple choice, 20 dimensions, CircularEval over option rotations | choice | CircularEval is the option-order test; scores pointer-head genericity |
| SEED-Bench (image subset) | ~14,000 multiple choice, 12 dimensions | choice | Volume, stable per-dimension breakdown |
| HallusionBench | yes/no with paired control images | noul | Harder minimal pairs |
| ScienceQA (image subset) | multiple choice with context text | choice + state text | Tests state text and image together |
| RealWorldQA, MMMU | multiple choice | choice | Harder; a 2B decider will do poorly on MMMU, worth reporting |

Cautions: these are eval sets, not training data, and Qwen3.5 has almost certainly seen
them, so report the untrained Qwen3.5-2B-Base readout beside the trained decider on each
set. None score calibration; add Brier and ECE from the option distributions, which is
the number nobody else publishes on these sets. DocVQA, ChartQA, TextVQA are open-answer
and only work after conversion to yes/no or hard-negative choice rows.

Recommended published set: MME and POPE for noul, MMBench with CircularEval and
SEED-Bench for choice, each with accuracy, Brier, ECE and order shift, plus held-out
families for the score type, which none of these cover.

**Effort for the eval suite:** converters 1 day; metric script half a day; untrained
baseline 3 to 4 h GPU; each trained checkpoint 3 to 4 h GPU; optional
caption-then-text-Decider baseline 4 to 6 h GPU; write-up half a day. Under $30 on one
L40S; one evening on an 8-GPU host with the passes in parallel.

## 10. What can be claimed at release

1. First open-source vision decision model (verify on release day that neither TypeSafe
   nor strands-labs has shipped image input).
2. No regression on text: run `evaluation/jevbench/jevbench.sh` with text-only states and
   report accuracy and Brier beside v19's 0.723 and 0.342, including if they drop.
3. The published image eval: four public benchmarks plus held-out families, minimal-pair
   accuracy, confidence drop without the image, order shift, latency by bucket, with rows
   and scripts committed. Match Decider's format: pre-registered runs, committed data,
   SHA256 sums.

Not claimable: state of the art on vision decisions, parity with Jev on images, or any
number on rows the model trained on.

## 11. Plan, timeline and cost

Critical path: clean data -> untrained-readout check -> run 1 -> eval -> run 2 -> export.
About six working days whatever the headcount on a single GPU; four days with an 8-GPU
host and three people.

**Three people, 7 working days, one p4d days 2 to 6** (assumed start Mon 5 Oct 2026)

| Day | A, data and eval | B, model and infra | C, eval set and release |
| --- | --- | --- | --- |
| 1 (Oct 5) | DocVQA + VQAv2 rows, pairs, balance report | Host, weights, prefix-cache test with image prefix | Eval script skeleton, pre-registration template |
| 2 (Oct 6) | Fix warning families; pre-registration; rows to B | Untrained-readout check; teacher labelling overnight | Model card and README drafts |
| 3 (Oct 7) | Rico + classification rows (held-out, v2) | Run 1, merger frozen (3 h on 8 GPUs) | Latency bench with image buckets |
| 4 (Oct 8) | Read run 1 eval, hunt shortcuts, fix data by midday | Calibrate, held-out eval, latency; run 2 on fixed data | Blog draft |
| 5 (Oct 9) | JevBench text regression | Pick winner; HF export; SHA256 manifest | Eval set and script committed |
| 6 (Oct 12) | Append outcomes to pre-registration | Second seed (variance) | PR or issue to strands-labs/strands-decider |
| 7 (Oct 13) | Buffer | Buffer | Publish |

Gates: rows to B end of Oct 6; run 1 eval to A start of Oct 8; fixed data to B midday
Oct 8; winner picked Oct 9.

**One engineer, 9 working days (11 with the benchmark suite).** Same order; GPU jobs
run overnight while the engineer does the next day's data, eval and writing.

**Shortest, 4 working days, one p4d (8x A100), three people**

| Day | Critical path | Parallel on spare GPUs |
| --- | --- | --- |
| 1 | Data: rows, pairs, balance report | Weights, cache test; untrained baseline on MME, POPE, MMBench x4, SEED across 4 GPUs; caption baseline on 2 |
| 2 am | Untrained-readout check (30 min, 2 GPUs) | Teacher labelling sharded 8 ways (~1 h) |
| 2 pm | Run 1, 8-GPU torchrun (~3 h) | Run 2 on a second host, or overnight on the same |
| 3 am | Calibrate; held-out eval; four benchmarks on both checkpoints, one per GPU (under 1 h) | Second seeds started |
| 3 pm | Read eval, pick winner; if a shortcut shows, fix data and rerun (3 h) | JevBench text regression |
| 4 | HF export, manifest, pre-registration outcomes, model card, publish | Second seeds finish |

The floor is set by the data day and the half day a human needs to read the first
eval. A second host lands run 1 and run 2 together on day 2 but the plan still ends day 4.

**Cost** (approximate on-demand us-east-1 list prices: p4d.24xlarge ~$32.80/h, g6e.xlarge
L40S ~$1.90/h; spot 60 to 70% off; verify in the calculator)

Single-GPU path:

| Item | Time | Hardware | USD |
| --- | --- | --- | --- |
| Data conversion, pairs, balance | 4 to 5 days | CPU | ~0 |
| Generated rows, 10 to 15k | 1 day async | hosted LLM API | 50 to 150 |
| Untrained-readout check | 1 to 2 h | 1 x L40S | ~5 |
| Teacher labelling, 100k rows | 6 to 10 h | 1 x L40S | ~20 |
| Run 1 / run 2 / second seed | 12 to 20 h each | 1 x L40S | 25 to 40 each |
| Calibration, eval, latency, cache test | 4 to 6 h | 1 x L40S | ~10 |
| Benchmark suite (four sets, two checkpoints, baselines) | ~12 h | 1 x L40S | ~25 |
| Total | | | 200 to 350 on-demand, under 120 spot |

Shortest 4-day plan, one p4d:

| Item | Host-hours | USD |
| --- | --- | --- |
| Day 1: baselines, converters, caption baseline, cache test | 8 | 262 |
| Day 2: readout check, sharded teacher, run 1, run 2 overnight | 16 | 525 |
| Day 3: calibration, held-out, benchmarks x2, JevBench, restart, seeds | 16 | 525 |
| Day 4: seeds finish, export, verify | 8 | 262 |
| Idle and setup (host up, not fully used) | ~16 | 525 |
| Generated rows | - | 50 to 150 |
| Storage, Hub | - | under 20 |
| Total | ~64 | ~2,200 to 2,300 on-demand; ~800 to 950 spot |

Tearing the host down nightly and rebuilding from S3 saves ~$400 at 30 minutes per
morning. Optional second host for day 2 only: +$262 on-demand, +$90 spot.

By activity: training runs ~40%, evaluation ~20%, teacher labelling and readout ~5%,
idle and setup ~25%, generated data and storage ~5 to 7%.

| Plan | Calendar | Compute on-demand / spot | People-days |
| --- | --- | --- | --- |
| One engineer, one L40S, GPU overnight | 11 days with eval suite | 200 to 350 / under 120 | 11 |
| Three people, one p4d days 2 to 6 | 7 days | ~650 / ~250 | 21 |
| Shortest, one p4d, 3 people | 4 days | ~2,250 / ~900 | 12 |

The 7-day plan is cheapest in total; the 4-day plan pays about $1,600 extra on-demand
for three calendar days, mostly on the eval suite, second seeds and idle time rather
than training.

## 12. Sources

- Introducing Strands Decider 2B, Strands blog, 1 Oct 2026:
  https://strandsagents.com/blog/introducing-strands-decider/
- strands-labs/strands-decider: https://github.com/strands-labs/strands-decider
  (docs/architecture.md, docs/inference.md, src/strands_decider/modeling.py,
  prompting.py, data/collate.py, infer.py, configs/train.yaml, read from a clone of main)
- StrandsAgents/strands-decider-2B-hobson-v19 weights:
  https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19
- Qwen/Qwen3.5-2B-Base model card: https://huggingface.co/Qwen/Qwen3.5-2B-Base;
  Qwen organisation: https://huggingface.co/Qwen
- The Qwen-VL lineage, CLIP features to early fusion:
  https://jerrickhoang.github.io/machine%20learning/2026/06/02/qwen/
- BLLM Qwen3.5-2B edge build (100 tokens per image at 320px):
  https://huggingface.co/ruisv/bllm-qwen3.5-2b
- Jev confidence documentation: https://docs.typesafe.ai/confidence
- transformers 5.18.0 source: models/qwen3_5/modeling_qwen3_5.py,
  models/qwen3_vl/processing_qwen3_vl.py
