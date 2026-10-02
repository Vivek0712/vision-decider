# Vision Decider: research document

The full research lives as a Claude Doc (diagrams, tables, timeline, sources) and
exports to Markdown, Word or PDF from the doc itself:

https://claude.ai/code/artifact/1374de98-2d9a-4acf-8be4-03544d9f9952

Sections:

1. What a decision model is (Jev, Strands Decider 2B, the category)
2. Strands Decider architecture: pointer head, LoRA torso, three question types,
   training objective, slot-head to pointer-head history, design decisions that carry over
3. Inference contract and serving: /v1/systemone, shared-prefix cache, window handling,
   measured latency on CUDA, MPS and CPU
4. The Qwen torso already sees: Qwen3.5-2B-Base is natively multimodal and Decider
   discards the vision tower on purpose (`_load_torso` in modeling.py)
5. Four ways to build a Vision Decider, and why the native-torso option wins
6. Recommended design: image as shared prefix, pointer head unchanged, training objective
   with image-side augmentation, data recipe by family
7. Reference implementation (this package): module map, the option-position math,
   torso loading, LoRA scope, request shape, known gaps before a first GPU run
8. Evaluation plan, risks, next steps
9. What can be claimed at release
10. Plan, cost and timeline: one engineer (9 days), three people (7 days), fastest honest
    release (7 days), cost table

Status of this package: written against strands-labs/strands-decider main (cloned
1 Oct 2026) and transformers 5.18.0. `python tests/test_prompting.py` passes (6 tests,
no GPU, no weights). The model, collator, trainer and engine compile but have not run
on a GPU yet.
