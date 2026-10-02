"""Rendering an image-bearing state for a Decider prompt, with no torch dependency.

Strands Decider renders a request as `<state> ... </state>` followed by one rendered
question that ends at `<answer>`. A Vision Decider keeps that contract and puts the
Qwen vision placeholder inside the state block, before any state text:

    <state>
    <|vision_start|><|image_pad|><|vision_end|>
    Page 3 of a signed rental agreement.
    </state>
    <question type="noul"> ...

The Qwen3.5 processor expands each `<|image_pad|>` into N copies, one per merged
patch (N = grid_thw.prod() // merge_size**2). Everything after the state is text, so
the question's option character spans still map onto tokens exactly as the text
Decider's `option_token_index` expects, once the base offset accounts for the expanded
placeholder. This module does that bookkeeping and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from strands_decider.prompting import RenderedQuestion, render_content, render_question
from strands_decider.schema import Content, Question

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"


@dataclass(frozen=True)
class VisionState:
    """The state of a vision request: zero or more images, then optional text.

    Images come first so that the expensive prefix (vision tower + image tokens through
    the decoder) is identical across every question in the request and can be cached
    once; see `infer.py`.
    """

    images: tuple[Any, ...]  # PIL images, paths, or anything the image processor accepts
    text: Content = ""

    @property
    def n_images(self) -> int:
        return len(self.images)


def render_vision_state(state: VisionState) -> str:
    """The shared prefix for a vision request, with one unexpanded placeholder per image."""
    placeholders = "\n".join(f"{VISION_START}{IMAGE_PAD}{VISION_END}" for _ in state.images)
    body = render_content(state.text) if state.text else ""
    inner = placeholders if not body else f"{placeholders}\n{body}" if placeholders else body
    return f"<state>\n{inner}\n</state>\n"


def build_vision_prompt(
    state: VisionState, question: Question, **kw: Any
) -> tuple[str, RenderedQuestion]:
    """Unexpanded prompt (one `<|image_pad|>` per image) plus the rendered question."""
    rq = render_question(question, **kw)
    return render_vision_state(state) + rq.text, rq


def expand_image_tokens(prompt: str, tokens_per_image: Sequence[int]) -> str:
    """Replace the i-th `<|image_pad|>` with `tokens_per_image[i]` copies.

    This mirrors `Qwen3VLProcessor.replace_image_token`, done here on the string so the
    tokenizer's offset mapping can be read against the same text the model sees.
    """
    parts = prompt.split(IMAGE_PAD)
    if len(parts) - 1 != len(tokens_per_image):
        raise ValueError(
            f"prompt holds {len(parts) - 1} image placeholders but {len(tokens_per_image)} "
            "token counts were given"
        )
    out = [parts[0]]
    for n, tail in zip(tokens_per_image, parts[1:], strict=True):
        out.append(IMAGE_PAD * int(n))
        out.append(tail)
    return "".join(out)


def question_base(expanded_prompt: str, rq: RenderedQuestion) -> int:
    """Character offset of the question inside the expanded prompt.

    The question is always the suffix of the prompt, so this is simply the prompt
    length minus the question length. Kept as a function so collate and infer agree.
    """
    base = len(expanded_prompt) - len(rq.text)
    if base < 0 or not expanded_prompt.endswith(rq.text):
        raise ValueError("rendered question is not the suffix of the prompt")
    return base


def option_token_index(
    offsets: Sequence[Sequence[int]], spans: Sequence[Sequence[int]], base: int
) -> list[int]:
    """Last token index of each option line. Identical to the text Decider's rule.

    Copied rather than imported so this module stays torch-free; the text version lives
    on a torch-importing collator class. The two are pinned equal in tests.
    """
    out: list[int] = []
    for s, e in spans:
        a, b = base + s, base + e
        last = -1
        for j, (lo, hi) in enumerate(offsets):
            if hi <= lo:
                continue
            if lo >= a and hi <= b:
                last = j
        if last < 0:
            raise ValueError(
                f"option span ({a},{b}) has no tokens left; the prompt was truncated "
                "through its option list"
            )
        out.append(last)
    return out


def image_tokens_for_grid(grid_thw: Sequence[Sequence[int]], merge_size: int) -> list[int]:
    """Tokens each image contributes after the patch merger: t*h*w // merge_size**2."""
    m = merge_size * merge_size
    return [int(t) * int(h) * int(w) // m for t, h, w in grid_thw]
