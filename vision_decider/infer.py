"""Serve a Vision Decider: encode the image state once, fork the cache per question.

Reuses `SystemOneEngine` for everything after the hidden states (option gathering,
temperatures, `_to_answer`). What changes is the prefix: it now carries `pixel_values`
through the vision tower and the image tokens through the decoder, so the cache that is
forked across questions already holds what the image contributed. Question suffixes are
text only and never see pixels, exactly as in the text engine.

Window budget order: question reserve first, then image tokens (fixed by the bucket),
then state text cut from the front. An image that does not fit after the reserve is
refused rather than cropped: a cropped image answers a different question.

Positions are passed explicitly. Qwen3.5 uses M-RoPE: an image advances the three
rotary axes by its grid size, not its token count, so text after an image sits at
`token index + rope_delta` (negative). Left to itself, a suffix-only forward after a
cached prefix builds positions from the full prefix+suffix mask (a shape error) and
reads a `rope_deltas` left on the module by the previous request.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from strands_decider.infer import (
    EngineConfig,
    SystemOneEngine,
    UnforkableCache,
    _expand_cache,
    _to_answer,
)
from strands_decider.modeling import (
    apply_temperature,
    gather_options,
    masked_log_softmax,
    pool_last_token,
)
from strands_decider.prompting import RenderedQuestion, render_question
from strands_decider.schema import Answer, Content, Question, SystemOneResponse, Usage

from .collate import _load_image
from .modeling import VisionDeciderModel, mm_token_type_ids, qwen_base
from .prompting import (
    VISION_END,
    VisionState,
    expand_image_tokens,
    image_tokens_for_grid,
    render_vision_state,
)


@dataclass
class VisionEngineConfig(EngineConfig):
    model_name: str = "vision-decider-0.1.0"
    image_long_side: int = 448


class VisionDeciderEngine(SystemOneEngine):
    def __init__(self, model: VisionDeciderModel, cfg: VisionEngineConfig | None = None, *, device: str = "cuda"):
        # SystemOneEngine takes (model, config); the device lives on the config.
        super().__init__(model, replace(cfg or VisionEngineConfig(), device=device))
        self.processor = AutoProcessor.from_pretrained(model.config.base_model)
        self.vcfg: VisionEngineConfig = self.cfg  # type: ignore[assignment]

    # ---- prefix ------------------------------------------------------------

    def _encode_images(self, images: tuple[Any, ...]) -> tuple[dict[str, torch.Tensor], list[int]]:
        if not images:
            return {}, []
        pil = [_load_image(im, self.vcfg.image_long_side) for im in images]
        img = self.processor.image_processor(images=pil, return_tensors="pt")
        counts = image_tokens_for_grid(img["image_grid_thw"].tolist(), self.processor.image_processor.merge_size)
        return {k: v.to(self.device) for k, v in img.items()}, counts

    def _fit_vision(self, state: VisionState, question_texts: list[str]) -> tuple[list[int], list[list[int]], dict[str, torch.Tensor]]:
        """Question reserve first, image tokens next, state text last (cut from the front)."""
        max_len = self.model.config.max_length
        enc = self.tok(question_texts, add_special_tokens=False, return_offsets_mapping=True)
        q, offs = enc["input_ids"], enc["offset_mapping"]
        longest = max(len(x) for x in q)
        reserve = min(longest, max(1, int(max_len * self.cfg.max_question_fraction)))
        cut = [max(0, len(x) - reserve) for x in q]
        self._last_offsets = [o[c:] for o, c in zip(offs, cut, strict=True)]
        q = [x[c:] for x, c in zip(q, cut, strict=True)]

        mm, counts = self._encode_images(state.images)
        raw = render_vision_state(state)
        expanded = expand_image_tokens(raw, counts)
        s = self.tok(expanded, add_special_tokens=True)["input_ids"]
        budget = max_len - reserve
        if len(s) > budget:
            # Cut state text from the front of the TEXT, never into the image block. The
            # placeholders sit at the start of the state, so everything up to the last
            # <|vision_end|> is kept whole and the text after it loses its head.
            vend = self.tok.convert_tokens_to_ids(VISION_END)
            keep_head = (max(i for i, t in enumerate(s) if t == vend) + 1) if counts else 0
            tail = budget - keep_head
            if tail < 8:
                raise ValueError(
                    f"image block of {keep_head} tokens leaves {tail} tokens for text in a "
                    f"{budget}-token state budget; lower image_long_side"
                )
            s = s[:keep_head] + s[len(s) - tail:]
        return s, q, mm

    @torch.inference_mode()
    def _vision_probs(
        self, state: VisionState, rendered: list[RenderedQuestion], kinds: list[str]
    ) -> tuple[torch.Tensor, int]:
        m = len(rendered)
        s, q, mm = self._fit_vision(state, [rq.text for rq in rendered])
        prefix_ids = torch.tensor([s], device=self.device)
        P = prefix_ids.size(1)
        text_pos = torch.arange(P, device=self.device).view(1, 1, -1)
        extra: dict[str, Any] = {}
        if mm:
            tt = mm_token_type_ids(self.model.torso, prefix_ids)
            mpos, delta_t = qwen_base(self.model.torso).get_rope_index(
                prefix_ids, tt, image_grid_thw=mm["image_grid_thw"])
            delta = int(delta_t.view(-1)[0])
            extra = {"mm_token_type_ids": tt, **mm}
        else:
            mpos, delta = text_pos.expand(3, 1, -1), 0
        prefix_out = self.model.torso(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            position_ids=torch.cat([text_pos, mpos], dim=0),  # [4, 1, P]: text row + 3 M-RoPE rows
            use_cache=True,
            return_dict=True,
            **extra,
        )
        if m == 1:
            # Single question: forward the suffix against the batch-1 cache directly.
            cache = prefix_out.past_key_values
        else:
            cache = _expand_cache(prefix_out.past_key_values, m)

        suffix_ids, suffix_mask = self._pad(q)
        full_mask = torch.cat([
            torch.ones(m, P, dtype=suffix_mask.dtype, device=self.device), suffix_mask
        ], dim=1)
        st = (torch.arange(suffix_ids.size(1), device=self.device) + P).view(1, 1, -1).expand(1, m, -1)
        suffix_pos = torch.cat([st, (st + delta).expand(3, m, -1)], dim=0)
        hidden = VisionDeciderModel.encode(self.model, suffix_ids, full_mask, past_key_values=cache,
                                           position_ids=suffix_pos)
        pooled = pool_last_token(hidden, full_mask).to(torch.float32)
        options = gather_options(hidden, self._option_idx(rendered, 0)).to(torch.float32)
        logits = apply_temperature(self.model.head(pooled, options), self._temperatures(kinds))
        n_slots = torch.tensor([rq.n_slots for rq in rendered], device=self.device)
        probs = masked_log_softmax(logits, n_slots).exp()
        return probs, P + int(suffix_mask.sum().item())

    # ---- public ------------------------------------------------------------

    def ask_vision(self, state: VisionState, questions: dict[str, Question]) -> SystemOneResponse:
        names = list(questions)
        rendered = [render_question(questions[n]) for n in names]
        answers: dict[str, Answer] = {}
        total = 0
        for start in range(0, len(names), self.cfg.max_batch):
            chunk = slice(start, start + self.cfg.max_batch)
            rqs = rendered[chunk]
            try:
                probs, ntok = self._vision_probs(state, rqs, [rq.kind for rq in rqs])
            except UnforkableCache as e:
                raise RuntimeError(f"hybrid cache could not be forked: {e}") from e
            total += ntok
            for i, name in enumerate(names[chunk]):
                row = probs[i, : rqs[i].n_slots].tolist()
                answers[name] = _to_answer(rqs[i], row, ordinal_smoothing=self.model.config.ordinal_smoothing)
        return SystemOneResponse(model=self.cfg.model_name, answers=answers,
                                 usage=Usage(input_tokens=total, output_tokens=len(names)))

    def ask(self, state: Content, questions: dict[str, Question]) -> SystemOneResponse:  # type: ignore[override]
        """Text-only requests still work: an empty image tuple is a plain Decider call."""
        return self.ask_vision(VisionState(images=(), text=state), questions)


def load_engine(path: str, *, device: str = "cuda", image_long_side: int = 448) -> VisionDeciderEngine:
    model = VisionDeciderModel.load(path)
    return VisionDeciderEngine(model, VisionEngineConfig(image_long_side=image_long_side), device=device)
