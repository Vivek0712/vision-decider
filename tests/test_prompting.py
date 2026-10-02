"""Torch-free tests: prompt rendering, image-token expansion, option position math, data rules."""
import pytest

from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from vision_decider.prompting import (
    IMAGE_PAD, VisionState, build_vision_prompt, expand_image_tokens,
    image_tokens_for_grid, option_token_index, question_base, render_vision_state,
)
from vision_decider.data import (
    VisionExample, add_ablations, balance_report, make_pairs, split_by_source,
    from_docvqa_yesno, from_labelled_images,
)


class FakeTok:
    """Whitespace tokenizer with offset mapping; each <|image_pad|> is one token."""
    def __call__(self, text):
        import re
        offs = [(m.start(), m.end()) for m in re.finditer(r"<\|[a-z_]+\|>|[^\s<]+|<", text)]
        return {"input_ids": list(range(len(offs))), "offset_mapping": offs}


def test_state_layout():
    s = render_vision_state(VisionState(images=("a.png", "b.png"), text="two pages"))
    assert s.startswith("<state>\n<|vision_start|><|image_pad|><|vision_end|>\n<|vision_start|>")
    assert s.endswith("two pages\n</state>\n")
    assert render_vision_state(VisionState(images=(), text="only text")) == "<state>\nonly text\n</state>\n"


def test_expand_and_grid():
    assert image_tokens_for_grid([[1, 32, 32], [1, 16, 24]], merge_size=2) == [256, 96]
    p = "x<|image_pad|>y<|image_pad|>z"
    assert expand_image_tokens(p, [3, 1]) == "x" + IMAGE_PAD * 3 + "y" + IMAGE_PAD + "z"
    try:
        expand_image_tokens(p, [3]); assert False
    except ValueError:
        pass


def test_option_positions_survive_expansion():
    q = ChoiceQuestion(instructions="Which team?", criteria={"billing": "money", "technical": "bugs", "sales": "pricing"})
    state = VisionState(images=("shot.png",), text="Help! payouts failing")
    for n in (1, 100, 576):
        prompt, rq = build_vision_prompt(state, q)
        expanded = expand_image_tokens(prompt, [n])
        base = question_base(expanded, rq)
        enc = FakeTok()(expanded)
        idx = option_token_index(enc["offset_mapping"], rq.option_spans, base)
        # Each option's last token must be the last word of that option line.
        words = [expanded[a:b] for a, b in enc["offset_mapping"]]
        assert [words[i] for i in idx] == ["money", "bugs", "pricing"], (n, [words[i] for i in idx])
        # and every option position is after all image tokens
        assert min(idx) > n


def test_option_order_permutation_tracks_labels():
    q = NoulQuestion(instructions="Is it signed?")
    state = VisionState(images=("p.png",))
    p0, rq0 = build_vision_prompt(state, q)
    p1, rq1 = build_vision_prompt(state, q, option_order=[1, 0])
    assert rq0.slot_labels == ("false", "true") and rq1.slot_labels == ("true", "false")
    assert p0 != p1


def test_pairs_ablations_split():
    rows = []
    for i in range(20):
        # Option order differs between rows: pairing keys on the option SET.
        opts = [["false", "no"], ["true", "yes"]] if i % 3 else [["true", "yes"], ["false", "no"]]
        gold = "true" if i % 2 else "false"
        rows.append(VisionExample(kind="noul", state="", instructions="Is the form signed?",
                                  options=opts, label=[o[0] for o in opts].index(gold),
                                  task="forms", images=[f"img{i}.png"]))
    rows = make_pairs(rows)
    paired = [r for r in rows if r.pair_id]
    assert len(paired) == 20
    for pid in {r.pair_id for r in paired}:
        a, b = [r for r in paired if r.pair_id == pid]
        assert a.options[a.label][0] != b.options[b.label][0] and a.images != b.images
    rows = add_ablations(rows, 0.5, seed=1)
    abl = [r for r in rows if r.ablation]
    assert abl and all(r.images == [] and r.weight == 0.0 and r.source_id for r in abl)
    rep = balance_report(rows)
    assert rep["forms"]["paired_share"] == 1.0 and not rep["forms"]["warn"]
    assert not rep["forms/ablation"]["warn"]  # ablation copies are not expected to pair
    tr, va, held = split_by_source(rows, 0.3, seed=0)
    assert not held and len(tr) + len(va) == len(rows)
    side = {}
    for name, part in (("tr", tr), ("va", va)):
        for r in part:
            for k in [r.pair_id, r.source_id, *r.images]:
                if k:
                    assert side.setdefault(k, name) == name, f"{k} split across train and val"
    # An ablation copy follows its source image to the same side.
    srcs_va = {r.images[0] for r in va if r.images}
    assert all(r.source_id in srcs_va for r in va if r.ablation)


def test_split_holds_out_task_families():
    rows = [VisionExample(kind="noul", state="", instructions="q", options=[["false", ""], ["true", ""]],
                          label=i % 2, task=t, images=[f"{t}{i}.png"]) for t in ("a", "b") for i in range(10)]
    rows = add_ablations(rows, 1.0)
    tr, va, held = split_by_source(rows, 0.2, held_out_tasks=["b"])
    assert {r.task for r in held} == {"b", "b/ablation"}
    assert all(r.task.startswith("a") for r in tr + va)


def test_ablation_skips_rows_the_frozen_readout_cannot_score():
    labels = {f"c{i}": "" for i in range(12)}
    rows = from_labelled_images([{"image": "x.png", "label": "c3"}], labels, task="t", instructions="?")
    assert rows[0].n_options == 12
    assert len(add_ablations(rows, 1.0)) == 1


def test_converters():
    rows = from_docvqa_yesno([{"image": "a.png", "question": "Is it signed?", "answers": ["Yes"]},
                              {"image": "b.png", "question": "What date?", "answers": ["1999"]}], "/root")
    assert len(rows) == 1 and rows[0].label == 1 and rows[0].images == ["/root/a.png"]
    labels = {f"c{i}": f"desc {i}" for i in range(10)}
    rows = from_labelled_images([{"image": "x.png", "label": "c3"}], labels, task="t",
                                instructions="What is it?", n_options=5, seed=0)
    r = rows[0]
    assert len(r.options) == 5 and r.options[r.label][0] == "c3"
    q = r.to_question()
    assert isinstance(q, ChoiceQuestion) and len(q.criteria) == 5
    # A fixed option set per task (n_options=None) is what lets rows pair.
    recs = [{"image": f"{i}.png", "label": f"c{i % 3}"} for i in range(6)]
    fixed = from_labelled_images(recs, {f"c{i}": "" for i in range(3)}, task="t", instructions="?",
                                 image_root="/root")
    assert all(len(x.options) == 3 for x in fixed) and fixed[0].images == ["/root/0.png"]
    assert sum(bool(x.pair_id) for x in make_pairs(fixed)) == 6
    sampled = from_labelled_images(recs, {f"c{i}": "" for i in range(10)}, task="t2",
                                   instructions="?", n_options=4)
    assert sum(bool(x.pair_id) for x in make_pairs(sampled)) <= 2


def test_option_index_matches_upstream():
    torch = pytest.importorskip("torch")  # the upstream rule lives on a torch-importing class
    from strands_decider.data.collate import SystemOneCollator
    q = ChoiceQuestion(instructions="Which team?", criteria={"billing": "money", "technical": "bugs"})
    prompt, rq = build_vision_prompt(VisionState(images=("s.png",), text="t"), q)
    expanded = expand_image_tokens(prompt, [7])
    enc = FakeTok()(expanded)
    base = question_base(expanded, rq)
    assert option_token_index(enc["offset_mapping"], rq.option_spans, base) == \
        SystemOneCollator.option_token_index(enc["offset_mapping"], rq.option_spans, base)
