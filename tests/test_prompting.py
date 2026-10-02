"""Torch-free tests: prompt rendering, image-token expansion, option position math."""
import sys
sys.path.insert(0, "/home/claude/strands-decider/src")
sys.path.insert(0, "/home/claude/vision-decider")

from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from vision_decider.prompting import (
    IMAGE_PAD, VisionState, build_vision_prompt, expand_image_tokens,
    image_tokens_for_grid, option_token_index, question_base, render_vision_state,
)
from vision_decider.data import (
    VisionExample, add_ablations, balance_report, make_pairs, split_by_pairs,
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
        rows.append(VisionExample(kind="noul", state="", instructions="Is the form signed?",
                                  options=[["false", "no"], ["true", "yes"]], label=i % 2,
                                  task="forms", images=[f"img{i}.png"]))
    rows = make_pairs(rows)
    paired = [r for r in rows if r.pair_id]
    assert len(paired) == 20
    for pid in {r.pair_id for r in paired}:
        a, b = [r for r in paired if r.pair_id == pid]
        assert a.label != b.label and a.images != b.images
    rows = add_ablations(rows, 0.5, seed=1)
    abl = [r for r in rows if r.ablation]
    assert abl and all(r.images == [] for r in abl)
    rep = balance_report(rows)
    assert rep["forms"]["paired_share"] == 1.0 and not rep["forms"]["warn"]
    tr, va = split_by_pairs(rows, 0.3, seed=0)
    pid_tr = {r.pair_id for r in tr if r.pair_id}
    pid_va = {r.pair_id for r in va if r.pair_id}
    assert not (pid_tr & pid_va)


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


if __name__ == "__main__":
    import inspect
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            fn(); print("ok", name)
