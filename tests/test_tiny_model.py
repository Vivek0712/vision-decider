"""End-to-end tests on a tiny random-weight Qwen3.5 (see conftest.py), CPU, fp32.

Each test pins a defect found in the audit: model construction (hidden size on the
composite config), mm_token_type_ids, LoRA scope, target widths, the frozen-KL and
ablation losses, engine construction, M-RoPE positions on the shared-prefix path, cross-
request state, save/load, the training entry point and the server.
"""

from __future__ import annotations

import base64
import io
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from PIL import Image  # noqa: E402

from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion  # noqa: E402

from vision_decider.collate import VisionCollator, VisionCollatorConfig  # noqa: E402
from vision_decider.data import VisionExample, add_ablations, write_jsonl  # noqa: E402
from vision_decider.modeling import VisionDeciderConfig, VisionDeciderModel  # noqa: E402
from vision_decider.prompting import VisionState  # noqa: E402


def _img(w: int, h: int, seed: int) -> Image.Image:
    g = torch.Generator().manual_seed(seed)
    return Image.fromarray((torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8).numpy())


def _cfg(base: str, **kw) -> VisionDeciderConfig:
    return VisionDeciderConfig(base_model=base, torch_dtype="float32", max_length=1024, **kw)


@pytest.fixture(scope="module")
def model(tiny_base):
    torch.manual_seed(0)
    m = VisionDeciderModel.from_pretrained_base(_cfg(tiny_base, kl_frozen_weight=0.3))
    # Fresh LoRA B is zero, which would make adapter-on equal adapter-off; perturb it so
    # the tests can tell the two apart.
    with torch.no_grad():
        for n, p in m.torso.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    return m


@pytest.fixture(scope="module")
def processor(tiny_base):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(tiny_base)


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    d = tmp_path_factory.mktemp("img")
    out = []
    for i, (w, h) in enumerate([(448, 448), (600, 800), (320, 320), (1000, 200)]):
        p = d / f"{i}.png"
        _img(w, h, i).save(p)
        out.append(str(p))
    return out


def _rows(images):
    return [
        VisionExample(kind="noul", state="Page 3 of a lease.", instructions="Is it signed?",
                      options=[["false", "no signature"], ["true", "signed by both parties"]],
                      label=1, task="forms", images=[images[0]]),
        VisionExample(kind="choice", state="", instructions="Which control next?",
                      options=[["submit", "the blue Submit button"], ["cancel", "the grey Cancel link"],
                               ["back", "return to the cart"]],
                      label=1, task="ui", images=[images[1]]),
        VisionExample(kind="score", state="", instructions="How sharp is the photo?",
                      options=[["0", "blurred"], ["1", "soft"], ["2", "sharp"], ["3", "very sharp"]],
                      label=2, task="quality", images=[images[2]]),
        VisionExample(kind="noul", state="Plain text row.", instructions="Is the text urgent?",
                      options=[["false", "not urgent"], ["true", "urgent"]],
                      label=0, task="text"),
    ]


# ---- construction ----------------------------------------------------------------

def test_lora_scope_and_frozen_vision(tiny_base):
    m = VisionDeciderModel.from_pretrained_base(_cfg(tiny_base))
    names = [n for n, _ in m.torso.named_modules() if n.endswith("lora_A")]
    assert names and all(".language_model." in n for n in names)
    trainable = [n for n, p in m.torso.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in trainable)
    assert not any(".visual." in n for n in trainable)


def test_lora_on_merger_attaches_and_stays_trainable(tiny_base):
    m = VisionDeciderModel.from_pretrained_base(_cfg(tiny_base, lora_on_merger=True))
    merger = [n for n, p in m.torso.named_parameters() if ".visual.merger." in n and p.requires_grad]
    assert merger and all("lora_" in n for n in merger)
    assert not any(".visual.blocks." in n for n, p in m.torso.named_parameters() if p.requires_grad)


# ---- collation ---------------------------------------------------------------------

def test_collator_positions_widths_and_drops(model, processor, images):
    ccfg = VisionCollatorConfig(num_slots=24, max_length=1024, head_type="pointer")
    coll = VisionCollator(processor, ccfg, train=False)
    b = coll(_rows(images))
    tok = processor.tokenizer
    for r, ex in enumerate(_rows(images)):
        for k, idx in enumerate(b["opt_idx"][r].tolist()[: ex.n_options]):
            # The last token of each option line ends that option's description.
            assert ex.options[k][1].endswith(tok.decode(b["input_ids"][r, idx]).strip())
    width = b["opt_idx"].size(1)
    assert b["label_dist"].shape == (4, width)
    assert int((b["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum()) == 196 + 140 + 100

    small = VisionCollator(processor, VisionCollatorConfig(num_slots=24, max_length=1024, head_type="pointer",
                                                           max_image_tokens=150), train=False)
    kept = small(_rows(images))
    assert kept["row_index"].tolist() == [1, 2, 3]  # the 196-token image is dropped, not cut
    assert int(kept["image_grid_thw"].size(0)) == 2


# ---- training losses ------------------------------------------------------------------

def test_train_step_grads_only_on_adapter_and_head(model, processor, images):
    from vision_decider.train import _forward

    coll = VisionCollator(processor, VisionCollatorConfig(num_slots=24, max_length=1024, head_type="pointer"))
    b = coll(_rows(images))
    model.train()
    model.zero_grad(set_to_none=True)
    out = _forward(model, b, labels=b["labels"], label_dist=b["label_dist"], weights=b["weights"])
    ref, eligible = model.frozen_slot_log_probs(b["input_ids"], b["attention_mask"], b["n_slots"],
                                                pixel_values=b["pixel_values"], image_grid_thw=b["image_grid_thw"])
    assert ref.shape[0] == 4 and bool(eligible.all())
    stu = out["log_probs"][eligible]
    ref = ref[eligible][:, : stu.size(-1)]
    valid = torch.isfinite(ref) & torch.isfinite(stu)
    kl = (ref.exp().masked_fill(~valid, 0) * (ref - stu).masked_fill(~valid, 0)).sum(-1).mean()
    loss = out["loss"] + 0.3 * kl
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    got = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert got and all("lora_" in n or n.startswith("head.") for n in got)
    assert not any(".visual." in n for n in got)
    model.zero_grad(set_to_none=True)


def test_frozen_readout_sees_the_image(model, processor, images):
    coll = VisionCollator(processor, VisionCollatorConfig(num_slots=24, max_length=1024, head_type="pointer"),
                          train=False)
    b = coll(_rows(images)[:1])
    a, _ = model.frozen_slot_log_probs(b["input_ids"], b["attention_mask"], b["n_slots"],
                                       pixel_values=b["pixel_values"], image_grid_thw=b["image_grid_thw"])
    blank = b["pixel_values"] * 0
    z, _ = model.frozen_slot_log_probs(b["input_ids"], b["attention_mask"], b["n_slots"],
                                       pixel_values=blank, image_grid_thw=b["image_grid_thw"])
    assert not torch.allclose(a[:, :2], z[:, :2])


# ---- inference -------------------------------------------------------------------------

QUESTIONS = {
    "signed": NoulQuestion(instructions="Is the form signed?"),
    "button": ChoiceQuestion(instructions="Which control next?",
                             criteria={"submit": "the blue Submit button", "cancel": "the grey Cancel link"}),
    "sharp": ScoreQuestion(instructions="How sharp?", criteria=["blurred", "soft", "sharp"]),
}


@pytest.fixture(scope="module")
def engine(model):
    from vision_decider.infer import VisionDeciderEngine, VisionEngineConfig
    return VisionDeciderEngine(model, VisionEngineConfig(), device="cpu")


def _reference(model, processor, state: VisionState, q) -> list[float]:
    """Plain full forward of prefix + one question, positions left to transformers."""
    from vision_decider.prompting import build_vision_prompt, expand_image_tokens, image_tokens_for_grid
    from vision_decider.collate import _load_image
    from strands_decider.modeling import masked_log_softmax

    prompt, rq = build_vision_prompt(state, q)
    mm = {}
    counts = []
    if state.images:
        img = processor.image_processor(images=[_load_image(i, 448) for i in state.images], return_tensors="pt")
        counts = image_tokens_for_grid(img["image_grid_thw"].tolist(), processor.image_processor.merge_size)
        mm = {"pixel_values": img["pixel_values"], "image_grid_thw": img["image_grid_thw"]}
    expanded = expand_image_tokens(prompt, counts)
    enc = processor.tokenizer(expanded, return_offsets_mapping=True)
    from vision_decider.prompting import option_token_index, question_base
    opt = option_token_index(enc["offset_mapping"], rq.option_spans, question_base(expanded, rq))
    ids = torch.tensor([enc["input_ids"]])
    with torch.no_grad():
        out = model(ids, torch.ones_like(ids), torch.tensor([rq.n_slots]), opt_idx=torch.tensor([opt]),
                    temperature=1.0, **mm)
    return masked_log_softmax(out["logits"], torch.tensor([rq.n_slots])).exp()[0, : rq.n_slots].tolist()


@pytest.mark.parametrize("which", ["one_image", "two_images", "text_only"])
def test_prefix_cache_matches_full_forward(engine, model, processor, images, which):
    from strands_decider.prompting import render_question

    imgs = {"one_image": (images[1],), "two_images": (images[0], images[3]), "text_only": ()}[which]
    state = VisionState(images=imgs, text="Checkout page after the user tapped Pay.")
    rendered = [render_question(q) for q in QUESTIONS.values()]
    probs, _ = engine._vision_probs(state, rendered, [rq.kind for rq in rendered])
    for i, q in enumerate(QUESTIONS.values()):
        ref = _reference(model, processor, state, q)
        got = probs[i, : len(ref)].tolist()
        assert max(abs(x - y) for x, y in zip(got, ref, strict=True)) < 1e-5, (which, i, got, ref)


def test_text_request_after_image_request(engine, images):
    state = VisionState(images=(), text="Help! My payouts have been failing for 3 days.")
    before = engine.ask(state.text, {"q": QUESTIONS["signed"]}).answers["q"]
    engine.ask_vision(VisionState(images=(images[1],), text="x"), {"q": QUESTIONS["signed"]})
    after = engine.ask(state.text, {"q": QUESTIONS["signed"]}).answers["q"]
    assert before == after


# ---- persistence, training entry point, server -------------------------------------------

def test_save_load_round_trip_and_upstream_config(model, tmp_path, images):
    from strands_decider.modeling import StrandsDeciderConfig, config_path
    from vision_decider.infer import VisionDeciderEngine, VisionEngineConfig

    model.save_pretrained(str(tmp_path))
    StrandsDeciderConfig.from_json(config_path(str(tmp_path)))  # upstream tools can read it
    loaded = VisionDeciderModel.load(str(tmp_path))
    assert loaded.config.image_long_side == model.config.image_long_side
    state = VisionState(images=(images[0],), text="t")
    e1 = VisionDeciderEngine(model, VisionEngineConfig(), device="cpu").ask_vision(state, QUESTIONS)
    e2 = VisionDeciderEngine(loaded, VisionEngineConfig(), device="cpu").ask_vision(state, QUESTIONS)
    assert e1.model_dump()["answers"] == e2.model_dump()["answers"]

    import shutil
    shutil.rmtree(tmp_path / "lora")
    with pytest.raises(FileNotFoundError):
        VisionDeciderModel.load(str(tmp_path))


def test_train_main_then_calibrate(tiny_base, images, tmp_path):
    import yaml
    from vision_decider import evaluate, train

    rows = []
    for i in range(16):
        r = _rows(images)[i % 4]
        r.images = list(r.images)
        rows.append(VisionExample.from_dict({**r.to_dict(), "source_id": f"s{i}"}))
    rows = add_ablations(rows, 0.5, seed=0)
    data = tmp_path / "rows.jsonl"
    write_jsonl(rows, data)
    teacher = tmp_path / "teacher.jsonl"
    with open(teacher, "w") as fh:
        fh.write(json.dumps({"i": 0, "probs": [0.3, 0.7]}) + "\n")
    cfg = {
        "train_files": [str(data)], "val_fraction": 0.25, "base_model": tiny_base, "torch_dtype": "float32",
        "num_slots": 24, "head_type": "pointer", "pointer_dim": 32, "kl_frozen_weight": 0.3,
        "teacher_file": str(teacher), "teacher_weight": 1.0, "max_length": 1024,
        "lora_r": 4, "lora_alpha": 8, "epochs": 2, "micro_batch_size": 2, "grad_accum": 3,
        "lr": 1e-3, "head_lr": 1e-2, "gradient_checkpointing": False, "group_by_length": True,
        "length_group_mega": 2, "output_dir": str(tmp_path / "ckpt"), "log_every": 1, "device": "cpu",
    }
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(cfg))
    train.main(str(tmp_path / "train.yaml"))
    hist = json.loads((tmp_path / "ckpt" / "train_config.json").read_text())["history"]
    assert hist and all(h["val_loss"] == h["val_loss"] for h in hist)  # not NaN

    eval_rows = [r for r in rows if not r.ablation]
    res = evaluate.calibrate(str(tmp_path / "ckpt"), eval_rows, device="cpu")
    assert "after" in res
    reloaded = VisionDeciderModel.load(str(tmp_path / "ckpt"))
    assert reloaded.config.temperature == pytest.approx(res["temperature"])


def test_server_accepts_base64_images(model, tmp_path, images, monkeypatch):
    from fastapi.testclient import TestClient
    from vision_decider import server

    model.save_pretrained(str(tmp_path))
    app = server.build_app(str(tmp_path), "cpu", 448)
    client = TestClient(app)
    buf = io.BytesIO()
    Image.open(images[0]).save(buf, format="PNG")
    body = {"state": "Checkout page.", "images": [base64.b64encode(buf.getvalue()).decode()],
            "questions": {"signed": {"type": "noul", "instructions": "Is it signed?"}}}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["model"].startswith("vision-decider") and "signed" in out["answers"]
    assert client.post("/v1/systemone", json={**body, "images": []}).status_code == 200


def test_ablation_rows_carry_no_label_loss(model, processor, images):
    """A weight-0 ablation row must not move the label loss: KL is its only target."""
    from vision_decider.train import _forward

    coll = VisionCollator(processor, VisionCollatorConfig(num_slots=24, max_length=1024, head_type="pointer"),
                          train=False)
    labelled = _rows(images)[:1]
    abl = [r for r in add_ablations(_rows(images)[:1], 1.0) if r.ablation]
    assert abl and abl[0].weight == 0.0
    with torch.no_grad():
        a = _forward(model, coll(labelled), labels=coll(labelled)["labels"], weights=coll(labelled)["weights"])
        both = coll(labelled + abl)
        b = _forward(model, both, labels=both["labels"], weights=both["weights"])
    assert float(a["loss"]) == pytest.approx(float(b["loss"]), abs=1e-6)
