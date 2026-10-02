"""A tiny random-weight Qwen3.5 multimodal checkpoint for CPU tests.

Same architecture and config keys as Qwen/Qwen3.5-2B-Base, shrunk (hidden 64, 3 GDN +
1 full-attention layer, a 2-layer ViT), with the REAL tokenizer and image processor so
token ids, image-pad expansion and M-RoPE geometry are the production ones. Only the
small config and tokenizer files are downloaded, never the weights. Tests that need it
skip when torch/transformers are missing or the files cannot be fetched.
"""

from __future__ import annotations

import json
import shutil

import pytest

BASE = "Qwen/Qwen3.5-2B-Base"
FILES = ["config.json", "merges.txt", "preprocessor_config.json", "tokenizer.json",
         "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json"]


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    hub = pytest.importorskip("huggingface_hub")
    dst = tmp_path_factory.mktemp("tiny-qwen35")
    try:
        src = {f: hub.hf_hub_download(BASE, f) for f in FILES}
    except Exception as e:  # offline, gated, proxy
        pytest.skip(f"cannot fetch {BASE} tokenizer/config: {e}")

    cfg = json.load(open(src["config.json"]))
    t = cfg["text_config"]
    t.update(hidden_size=64, num_hidden_layers=4, intermediate_size=128, head_dim=64,
             num_attention_heads=2, num_key_value_heads=1,
             linear_num_key_heads=4, linear_num_value_heads=4,
             linear_key_head_dim=16, linear_value_head_dim=16,
             layer_types=["linear_attention"] * 3 + ["full_attention"],
             mtp_num_hidden_layers=0, dtype="float32")
    t["rope_parameters"]["mrope_section"] = [3, 3, 2]  # 64 * 0.25 = 16 rotary dims
    cfg["vision_config"].update(depth=2, hidden_size=64, num_heads=4, intermediate_size=128,
                                out_hidden_size=64)
    cfg.pop("transformers_version", None)
    config = transformers.Qwen3_5Config(
        **{k: v for k, v in cfg.items() if k not in ("architectures", "model_type")})
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    torch.manual_seed(0)
    transformers.Qwen3_5ForConditionalGeneration(config).to(torch.float32).save_pretrained(dst)
    for f in FILES[1:]:
        shutil.copy(src[f], dst / f)
    return str(dst)
