# vision-decider

Strands Decider's pointer readout on the full Qwen3.5-2B multimodal torso. Images go
inside `<state>`; the pointer head, the three question types (noul, choice, score), the
confidence formulas and calibration are inherited from `strands_decider` unchanged.

Install next to the Decider repo:

    pip install strands-decider          # or pip install -e ../strands-decider
    pip install -e .                     # this package

Build data (torch-free), train, calibrate, serve:

    python -c "from vision_decider.data import *"   # converters: from_docvqa_yesno, from_labelled_images, from_ordinal_ratings
    python -m vision_decider.train configs/train-vision.yaml
    python -m vision_decider.evaluate calibrate checkpoints/vision-decider-2b-v1 data/vision/calib.jsonl
    python -m vision_decider.evaluate eval checkpoints/vision-decider-2b-v1 data/vision/test.jsonl
    python -m vision_decider.server checkpoints/vision-decider-2b-v1 --port 8099
    python scripts/ask.py checkpoints/vision-decider-2b-v1 page3.png --noul "Is the signature block filled in?"

`strands-decider calibrate` feeds no pixels, so calibration of image questions goes
through `vision_decider.evaluate`, which reuses upstream's temperature fitting.

Tests, CPU only: `pytest tests/`. `test_prompting.py` needs neither torch nor weights.
`test_tiny_model.py` builds a random-weight Qwen3.5 with the real tokenizer and image
processor (only small config files are downloaded) and runs every module end to end:
LoRA scope, collation, the loss terms, prefix-cache exactness against a full forward,
save/load, training, calibration and the server.

Status: written against strands-decider main (1 Oct 2026) and transformers 5.18.0
(`Qwen3_5ForConditionalGeneration`, `Qwen3VLProcessor`). Every module runs end to end on
CPU with a tiny model; nothing has been trained on real data or run on a GPU yet.
