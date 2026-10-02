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
    strands-decider calibrate checkpoints/vision-decider-2b-v1    # temperatures, unchanged tool
    python -m vision_decider.server checkpoints/vision-decider-2b-v1 --port 8099
    python scripts/ask.py checkpoints/vision-decider-2b-v1 page3.png --noul "Is the signature block filled in?"

Tests that need no GPU or weights: `python tests/test_prompting.py`.

Status: written against strands-decider main (Oct 2026) and transformers 5.18
(`Qwen3_5ForConditionalGeneration`, `Qwen3VLProcessor`). Logic modules are tested;
the model, collator, trainer and engine have not been run on a GPU yet.
