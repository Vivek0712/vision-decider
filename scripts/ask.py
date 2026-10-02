"""python scripts/ask.py CKPT image.png --noul "Is the form signed?" --choice "Which button?=Submit,Cancel,Back" """
import argparse, json, sys
sys.path.insert(0, ".")
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from vision_decider.infer import load_engine
from vision_decider.prompting import VisionState

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint"); ap.add_argument("images", nargs="*")
ap.add_argument("--state", default=""); ap.add_argument("--device", default="cuda")
ap.add_argument("--noul", action="append", default=[]); ap.add_argument("--choice", action="append", default=[])
ap.add_argument("--score", action="append", default=[])
a = ap.parse_args()
qs = {}
for i, t in enumerate(a.noul):
    qs[f"noul_{i}"] = NoulQuestion(instructions=t)
for i, t in enumerate(a.choice):
    q, opts = t.split("=", 1)
    qs[f"choice_{i}"] = ChoiceQuestion(instructions=q, criteria={o.strip(): "" for o in opts.split(",")})
for i, t in enumerate(a.score):
    q, lv = t.split("=", 1)
    qs[f"score_{i}"] = ScoreQuestion(instructions=q, criteria=[o.strip() for o in lv.split(",")])
eng = load_engine(a.checkpoint, device=a.device)
print(json.dumps(eng.ask_vision(VisionState(images=tuple(a.images), text=a.state), qs).model_dump(), indent=2))
