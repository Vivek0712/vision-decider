"""POST /v1/systemone with an optional `images` field (base64 PNG/JPEG list).

Same request shape as strands-decider's server, plus `images`. Binds to 127.0.0.1 and
has no auth; wrap it in AgentCore Runtime or a Lambda/Fargate front for anything shared.
Run: python -m vision_decider.server CHECKPOINT --port 8099 --device cuda
"""

from __future__ import annotations

import argparse
import base64
import io
import threading
from typing import Any

import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from strands_decider.schema import Question, SystemOneResponse

from .infer import load_engine
from .prompting import VisionState


class VisionRequest(BaseModel):
    state: Any = ""
    images: list[str] = []          # base64-encoded image bytes
    model: str = "vision-decider-latest"
    questions: dict[str, Question]


def build_app(ckpt: str, device: str, image_long_side: int) -> FastAPI:
    engine = load_engine(ckpt, device=device, image_long_side=image_long_side)
    app = FastAPI(title="vision-decider")
    # FastAPI runs sync handlers on a thread pool; one model on one device serves one
    # request at a time.
    lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"model": engine.cfg.model_name, "checkpoint": ckpt, "device": device,
                "base_model": engine.model.config.base_model, "image_long_side": image_long_side}

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(req: VisionRequest) -> SystemOneResponse:
        pil = tuple(Image.open(io.BytesIO(base64.b64decode(b))) for b in req.images)
        with lock:
            return engine.ask_vision(VisionState(images=pil, text=req.state), req.questions)

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image-long-side", type=int, default=448)
    a = ap.parse_args()
    uvicorn.run(build_app(a.checkpoint, a.device, a.image_long_side), host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
