#!/usr/bin/env python3
"""Dedicated Qwen3-VL-Reranker host server (ADR-0021 / docs/artifact/visual-rag.md Stage 4).

Serves `mlx-community/Qwen3-VL-Reranker-2B-mxfp8` via **mlx_vlm** — a vision-language
cross-encoder reranker. Scoring is the SAME yes/no causal-logit method as the text
Qwen3-Reranker (arXiv 2601.04720), mirroring the validated bench
`backend/tests/benchmark/shared/rerankers.py::MLXVLReranker` — production MUST match
the bench, NOT the mxfp8 HF card (whose `logits_per_image`/sigmoid blurb is a
copy-pasted CLIP/embedding-card artifact; a reranker emits no `logits_per_image`).

Per candidate: wrap `(query, document)` in the fixed judge chat-template, one forward,
score = softmax([no_logit, yes_logit])[yes] on the last-token logits.
  - TEXT candidate  -> route through `model.language_model(ids)`.
  - IMAGE candidate -> full vision forward (`apply_chat_template` + processor with the
    image); area-capped (~4MP) before the forward to bound vision tokens / avoid OOM
    on tall infographics. This image path is the reranker's actual payoff (page
    images: figures/tables/layout) over a text reranker that only sees OCR.

One model, one process, one port (ADR fleet/009 — no co-serving). A worker thread owns
the Metal stream; async endpoints submit work and await futures.

  POST /rerank  {"query": str,
                 "documents": [ {"text": str} | {"image": "<data-uri|base64>"} ],
                 "instruction"?: str}
                -> {"scores": [float], "input_tokens": int}   # scores in request order
  GET  /health  -> {"ok": bool, "model": str}

Run:  SHUBO_RERANKER_MODEL=mlx-community/Qwen3-VL-Reranker-2B-mxfp8 \
      SHUBO_RERANKER_PORT=<port> python server.py
Deps: mlx-vlm, fastapi, uvicorn, pillow, numpy (see requirements.txt).
"""
import asyncio
import base64
import io
import logging
import os
import queue
import re
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

MODEL_ID = os.environ.get("SHUBO_RERANKER_MODEL", "mlx-community/Qwen3-VL-Reranker-2B-mxfp8")
PORT = int(os.environ.get("SHUBO_RERANKER_PORT", os.environ.get("PORT", "12414")))  # shared_server_port reranker <model>
HOST = os.environ.get("SHUBO_RERANKER_HOST", "0.0.0.0")
MAX_LEN = int(os.environ.get("SHUBO_RERANKER_MAXLEN", "4096"))
MAX_PIXELS = int(os.environ.get("SHUBO_RERANKER_MAX_PIXELS", "4000000"))
DEFAULT_INSTRUCTION = os.environ.get(
    "SHUBO_RERANKER_INSTRUCTION",
    # Matches the bench default (rerankers.py:_DEFAULT_INSTRUCTION) so served scores == offline scores.
    "Given a question, retrieve the passage that contains the evidence needed to answer it",
)

# The fixed Qwen3-Reranker judge chat-template the model was trained on — kept
# byte-identical to rerankers.py:_RERANK_PREFIX/_RERANK_SUFFIX so served scores ==
# offline bench scores.
_RERANK_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the '
    'Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_DATA_URI = re.compile(r"^data:[^;]+;base64,(?P<data>.+)$", re.DOTALL)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reranker")
app = FastAPI(title="qwen3-vl-reranker", version="0.1.0")

# (query, documents, instruction, loop, future) — drained by the single model thread.
_work: "queue.Queue" = queue.Queue()
_ready = threading.Event()


def _decode_image(s: str) -> Image.Image:
    m = _DATA_URI.match(s)
    raw = base64.b64decode(m.group("data") if m else s)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _rerank_pair(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def _worker() -> None:
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template

    logger.info("loading %s …", MODEL_ID)
    model, processor = load(MODEL_ID)
    htok = getattr(processor, "tokenizer", processor)
    yes_id = htok.convert_tokens_to_ids("yes")
    no_id = htok.convert_tokens_to_ids("no")
    pre = htok.encode(_RERANK_PREFIX, add_special_tokens=False)
    suf = htok.encode(_RERANK_SUFFIX, add_special_tokens=False)
    budget = MAX_LEN - len(pre) - len(suf)
    logger.info("model ready: %s", MODEL_ID)
    _ready.set()

    def score_text(query, doc, instruction):
        body = htok.encode(_rerank_pair(instruction, query, doc), add_special_tokens=False)[:budget]
        ids = mx.array([pre + body + suf])
        logits = model.language_model(ids).logits[0, -1, :]
        return float(mx.softmax(mx.stack([logits[no_id], logits[yes_id]]))[1])

    def score_image(query, img, instruction):
        img = img.convert("RGB")
        w, h = img.size
        if w * h > MAX_PIXELS:
            s = (MAX_PIXELS / (w * h)) ** 0.5
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
        msg = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>:"
        prompt = apply_chat_template(processor, model.config, msg, num_images=1)
        enc = processor(text=prompt, images=[img], return_tensors="np")
        ids = mx.array(enc["input_ids"])
        pix = mx.array(enc["pixel_values"])
        grid = mx.array(enc["image_grid_thw"]) if "image_grid_thw" in enc else None
        res = model(ids, pix, image_grid_thw=grid) if grid is not None else model(ids, pix)
        logits = (res.logits if hasattr(res, "logits") else res)[0, -1, :]
        return float(mx.softmax(mx.stack([logits[no_id], logits[yes_id]]))[1])

    while True:
        query, documents, instruction, loop, fut = _work.get()
        try:
            scores = []
            for d in documents:
                if d.get("image"):
                    scores.append(score_image(query, _decode_image(d["image"]), instruction))
                else:
                    scores.append(score_text(query, d.get("text", ""), instruction))
            loop.call_soon_threadsafe(fut.set_result, (scores, len(documents)))
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(fut.set_exception, exc)


threading.Thread(target=_worker, daemon=True, name="reranker-worker").start()


class Document(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None


class RerankRequest(BaseModel):
    query: str
    documents: list[Document]
    instruction: Optional[str] = None


@app.get("/health")
async def health():
    return {"ok": _ready.is_set(), "model": MODEL_ID}


@app.post("/rerank")
async def rerank(req: RerankRequest):
    if not _ready.is_set():
        raise HTTPException(503, "model not ready")
    if not req.documents:
        return {"scores": [], "input_tokens": 0}
    docs = [d.model_dump() for d in req.documents]
    instruction = req.instruction or DEFAULT_INSTRUCTION
    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    _work.put((req.query, docs, instruction, loop, fut))
    scores, ntok = await fut
    return {"scores": scores, "input_tokens": ntok}


if __name__ == "__main__":
    import uvicorn

    logger.info("starting reranker server on %s:%d (model: %s)", HOST, PORT, MODEL_ID)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
