"""granite-docling MLX host server (plain FastAPI MVP) — Metal-accelerated document parsing.

The simplest deployable form (mirrors buckle's qwen3-asr-server.py): one model, single process.
For multi-replica throughput use the Ray Serve front (`serve_app.py`) — same core, wrapped.

Run:  uvicorn server:app --host 0.0.0.0 --port $PORT
"""
import base64

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from granite_docling import GraniteDocling, render_pdf, decode_image, DEFAULT_MODEL, DEFAULT_PDF_DPI

app = FastAPI(title="granite-docling-mlx", version="0.1.0")
engine = GraniteDocling()


class ConvertRequest(BaseModel):
    image_b64: str | None = None   # a single page image (base64)
    pdf_b64: str | None = None     # a PDF (base64); rendered to page images here


@app.get("/health")
def health():
    return {"status": "ok", "model": DEFAULT_MODEL, "loaded": engine.loaded}


@app.post("/convert")
def convert(req: ConvertRequest):
    if req.pdf_b64:
        images = render_pdf(base64.b64decode(req.pdf_b64), DEFAULT_PDF_DPI)
    elif req.image_b64:
        images = [decode_image(base64.b64decode(req.image_b64))]
    else:
        raise HTTPException(status_code=400, detail="provide image_b64 or pdf_b64")
    if not images:
        raise HTTPException(status_code=400, detail="no pages to convert")
    try:
        return engine.convert(images)
    except Exception as exc:  # noqa: BLE001 — surface the failure to the caller
        raise HTTPException(status_code=500, detail=f"docling conversion failed: {exc}") from exc


@app.on_event("startup")
def _warm():
    import os
    if os.environ.get("SHUBO_DOCLING_EAGER_LOAD", "true").lower() == "true":
        engine.load()
