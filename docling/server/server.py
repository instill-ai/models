"""granite-docling MLX host server (plain FastAPI MVP) — Metal-accelerated document parsing.

The simplest deployable form (mirrors buckle's qwen3-asr-server.py): one model, single process.
For multi-replica throughput use the Ray Serve front (`serve_app.py`) — same core, wrapped.

Run:  uvicorn server:app --host 0.0.0.0 --port $PORT
"""
import base64
import json
import re

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import BaseModel

from granite_docling import GraniteDocling, render_pdf, decode_image, DEFAULT_MODEL, DEFAULT_PDF_DPI

app = FastAPI(title="granite-docling-mlx", version="0.1.0")
engine = GraniteDocling()

_DATA_URI = re.compile(r"^data:(?P<mime>[^;,]*)(?:;base64)?,(?P<data>.*)$", re.DOTALL)


def _images_from_doc(doc_content: str):
    """A data-uri (or bare base64) document -> page images (PDF rendered, image decoded)."""
    m = _DATA_URI.match(doc_content)
    raw = base64.b64decode(m.group("data") if m else doc_content)
    mime = (m.group("mime") if m else "") or ""
    if "pdf" in mime.lower() or raw[:5] == b"%PDF-":
        return render_pdf(raw, DEFAULT_PDF_DPI)
    return [decode_image(raw)]


class ConvertRequest(BaseModel):
    image_b64: str | None = None   # a single page image (base64)
    pdf_b64: str | None = None     # a PDF (base64); rendered to page images here


@app.get("/health")
def health():
    return {"status": "ok", "model": DEFAULT_MODEL, "loaded": engine.loaded}


# NOTE: async so the handler runs on the event-loop thread — the SAME thread that loads the MLX
# model (startup) — because MLX's Metal stream is thread-local (a threadpool worker, which FastAPI
# would use for a sync `def`, raises "no Stream(gpu, 0)"). The generate blocks this single-process
# server for ~2.3s/page; for concurrent throughput use the Ray Serve front (serve_app.py).
@app.post("/convert")
async def convert(req: ConvertRequest):
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


@app.post("/v1alpha/namespaces/{namespace}/models/{model}/versions/{version}/trigger")
async def trigger(namespace: str, model: str, version: str, request: Request):
    """model-backend-trigger-compatible drop-in: the parsing-router's `docling` step posts
    {taskInputs:[{data:{doc_content:<data-uri>}}]} and reads body.taskOutputs[0].data. Pointing the
    step's endpoint at this server makes it a drop-in for the served `models/docling` model — the
    response data carries markdown_pages + structured_document, exactly what routedConvertResultParser
    consumes. (Async for the same MLX-thread reason as /convert.)

    Parse the body leniently (Request, not `dict = Body(...)`) so a quirk in the caller's body
    encoding surfaces as a logged, actionable 400 rather than an opaque FastAPI 422."""
    raw = await request.body()
    try:
        body = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[trigger] non-JSON body ({exc}); len={len(raw)} head={raw[:200]!r} tail={raw[-120:]!r}",
              flush=True)
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
    try:
        doc_content = body["taskInputs"][0]["data"]["doc_content"]
    except (KeyError, IndexError, TypeError):
        shape = list(body) if isinstance(body, dict) else type(body).__name__
        print(f"[trigger] unexpected body shape: {shape}", flush=True)
        raise HTTPException(status_code=400, detail="expected taskInputs[0].data.doc_content (data-uri)")
    images = _images_from_doc(doc_content)
    if not images:
        raise HTTPException(status_code=400, detail="no pages to convert")
    try:
        data = engine.convert(images)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"docling conversion failed: {exc}") from exc
    # Emit structured_document as a JSON STRING (the producer-wiring contract that
    # routedConvertResultParser expects). The parsing-router's HTTP component drops a
    # deeply-nested object field on the way to the artifact (the flat markdown_pages
    # list survives, the DoclingDocument tree does not), so a string is what actually
    # reaches converted_file.evidence_tree for grounded chunking + the bbox overlay.
    #
    # Strip pages[*].image from structured_document before stringifying: the rasterized
    # page images (base64 PNG, ~60 KB each) would bloat the string to ~65 KB and can
    # exceed proto/pipeline size limits → evidenceTreeBytes stays 0 for complex docs.
    # Expose them instead as a separate page_images list so routedConvertResultParser
    # can persist them through the PageImages channel without touching the tree.
    if isinstance(data, dict) and isinstance(data.get("structured_document"), (dict, list)):
        sd = data["structured_document"]
        page_images = []
        if isinstance(sd, dict) and isinstance(sd.get("pages"), dict):
            pages_copy = {}
            for page_no, page_info in sd["pages"].items():
                img = page_info.get("image") if isinstance(page_info, dict) else None
                if img and img.get("uri"):
                    page_images.append({
                        "page_no": int(page_no),
                        "uri": img["uri"],
                        "width": page_info.get("size", {}).get("width", 0),
                        "height": page_info.get("size", {}).get("height", 0),
                    })
                # Keep the page entry but without the image blob
                pages_copy[page_no] = {k: v for k, v in page_info.items() if k != "image"}
            sd = {**sd, "pages": pages_copy}
        data = {**data, "structured_document": json.dumps(sd)}
        if page_images:
            data = {**data, "page_images": page_images}
    return {"taskOutputs": [{"data": data}]}


@app.on_event("startup")
async def _warm():
    import os
    if os.environ.get("SHUBO_DOCLING_EAGER_LOAD", "true").lower() == "true":
        engine.load()  # on the event-loop thread, so /convert shares the MLX stream


if __name__ == "__main__":
    # Launchable as `python server.py` (how buckle supervises host model servers). Port comes from
    # the shared-server slot (SHUBO_DOCLING_PORT). For multi-replica prod throughput use serve_app.py.
    import os
    import uvicorn
    port = int(os.environ.get("SHUBO_DOCLING_PORT", os.environ.get("PORT", "8088")))
    uvicorn.run(app, host="0.0.0.0", port=port)
