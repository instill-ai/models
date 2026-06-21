"""granite-docling MLX — Ray Serve front (multi-replica, page-parallel).

Two deployments so a single document's pages run *concurrently across replicas* (the throughput
lever — per-page latency has a ~2.3s floor, so parallelism is how a multi-page doc gets fast):

  • DoclingModel  — the MLX model. Autoscaling replicas; each holds one ~631MB model and converts
                    ONE page image -> DocTags. This is the only Metal-bound work.
  • DoclingIngress— HTTP ingress (FastAPI). Renders a PDF to page images, fans the pages out to the
                    model replicas concurrently, then assembles the merged DoclingDocument + the
                    per-page markdown and returns the canonical contract.

Run on the host (native arm64 — MLX needs Metal, so NOT in a container):
    serve run serve_app:app          # or programmatic serve.run(app)
Scale: SHUBO_DOCLING_MIN_REPLICAS / SHUBO_DOCLING_MAX_REPLICAS (default 1..6).
"""
import asyncio
import base64
import io
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve

from granite_docling import GraniteDocling, render_pdf, DEFAULT_MODEL, DEFAULT_PDF_DPI

MIN_REPLICAS = int(os.environ.get("SHUBO_DOCLING_MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.environ.get("SHUBO_DOCLING_MAX_REPLICAS", "6"))

api = FastAPI(title="granite-docling-mlx (ray serve)", version="0.1.0")


class ConvertRequest(BaseModel):
    image_b64: str | None = None
    pdf_b64: str | None = None


@serve.deployment(
    autoscaling_config={"min_replicas": MIN_REPLICAS, "max_replicas": MAX_REPLICAS,
                        "target_ongoing_requests": 1},
    # MLX uses the GPU via Metal (scheduled by replica count, not a CUDA num_gpus request).
    ray_actor_options={"num_cpus": 2},
    # One page at a time per replica: MLX's Metal stream is thread-local and the model is loaded
    # on the replica's main/event-loop thread, so generation must run there (no thread offload).
    # Page-level PARALLELISM therefore comes from REPLICAS, which the ingress fans out across.
    max_ongoing_requests=1,
)
class DoclingModel:
    def __init__(self):
        self.engine = GraniteDocling().load()

    async def doctags(self, png_bytes: bytes) -> str:
        from granite_docling import decode_image
        # Run on the replica's main thread (same thread that loaded the MLX model) — MLX's GPU
        # stream is thread-local, so an asyncio.to_thread offload breaks it. Blocks this replica's
        # loop for the generation; that's fine, the router sends concurrent pages to other replicas.
        return self.engine.page_to_doctags(decode_image(png_bytes))


@serve.deployment(ray_actor_options={"num_cpus": 1})
@serve.ingress(api)
class DoclingIngress:
    def __init__(self, model):
        self.model = model  # DeploymentHandle -> DoclingModel

    @api.get("/health")
    async def health(self):
        return {"status": "ok", "model": DEFAULT_MODEL,
                "replicas": f"{MIN_REPLICAS}..{MAX_REPLICAS}"}

    @api.post("/convert")
    async def convert(self, req: ConvertRequest):
        if req.pdf_b64:
            images = render_pdf(base64.b64decode(req.pdf_b64), DEFAULT_PDF_DPI)
        elif req.image_b64:
            from granite_docling import decode_image
            images = [decode_image(base64.b64decode(req.image_b64))]
        else:
            raise HTTPException(status_code=400, detail="provide image_b64 or pdf_b64")
        if not images:
            raise HTTPException(status_code=400, detail="no pages to convert")

        # fan pages out to the model replicas concurrently (page-level parallelism)
        def png(im):
            buf = io.BytesIO(); im.save(buf, format="PNG"); return buf.getvalue()
        try:
            doctags = await asyncio.gather(*[self.model.doctags.remote(png(im)) for im in images])
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"docling conversion failed: {exc}") from exc

        # assemble in the ingress (cheap, no Metal)
        from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
        merged = DoclingDocument.load_from_doctags(
            DocTagsDocument.from_doctags_and_image_pairs(list(doctags), images))
        markdown_pages = [
            DoclingDocument.load_from_doctags(
                DocTagsDocument.from_doctags_and_image_pairs([dt], [im])).export_to_markdown()
            for dt, im in zip(doctags, images)
        ]
        return {
            "markdown_pages": markdown_pages,
            "structured_document": merged.export_to_dict(),   # schema_name == "DoclingDocument"
            "num_pages": len(images),
        }


# `serve run serve_app:app`
app = DoclingIngress.bind(DoclingModel.bind())
