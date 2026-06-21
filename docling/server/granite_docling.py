"""Core granite-docling-MLX conversion logic — shared by the plain server and the Ray Serve front.

No web framework here. A `GraniteDocling` instance loads the MLX model once and converts page
images (or a rendered PDF) into the canonical DoclingDocument contract:

    {markdown_pages: [...], structured_document: {...}}   # schema_name == "DoclingDocument"

`structured_document` is exactly the `export_to_dict()` tree `docdoc.FromDoclingExport` decodes
(round-trip verified), so this is the M7 producer — Metal-accelerated, on the macOS host.
"""
import io
import os
import time

from PIL import Image

DEFAULT_MODEL = os.environ.get("SHUBO_DOCLING_MODEL", "ibm-granite/granite-docling-258M-mlx")
DEFAULT_MAX_TOKENS = int(os.environ.get("SHUBO_DOCLING_MAX_TOKENS", "4096"))
DEFAULT_PDF_DPI = int(os.environ.get("SHUBO_DOCLING_PDF_DPI", "150"))
PROMPT = "Convert this page to docling."


class GraniteDocling:
    """One MLX model instance. Construct once per worker/replica; `convert` is re-entrant per call."""

    def __init__(self, model_id: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._model = self._processor = self._config = None

    # ── lifecycle ────────────────────────────────────────────────────────────────────────────
    def load(self):
        if self._model is None:
            from mlx_vlm import load
            loaded = load(self.model_id)
            if len(loaded) == 3:
                self._model, self._processor, self._config = loaded
            else:
                self._model, self._processor = loaded
                self._config = getattr(self._model, "config", None)
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ── inference ────────────────────────────────────────────────────────────────────────────
    def page_to_doctags(self, pil: Image.Image) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        self.load()
        try:
            prompt = apply_chat_template(self._processor, self._config, PROMPT, num_images=1)
        except Exception:
            prompt = PROMPT
        out = generate(self._model, self._processor, prompt, image=[pil],
                       max_tokens=self.max_tokens, temperature=0.0, verbose=False)
        return out.text if hasattr(out, "text") else str(out)

    def convert(self, images: list) -> dict:
        """Run granite-docling per page, merge into one DoclingDocument, return the contract."""
        from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
        t0 = time.time()
        doctags = [self.page_to_doctags(im) for im in images]
        merged = DoclingDocument.load_from_doctags(
            DocTagsDocument.from_doctags_and_image_pairs(doctags, images))
        markdown_pages = [
            DoclingDocument.load_from_doctags(
                DocTagsDocument.from_doctags_and_image_pairs([dt], [im])).export_to_markdown()
            for dt, im in zip(doctags, images)
        ]
        return {
            "markdown_pages": markdown_pages,
            "structured_document": merged.export_to_dict(),   # schema_name == "DoclingDocument"
            "num_pages": len(images),
            "inference_seconds": round(time.time() - t0, 2),
        }


def render_pdf(pdf_bytes: bytes, dpi: int = DEFAULT_PDF_DPI) -> list:
    """PDF -> list[PIL.Image], one RGB image per page (granite-docling is page-image → DocTags)."""
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    return [
        Image.open(io.BytesIO(page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png"))).convert("RGB")
        for page in doc
    ]


def decode_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
