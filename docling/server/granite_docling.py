"""Core Docling MLX conversion logic — shared by the plain server and the Ray Serve front.

No web framework here. A `GraniteDocling` instance loads the MLX model once and converts page
images (or a rendered PDF) into the canonical DoclingDocument contract:

    {markdown_pages: [...], structured_document: {...}}   # schema_name == "DoclingDocument"

Granite-docling emits DocTags, which are converted through docling-core. Unlimited-OCR emits
Markdown, so that path wraps one page-level text node per page into a minimal DoclingDocument tree.
Both forms keep the backend seam stable for `docdoc.FromDoclingExport`.
"""
import base64
import io
import os
import time

from PIL import Image

DEFAULT_MODEL = os.environ.get("SHUBO_DOCLING_MODEL", "sahilchachra/unlimited-ocr-mxfp8-mlx")
DEFAULT_MAX_TOKENS = int(os.environ.get("SHUBO_DOCLING_MAX_TOKENS", "4096"))
DEFAULT_PDF_DPI = int(os.environ.get("SHUBO_DOCLING_PDF_DPI", "150"))
GRANITE_PROMPT = "Convert this page to docling."
UNLIMITED_OCR_PROMPT = "<image>document parsing."


def _is_unlimited_ocr_model(model_id: str) -> bool:
    normalized = model_id.lower().replace("_", "-")
    return "unlimited-ocr" in normalized


def _default_prompt(model_id: str) -> str:
    if override := os.environ.get("SHUBO_DOCLING_PROMPT"):
        return override
    if _is_unlimited_ocr_model(model_id):
        return UNLIMITED_OCR_PROMPT
    return GRANITE_PROMPT


def _markdown_to_docling_document(
    markdown_pages: list[str],
    images: list[Image.Image],
    name: str,
) -> dict:
    """Build a minimal DoclingDocument from OCR Markdown, one citable text leaf per page."""
    texts = []
    body_children = []
    pages = {}
    for idx, (markdown, image) in enumerate(zip(markdown_pages, images)):
        page_no = idx + 1
        self_ref = f"#/texts/{idx}"
        width, height = image.size
        page_png = io.BytesIO()
        image.save(page_png, format="PNG")
        body_children.append({"$ref": self_ref})
        pages[str(page_no)] = {
            "size": {"width": width, "height": height},
            "image": {
                "uri": "data:image/png;base64,"
                + base64.b64encode(page_png.getvalue()).decode("ascii"),
            },
        }
        texts.append(
            {
                "self_ref": self_ref,
                "parent": {"$ref": "#/body"},
                "children": [],
                "content_layer": "body",
                "label": "text",
                "prov": [
                    {
                        "page_no": page_no,
                        "bbox": {
                            "l": 0,
                            "t": 0,
                            "r": width,
                            "b": height,
                            "coord_origin": "TOPLEFT",
                        },
                        "charspan": [0, len(markdown)],
                    }
                ],
                "source_parser": "docling",
                "is_original_evidence": True,
                "orig": markdown,
                "text": markdown,
            }
        )
    return {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": name,
        "origin": {"mimetype": "application/pdf" if len(images) > 1 else "image/*"},
        "body": {
            "self_ref": "#/body",
            "children": body_children,
            "content_layer": "body",
            "name": "_root_",
            "label": "unspecified",
        },
        "furniture": {
            "self_ref": "#/furniture",
            "children": [],
            "content_layer": "furniture",
            "name": "_root_",
            "label": "unspecified",
        },
        "groups": [],
        "texts": texts,
        "tables": [],
        "pictures": [],
        "key_value_items": [],
        "pages": pages,
    }


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
    @property
    def model_family(self) -> str:
        return "unlimited_ocr" if _is_unlimited_ocr_model(self.model_id) else "doctags"

    def page_to_text(self, pil: Image.Image) -> str:
        from mlx_vlm import generate
        self.load()
        prompt = _default_prompt(self.model_id)
        if self.model_family != "unlimited_ocr":
            from mlx_vlm.prompt_utils import apply_chat_template
            try:
                prompt = apply_chat_template(self._processor, self._config, prompt, num_images=1)
            except Exception:
                pass
        out = generate(self._model, self._processor, prompt, image=[pil],
                       max_tokens=self.max_tokens, temperature=0.0, verbose=False)
        return out.text if hasattr(out, "text") else str(out)

    def convert(self, images: list) -> dict:
        """Run the page parser, merge into one DoclingDocument-compatible contract."""
        t0 = time.time()
        page_outputs = [self.page_to_text(im) for im in images]
        if self.model_family == "unlimited_ocr":
            return {
                "markdown_pages": page_outputs,
                "structured_document": _markdown_to_docling_document(
                    page_outputs,
                    images,
                    self.model_id,
                ),
                "num_pages": len(images),
                "model": self.model_id,
                "model_family": self.model_family,
                "inference_seconds": round(time.time() - t0, 2),
            }

        from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
        merged = DoclingDocument.load_from_doctags(
            DocTagsDocument.from_doctags_and_image_pairs(page_outputs, images))
        markdown_pages = [
            DoclingDocument.load_from_doctags(
                DocTagsDocument.from_doctags_and_image_pairs([dt], [im])).export_to_markdown()
            for dt, im in zip(page_outputs, images)
        ]
        return {
            "markdown_pages": markdown_pages,
            "structured_document": merged.export_to_dict(),   # schema_name == "DoclingDocument"
            "num_pages": len(images),
            "model": self.model_id,
            "model_family": self.model_family,
            "inference_seconds": round(time.time() - t0, 2),
        }


def render_pdf(pdf_bytes: bytes, dpi: int = DEFAULT_PDF_DPI) -> list:
    """PDF -> list[PIL.Image], one RGB image per page."""
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    return [
        Image.open(io.BytesIO(page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png"))).convert("RGB")
        for page in doc
    ]


def decode_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
