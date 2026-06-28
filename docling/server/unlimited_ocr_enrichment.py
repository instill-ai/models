"""Structure (Docling) + Unlimited-OCR text via full-page OCR mapped onto the layout.

The shubo MLX docling host server (granite_docling.py) trades Docling's layout pipeline for a
single VLM. granite-docling emits DocTags directly; Unlimited-OCR emits flat Markdown, so the
structure (tables, headers, per-element bboxes / DocTags) was lost (models #72 regression).

Unlimited-OCR is a **full-page** document parser (DeepSeek-OCR family): on a whole page it emits
clean, accurate, already-grounded output — `<|det|>label [x,y,x,y]<|/det|>text` regions, with HTML
tables — but on small per-element crops it HALLUCINATES, and its tokens carry GPT-2 byte-BPE
artifacts (`Ġ`=space, `Ċ`=newline). So we do NOT crop per element. Instead:

  1. Docling's real DocumentConverter runs layout + table-structure (do_ocr off — RapidOCR is
     unused and broken here) → DoclingDocument structure + per-element bboxes + DocTags.
  2. Each page is OCR'd ONCE with Unlimited-OCR; its grounded regions are byte-BPE-decoded and
     mapped onto Docling's elements by bbox overlap (tables: the region's HTML is aligned to
     Docling's cell grid). Each element gets Unlimited-OCR's high-accuracy text.

`ocr_raw` is injected (a `PIL.Image -> str` raw-grounded callable) so this is testable without the
~3.7 GB MLX model; the server passes the real Unlimited-OCR page call.
"""
from __future__ import annotations

import io
import logging
import os
import re
from collections.abc import Callable
from html.parser import HTMLParser
from typing import List, Optional, Tuple, Union

from PIL import Image

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableItem,
    TextItem,
)

logger = logging.getLogger(__name__)

OcrRaw = Callable[[Image.Image], str]

# A grounded region: (label, (l, t, r, b) normalized 0..1 top-left, text).
Region = Tuple[str, Tuple[float, float, float, float], str]

# ── digital-text fast path ─────────────────────────────────────────────────────────────────────
# The MLX OCR forward pass is ~93% of conversion wall-clock (autoregressive decode, sequential on
# one Metal GPU) and on dense pages it runs to the token cap — sometimes looping. But Docling's
# layout pass already runs with do_ocr=False, which extracts the PDF's EMBEDDED (digital) text
# layer for free (CPU) and exactly. For a digital PDF that text is ground truth — better than OCR,
# which can hallucinate/loop — so re-OCR'ing those pages is pure waste. A scanned/image page yields
# ~0 embedded chars; a digital page carries hundreds–thousands. So a simple per-page character
# floor cleanly separates them: pages above the floor keep Docling's exact text and SKIP OCR; only
# pages below it (scanned, or a near-empty text layer) get the MLX OCR pass. Tables (cell text from
# the layer), figures (PICTURE leaves → downstream VLM describe) and captions are untouched, so the
# visual-leaf / TYPE_VISUAL path is preserved. Tunable + killswitch via env for safe A/B in deploy.
_DIGITAL_FASTPATH = (
    os.environ.get("SHUBO_DOCLING_DIGITAL_TEXT_FASTPATH", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
_MIN_DIGITAL_TEXT_CHARS = int(os.environ.get("SHUBO_DOCLING_MIN_DIGITAL_CHARS", "100"))

# ── formula enrichment (CodeFormula → LaTeX) ─────────────────────────────────────────────────────
# Docling's layout detects FORMULA regions VISUALLY (works on both digital and scanned pages — proven
# even with do_ocr=False), but the text it leaves on them is either the PDF's raw math glyphs (digital)
# or full-page-OCR text (scanned) — neither is usable LaTeX. The CodeFormulaV2 model (docling's
# `CodeFormulaVlmModel`) converts a rendered formula region → deterministic LaTeX, which is far better
# for math than the general VLM describe path. So we run it as an explicit pass over every FORMULA leaf
# and make the leaf's text the LaTeX, so the persisted formula chunk carries LaTeX.
#
# It is run MANUALLY (not via the converter's `do_formula_enrichment`) on purpose: enabling it inside
# the pipeline turns the formula into a TextItem, which would flip the `docling_has_structure` check to
# True on an image-only PDF and SKIP the grounded-OCR rebuild → the page body text (only Unlimited-OCR
# has it) would be lost. Running it ourselves on the original layout doc lets us keep the rebuild branch
# intact and re-inject the formulas into the grounded doc. Env-gated with a hard killswitch.
_FORMULA_ENRICHMENT = (
    os.environ.get("SHUBO_DOCLING_FORMULA_ENRICHMENT", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
# 18% bbox expansion + the model's own 120-DPI training scale are matched by docling's pipeline; we
# crop from the page image we already render (images_scale=2.0), expanding the same 18% for margin.
_FORMULA_BBOX_EXPANSION = 0.18


def _digital_text_chars_by_page(doc: DoclingDocument) -> dict:
    """Per-page count of Docling's embedded-text-layer characters (TextItem text + table cell text),
    the signal for the digital-text fast path. Scanned pages ≈ 0; digital pages ≫ the floor."""
    chars: dict = {}
    for item, _ in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        page_no = prov[0].page_no
        if isinstance(item, TextItem):
            # A FORMULA leaf's text is CodeFormula LaTeX, not embedded-layer body text — counting it
            # would let a single formula make a fully-scanned page look "digital" and wrongly skip OCR.
            if item.label == DocItemLabel.FORMULA:
                continue
            chars[page_no] = chars.get(page_no, 0) + len((item.text or "").strip())
        elif isinstance(item, TableItem):
            chars[page_no] = chars.get(page_no, 0) + sum(
                len((getattr(c, "text", "") or "").strip()) for c in item.data.table_cells
            )
    return chars


# ── byte-BPE decode ──────────────────────────────────────────────────────────────────────────
def _build_byte_decoder() -> dict:
    """GPT-2 byte-level map (unicode char -> byte), to undo Ġ/Ċ and byte-level mojibake."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _build_byte_decoder()


def decode_bpe(s: str) -> str:
    """Decode a byte-BPE token string back to UTF-8 (Ġ→space, Ċ→newline, mojibake→glyph)."""
    try:
        return bytes(_BYTE_DECODER[c] for c in s if c in _BYTE_DECODER).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return s.replace("Ġ", " ").replace("Ċ", "\n")


# ── grounded-output parser ───────────────────────────────────────────────────────────────────
_DET = re.compile(r"<\|det\|>(.*?)<\|/det\|>(.*?)(?=<\|det\|>|$)", re.DOTALL)
_LABEL_BOX = re.compile(r"([A-Za-z_]+)?\s*\[\s*([\d.\s,]+?)\s*\]")


def parse_grounded_regions(raw: str) -> List[Region]:
    """Parse Unlimited-OCR's `[<|det|>]label [x,y,x,y]<|/det|>text` output into decoded regions.

    `<|det|>` is a region SEPARATOR (the very first region often has no leading one), so we split on
    it rather than requiring a matched pair — otherwise the first element (e.g. the heading) is lost.
    Coordinates are the model's 0..999 top-left space → clamped to 0..1 (the model occasionally emits
    a runaway coordinate). Text is byte-BPE-decoded.
    """
    regions: List[Region] = []
    for chunk in raw.split("<|det|>"):
        if "<|/det|>" not in chunk:
            continue
        box_part, text_part = chunk.split("<|/det|>", 1)
        m = _LABEL_BOX.search(decode_bpe(box_part))
        if not m:
            continue
        label = (m.group(1) or "text").lower()
        try:
            nums = [float(x) for x in m.group(2).split(",") if x.strip()]
        except ValueError:
            continue
        if len(nums) < 4:
            continue
        box = tuple(min(1.0, max(0.0, n / 999.0)) for n in nums[:4])
        text = decode_bpe(text_part).strip()
        if text:
            regions.append((label, box, text))  # type: ignore[arg-type]
    return regions


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _best_region(box: Tuple[float, float, float, float], regions: List[Region]) -> Optional[Region]:
    """The OCR region this element's text belongs to. Prefer a region that contains the element's
    center (robust to the OCR and layout boxes not being pixel-identical); pick the smallest such.
    Otherwise fall back to the highest-IoU region above a floor — so an element with no real OCR
    coverage isn't handed a stray region."""
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    containing = [
        r
        for r in regions
        if r[1][0] - 1e-6 <= cx <= r[1][2] + 1e-6 and r[1][1] - 1e-6 <= cy <= r[1][3] + 1e-6
    ]
    if containing:
        return min(containing, key=lambda r: (r[1][2] - r[1][0]) * (r[1][3] - r[1][1]))
    best, best_iou = None, 0.0
    for reg in regions:
        score = _iou(box, reg[1])
        if score > best_iou:
            best, best_iou = reg, score
    return best if best_iou >= 0.1 else None


# ── HTML table → cell grid ───────────────────────────────────────────────────────────────────
class _TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self._cell: Optional[List[str]] = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            if self.rows:
                self.rows[-1].append("".join(self._cell).strip())
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _html_to_grid(html: str) -> List[List[str]]:
    p = _TableHTMLParser()
    try:
        p.feed(html)
    except Exception:  # noqa: BLE001
        return []
    return [r for r in p.rows if r]


def _fill_table_from_html(table: TableItem, html: str) -> None:
    """Align the OCR region's HTML table to Docling's detected cell grid by (row, col) index."""
    grid = _html_to_grid(html)
    if not grid:
        return
    for cell in table.data.table_cells:
        r = getattr(cell, "start_row_offset_idx", None)
        c = getattr(cell, "start_col_offset_idx", None)
        if r is None or c is None:
            continue
        if 0 <= r < len(grid) and 0 <= c < len(grid[r]):
            text = grid[r][c].strip()
            if text:
                cell.text = text


# Map a region label onto the FULL DoclingDocument label vocabulary (docling
# datamodel.document.layout_label_to_ds_type) — not the subset the official DeepSeek-OCR parser
# covers. Aliases handle the model's label strings; any label that already IS a DocItemLabel value
# (formula, list_item, code, footnote, reference, caption, page_header, …) passes straight through.
_LABEL_ALIASES = {
    "title": DocItemLabel.SECTION_HEADER,
    "sub_title": DocItemLabel.SECTION_HEADER,
    "subtitle": DocItemLabel.SECTION_HEADER,
    "heading": DocItemLabel.SECTION_HEADER,
    "header": DocItemLabel.PAGE_HEADER,
    "footer": DocItemLabel.PAGE_FOOTER,
    "list": DocItemLabel.LIST_ITEM,
    "image": DocItemLabel.PICTURE,
    "figure": DocItemLabel.PICTURE,
    "image_caption": DocItemLabel.CAPTION,
    "figure_caption": DocItemLabel.CAPTION,
    "table_caption": DocItemLabel.CAPTION,
    "equation": DocItemLabel.FORMULA,
}
# Generic body labels go through the furniture/position heuristic, not straight through.
_GENERIC_TEXT_LABELS = {"text", "paragraph", "plain_text", ""}
_VALID_LABELS = {e.value: e for e in DocItemLabel}
# Labels not emitted as a text node (built elsewhere / imageful) — a text region carrying one of
# these names still falls through to the furniture/text logic.
_NON_TEXT_LABELS = {
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.CHART,
    DocItemLabel.FORM,
    DocItemLabel.KEY_VALUE_REGION,
}
# Unlimited-OCR labels page body as plain "text" (no furniture labels, and it often misses
# footers), so page header/footer are recovered by vertical position — restoring the
# page_header/page_footer nodes granite-docling emitted (the backend furniture chunking keys off
# them).
_HEADER_BAND = 0.07
_FOOTER_BAND = 0.93


def _resolve_label(label: str, box: Tuple[float, float, float, float]) -> DocItemLabel:
    key = (label or "").lower().strip()
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    if (
        key not in _GENERIC_TEXT_LABELS
        and key in _VALID_LABELS
        and _VALID_LABELS[key] not in _NON_TEXT_LABELS
    ):
        return _VALID_LABELS[key]
    cy = (box[1] + box[3]) / 2.0
    if cy < _HEADER_BAND:
        return DocItemLabel.PAGE_HEADER
    if cy > _FOOTER_BAND:
        return DocItemLabel.PAGE_FOOTER
    return DocItemLabel.TEXT


def _build_doc_from_grounded(page_data: dict) -> DoclingDocument:
    """Build a DoclingDocument from Unlimited-OCR's grounded regions — the fallback for image-only
    PDFs, where Docling's layout (OCR off, no text cells) produces nothing. Each region maps to the
    FULL DocItemLabel vocabulary (`_resolve_label`); tables use Docling's official robust HTML parser
    (`deepseekocr_utils._parse_table_html`, colspan/rowspan); furniture is recovered by position.

    page_data: {page_no: (regions, page_width, page_height)}.
    """
    from docling.utils.deepseekocr_utils import _parse_table_html

    doc = DoclingDocument(name="document")
    for page_no in sorted(page_data):
        regions, pw, ph = page_data[page_no]
        doc.add_page(page_no=page_no, size=Size(width=pw, height=ph))
        for label, box, text in regions:
            prov = ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(
                    l=box[0] * pw, t=box[1] * ph, r=box[2] * pw, b=box[3] * ph,
                    coord_origin=CoordOrigin.TOPLEFT,
                ),
                charspan=(0, len(text)),
            )
            resolved = _resolve_label(label, box)
            if (label or "").lower().strip() == "table" or resolved == DocItemLabel.TABLE:
                try:
                    td = _parse_table_html(text)
                except Exception:  # noqa: BLE001
                    td = None
                if td is not None and td.table_cells:
                    doc.add_table(data=td, prov=prov)
                continue
            if resolved == DocItemLabel.PICTURE:
                continue  # no OCR text to attach to a bare picture
            if resolved == DocItemLabel.LIST_ITEM:
                try:
                    doc.add_list_item(text=text, prov=prov)
                    continue
                except Exception:  # noqa: BLE001 — no list group → plain text node
                    resolved = DocItemLabel.TEXT
            doc.add_text(label=resolved, text=text, orig=text, prov=prov)
    return doc


# ── conversion ───────────────────────────────────────────────────────────────────────────────
def _structure_converter() -> DocumentConverter:
    """Docling structure only: layout + table-structure → DoclingDocument + bboxes + DocTags.
    OCR is OFF (text comes from Unlimited-OCR); page images are generated for the OCR pass."""
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    opts.generate_page_images = True
    # Page images feed the OCR model; at the 72-DPI default it loops/hallucinates, so render at
    # ~150 DPI (scale 2.0) where Unlimited-OCR is stable.
    opts.images_scale = 2.0
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _norm_tl(bbox, page_w: float, page_h: float) -> Tuple[float, float, float, float]:
    """A Docling element bbox → 0..1 top-left coords (matching the OCR region space)."""
    tl = bbox.to_top_left_origin(page_h)
    return (tl.l / page_w, tl.t / page_h, tl.r / page_w, tl.b / page_h)


def _correct_furniture_labels(doc: DoclingDocument) -> None:
    """Demote a BODY-positioned line that Docling mislabeled page_header/page_footer back to TEXT.

    The digital/structured path keeps Docling's layout labels verbatim, and Docling's layout model
    sometimes tags a real content line as page_header/page_footer. That silently drops it from content
    chunks / retrieval, because the backend furniture chunking keys off those labels (and they fail
    IsEvidenceLabel). This is the inverse of `_resolve_label`'s margin rule, applied as a CORRECTION
    over the FINAL doc so it covers every path (digital/structured, OCR-mapped, and the grounded
    rebuild). Position is the strong signal — genuine furniture is ALWAYS in the top/bottom margin
    band; the content-like guard (long or multi-word) keeps a short page number that sits slightly
    inside the body from being demoted. Mutates labels in place; no-op when nothing qualifies.
    """
    for item, _ in doc.iterate_items():
        if not isinstance(item, TextItem):
            continue
        if item.label not in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER):
            continue
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        page = doc.pages.get(prov[0].page_no)
        if page is None or page.size is None:
            continue
        _, t, _, b = _norm_tl(prov[0].bbox, float(page.size.width), float(page.size.height))
        cy = (t + b) / 2.0
        txt = (item.text or "").strip()
        content_like = len(txt) > 15 or " " in txt
        if not content_like:
            continue  # short / number-like (page numbers, running heads) → genuine furniture, keep
        # A genuine footer sits in the bottom band (cy > _FOOTER_BAND); a genuine header in the top
        # band (cy < _HEADER_BAND). A content-like line OUTSIDE its band carries the label wrongly.
        if item.label == DocItemLabel.PAGE_FOOTER and cy <= _FOOTER_BAND:
            item.label = DocItemLabel.TEXT
        elif item.label == DocItemLabel.PAGE_HEADER and cy >= _HEADER_BAND:
            item.label = DocItemLabel.TEXT


# ── formula enrichment ───────────────────────────────────────────────────────────────────────────
_FORMULA_MODEL = None  # lazily-loaded singleton (load once, reuse across requests; serial on one GPU)
_FORMULA_MODEL_FAILED = False


def _get_formula_model():
    """Lazily load docling's CodeFormulaV2 enrichment model once per process and cache it. On any
    failure (missing weights, OOM, unsupported runtime) it disables itself permanently and returns
    None — formula enrichment then degrades to "leave the formula text as Docling/OCR produced it"."""
    global _FORMULA_MODEL, _FORMULA_MODEL_FAILED
    if _FORMULA_MODEL is not None or _FORMULA_MODEL_FAILED:
        return _FORMULA_MODEL
    try:
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.models.stages.code_formula.code_formula_vlm_model import (
            CodeFormulaVlmModel,
        )

        # Derive the full default CodeFormulaVlmOptions (engine + model_spec) and flip it to
        # formulas-only (no code enrichment — we only surface math LaTeX).
        opts = PdfPipelineOptions().code_formula_options.model_copy(
            update={"extract_code": False, "extract_formulas": True}
        )
        _FORMULA_MODEL = CodeFormulaVlmModel(
            enabled=True,
            enable_remote_services=False,
            artifacts_path=None,
            options=opts,
            accelerator_options=AcceleratorOptions(),
        )
        logger.info("CodeFormula formula-enrichment model loaded")
    except Exception:  # noqa: BLE001
        _FORMULA_MODEL_FAILED = True
        logger.exception(
            "CodeFormula model load failed; formula enrichment disabled for this process"
        )
    return _FORMULA_MODEL


def _crop_formula_region(
    pil_image: Image.Image, bbox, page_w: float, page_h: float
) -> Tuple[Optional[Image.Image], Optional[Tuple[float, float, float, float]]]:
    """Crop a formula's bbox out of the rendered page image (with margin), returning the PIL crop and
    its 0..1 top-left box (for re-injection into the grounded image-PDF doc)."""
    iw, ih = pil_image.size
    tl = bbox.to_top_left_origin(page_h)
    l, t, r, b = tl.l / page_w, tl.t / page_h, tl.r / page_w, tl.b / page_h
    dw, dh = (r - l) * _FORMULA_BBOX_EXPANSION / 2.0, (b - t) * _FORMULA_BBOX_EXPANSION / 2.0
    l, r = max(0.0, l - dw), min(1.0, r + dw)
    t, b = max(0.0, t - dh), min(1.0, b + dh)
    px = (int(l * iw), int(t * ih), int(r * iw), int(b * ih))
    if px[2] <= px[0] or px[3] <= px[1]:
        return None, None
    return pil_image.crop(px), (l, t, r, b)


def _enrich_formulas(doc: DoclingDocument) -> dict:
    """Replace every FORMULA leaf's text with deterministic CodeFormula LaTeX (better than OCR/VLM for
    math). Mutates `doc` in place AND returns {page_no: [(box_tl_norm, latex)]} so the image-only
    grounded-rebuild path (which throws this doc away) can re-inject the formulas it would otherwise
    drop. No-op (no model load) when disabled or when the doc has no formula leaves; never fatal."""
    harvested: dict = {}
    if not _FORMULA_ENRICHMENT:
        return harvested
    items = []
    for it, _ in doc.iterate_items():
        if isinstance(it, TextItem) and it.label == DocItemLabel.FORMULA and it.prov:
            page = doc.pages.get(it.prov[0].page_no)
            if page is not None and page.image is not None and page.size is not None:
                items.append((it, it.prov[0], page))
    if not items:
        return harvested
    model = _get_formula_model()
    if model is None:
        return harvested

    from docling.datamodel.base_models import ItemAndImageEnrichmentElement

    batch = []
    meta = []
    for it, prov, page in items:
        crop, box = _crop_formula_region(
            page.image.pil_image, prov.bbox, float(page.size.width), float(page.size.height)
        )
        if crop is None:
            continue
        batch.append(ItemAndImageEnrichmentElement(item=it, image=crop))
        meta.append((it, prov.page_no, box))
    if not batch:
        return harvested
    try:
        list(model(doc, batch))  # mutates each item.text → LaTeX (batched, single GPU pass)
    except Exception:  # noqa: BLE001
        logger.exception("CodeFormula enrichment failed; leaving formula text as-is")
        return harvested
    for it, page_no, box in meta:
        latex = (it.text or "").strip()
        if latex:
            harvested.setdefault(page_no, []).append((box, latex))
    logger.debug("formula enrichment: %d formula leaf/leaves → LaTeX", len(meta))
    return harvested


def _center_in_any(
    box: Tuple[float, float, float, float],
    boxes: List[Tuple[float, float, float, float]],
) -> bool:
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    for fb in boxes:
        if fb[0] <= cx <= fb[2] and fb[1] <= cy <= fb[3]:
            return True
    return False


def _inject_formulas(page_data: dict, formula_regions: dict) -> None:
    """Splice CodeFormula formulas into the grounded image-PDF region map: add a `formula`-labelled
    region (text = LaTeX) per page and drop any OCR text region centred inside a formula box (it is the
    garbled full-page-OCR transcription of the same math, which LaTeX supersedes). Mutates page_data."""
    for page_no, fr in formula_regions.items():
        if page_no not in page_data:
            continue
        regions, pw, ph = page_data[page_no]
        boxes = [b for b, _ in fr]
        kept = [
            reg for reg in regions if reg[0] == "table" or not _center_in_any(reg[1], boxes)
        ]
        kept.extend(("formula", b, latex) for b, latex in fr)
        page_data[page_no] = (kept, pw, ph)


def convert_to_contract(source: Union[str, bytes], ocr_raw: OcrRaw) -> dict:
    """PDF (path or bytes) → host-server contract `{markdown_pages, structured_document}`.

    Docling gives the structure (DocTags + bboxes); each page is OCR'd once with Unlimited-OCR and
    its grounded regions are mapped onto Docling's text elements (by bbox overlap) and tables (by
    aligning the region's HTML to the cell grid). `structured_document` is the DoclingDocument tree
    the artifact persists; `markdown_pages` is one Markdown string per page.
    """
    from docling.datamodel.base_models import DocumentStream

    if isinstance(source, bytes):
        src: Union[str, DocumentStream] = DocumentStream(
            name="document.pdf", stream=io.BytesIO(source)
        )
    else:
        src = source
    doc: DoclingDocument = _structure_converter().convert(src).document

    # Formula enrichment: turn every layout-detected FORMULA leaf into deterministic LaTeX. Run on the
    # original layout doc (formula regions + page images exist for BOTH digital and image-only PDFs);
    # `formula_regions` carries them forward so the image-PDF grounded rebuild can re-inject them.
    formula_regions = _enrich_formulas(doc)

    # OCR each page exactly once (Unlimited-OCR is full-page); cache the grounded regions.
    # Digital-text fast path: skip the (expensive, GPU-bound) OCR for any page whose embedded text
    # layer already covers it — Docling's extracted text is exact and free, so those pages keep it
    # and are simply absent from page_data (the mapping below is a no-op for them). Scanned / empty
    # -layer pages fall through to OCR exactly as before.
    digital_chars = _digital_text_chars_by_page(doc) if _DIGITAL_FASTPATH else {}
    page_data: dict = {}
    skipped_digital = 0
    for page_no in sorted(doc.pages):
        page = doc.pages[page_no]
        img = page.image.pil_image if (page.image is not None) else None
        if img is None or page.size is None:
            continue
        if _DIGITAL_FASTPATH and digital_chars.get(page_no, 0) >= _MIN_DIGITAL_TEXT_CHARS:
            skipped_digital += 1
            continue  # usable digital text layer → keep Docling's exact text, skip MLX OCR
        page_data[page_no] = (
            parse_grounded_regions(ocr_raw(img)),
            float(page.size.width),
            float(page.size.height),
        )
    logger.debug(
        "digital-text fast path: %d/%d page(s) used the embedded layer (OCR skipped); %d OCR'd",
        skipped_digital, len(doc.pages), len(page_data),
    )

    # Does Docling's layout carry usable structure? Digital PDFs: yes. Image-only PDFs: no text
    # elements and empty tables (layout needs embedded text / a working OCR) → build from the
    # grounded OCR instead.
    # "Usable structure" = real BODY text or a populated table. FORMULA leaves are excluded: layout
    # emits a (visually-detected) FORMULA TextItem even on a fully-scanned page where it can extract no
    # body text, and a lone formula must NOT mask an image-only PDF — otherwise the grounded rebuild is
    # skipped and the page body (which only Unlimited-OCR has) is lost. Empty text items are excluded
    # for the same reason.
    text_items = [
        it
        for it, _ in doc.iterate_items()
        if isinstance(it, TextItem)
        and it.label != DocItemLabel.FORMULA
        and (it.text or "").strip()
    ]
    table_items = [it for it, _ in doc.iterate_items() if isinstance(it, TableItem)]
    docling_has_structure = bool(text_items) or any(t.data.table_cells for t in table_items)

    if not docling_has_structure and page_data:
        # Image-only PDF: rebuild from grounded OCR, first splicing in the CodeFormula LaTeX so the
        # formulas Docling detected visually survive (the rebuilt doc is built from OCR regions only).
        _inject_formulas(page_data, formula_regions)
        doc = _build_doc_from_grounded(page_data)
    else:
        for item, _ in doc.iterate_items():
            prov = getattr(item, "prov", None)
            if not prov or prov[0].page_no not in page_data:
                continue
            regions, pw, ph = page_data[prov[0].page_no]
            box = _norm_tl(prov[0].bbox, pw, ph)
            if isinstance(item, TableItem):
                treg = _best_region(box, [r for r in regions if r[0] == "table"]) or _best_region(
                    box, regions
                )
                if treg:
                    _fill_table_from_html(item, treg[2])
            elif isinstance(item, TextItem):
                # Keep CodeFormula's LaTeX on formula leaves — never overwrite it with the (garbled)
                # full-page-OCR text that overlaps the formula region.
                if item.label == DocItemLabel.FORMULA:
                    continue
                reg = _best_region(box, [r for r in regions if r[0] != "table"])
                if reg and reg[2]:
                    item.text = reg[2]
                    item.orig = reg[2]

    # Correct any body line Docling's layout mislabeled as page furniture — runs on the FINAL doc so
    # it covers both the grounded rebuild and the digital/structured path (which keeps layout labels
    # verbatim). Without this, a mislabeled content line is dropped from chunks by furniture chunking.
    _correct_furniture_labels(doc)

    page_nos = sorted(doc.pages)
    markdown_pages = (
        [doc.export_to_markdown(page_no=p) for p in page_nos]
        if page_nos
        else [doc.export_to_markdown()]
    )
    return {"markdown_pages": markdown_pages, "structured_document": doc.export_to_dict()}
