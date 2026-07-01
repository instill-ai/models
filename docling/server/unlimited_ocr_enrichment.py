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
import string
import time
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

# Visual-region capture for SCANNED / image-dominated pages. On a scan the OCR model transcribes a
# signature/figure as text and the layout model finds no picture cluster, so non-text graphics
# (signatures, stamps, logos, handwriting) are invisible to the visual-description pipeline. We
# recover them by RESIDUAL-INK detection: on a page whose NATIVE (pre-OCR) text layer is sparse, blank
# out every detected text/table/picture box on the rendered page image, then box the ink that remains
# — that residual ink IS the non-text graphic, and its bbox is tight (just the signature, not the
# whole page). A page with a richer native text layer is treated as digital and is left untouched.
_PAGE_PICTURE_ENABLED = (
    os.environ.get("SHUBO_DOCLING_PAGE_PICTURE", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
_PAGE_PICTURE_MAX_NATIVE_CHARS = int(
    os.environ.get("SHUBO_DOCLING_PAGE_PICTURE_MAX_NATIVE_CHARS", "500")
)
# Residual-ink detector tunables (fractions of page unless noted).
_VIS_INK_THRESHOLD = int(os.environ.get("SHUBO_DOCLING_VIS_INK_THRESHOLD", "160"))  # gray < ⇒ ink
_VIS_MIN_AREA_FRAC = float(os.environ.get("SHUBO_DOCLING_VIS_MIN_AREA_FRAC", "0.0008"))
_VIS_MAX_AREA_FRAC = float(os.environ.get("SHUBO_DOCLING_VIS_MAX_AREA_FRAC", "0.40"))
_VIS_ERASE_PAD_FRAC = float(os.environ.get("SHUBO_DOCLING_VIS_ERASE_PAD_FRAC", "0.012"))
_VIS_BAND_FRAC = float(os.environ.get("SHUBO_DOCLING_VIS_BAND_FRAC", "0.08"))  # header/footer skip
_VIS_MIN_DIM_FRAC = float(os.environ.get("SHUBO_DOCLING_VIS_MIN_DIM_FRAC", "0.012"))  # drop thin rules

# Scanned-page geometry. A SCANNED page is rendered from a (near) full-page raster image, even if it
# also carries an embedded text layer. On such a page Docling's PDF backend extracts a REDUCED / lower
# text-cell bbox (~x-height) that clips the visual ink, while the Unlimited-OCR region (detected on the
# rendered image) matches the page image exactly. So on scanned pages we (a) never digital-fast-path
# them and (b) adopt the matched OCR region's bbox as each element's geometry. Killable.
_SCAN_VISUAL_BBOX = (
    os.environ.get("SHUBO_DOCLING_SCAN_VISUAL_BBOX", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
_IMAGE_BACKED_MIN_COVER = float(os.environ.get("SHUBO_DOCLING_IMAGE_BACKED_MIN_COVER", "0.8"))
# Table cells carry their OWN reduced bbox and have no per-cell OCR region to snap to, so on a scanned
# page they're expanded vertically by the page's MEASURED body-text reduction (the same clip). Killable.
_TABLE_CELL_EXPAND = (
    os.environ.get("SHUBO_DOCLING_TABLE_CELL_EXPAND", "1").strip().lower()
    not in ("0", "false", "no", "off")
)

# ── garbled embedded-span reconcile (hybrid digital + handwriting pages) ─────────────────────────
# A HYBRID page is a clean, digitally-generated page (e.g. a DocuSign-stamped agreement) that ALSO
# carries a line some UPSTREAM tool already OCR'd into the PDF text layer as garbage — a handwritten
# "Dated: 12 December 2022" stored as the embedded span "Dated:  I 2- D e lo cf.,r  Zo  ZZ". The page
# sails over the digital-text char floor above, so the blind fast path TRUSTS that garbled span and
# never OCR's it — the date is lost. The MLX whole-page OCR reads that exact cursive correctly, so the
# fix is to reconcile: when a digital page carries garbled span(s), run the whole-page OCR for THAT
# page only and replace each garbled element's text with the matching OCR region, while keeping every
# clean digital element (and long alphanumeric IDs) verbatim. Clean digital pages never trip this, so
# they keep the zero-cost fast path. Killswitch via env for safe A/B in deploy.
_RECONCILE_GARBLED = (
    os.environ.get("SHUBO_DOCLING_RECONCILE_GARBLED", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
# Detector thresholds (token-level so a long no-space alphanumeric ID is never mistaken for garbage).
_GARBLE_MIN_TOKENS = 4  # too few whitespace tokens to judge reliably → never flagged (clean labels)
_GARBLE_WORD_FLOOR = 0.5  # necessary gate: a span with >=50% real-word/ID tokens is NOT garbled
_GARBLE_SINGLE_CHAR_MIN = 0.2  # >=20% bare single-char tokens ("I","D","e") → cursive split per glyph
_GARBLE_SHORT_MIN = 0.5  # >=50% <=2-char tokens → heavy fragmentation
_WORD_RUN = re.compile(r"[^\W\d_]{3,}")  # a run of >=3 letters (a real word, unicode-aware)
_LONG_ALNUM = re.compile(r"[A-Za-z0-9]{5,}")  # a long no-space alphanumeric token (e.g. a DocuSign ID)
_PUNCT_RUN = re.compile(r"[^\w\s]{2,}")  # a run of >=2 consecutive punctuation marks ("cf.,r")


def _is_garbled_text(text: str) -> bool:
    """Score an embedded text span as garbled/low-quality OCR (vs clean digital prose or an ID).

    Robust token-level heuristic tuned against real DocuSign-page spans:
      garbled  "Dated:  I 2- D e lo cf.,r  Zo  ZZ"  (handwriting an upstream tool glyph-split)
      clean    "Name of Director: Ping-Lin Chang"   (digital prose)
      clean    "DocuSign Envelope ID: A7E30573-..." (a long no-space alphanumeric ID — never flagged)

    A span is garbled only when it has enough tokens, FEW real words/IDs (`word_like_ratio` below the
    floor — the necessary gate that protects clean text and IDs), AND a strong fragmentation signal:
    many single-character tokens, mostly <=2-char tokens, or a broken punctuation run. Operating on
    whitespace tokens (never raw characters) is what keeps a long ID a single, clearly-good token.
    """
    s = (text or "").strip()
    if not s:
        return False
    tokens = s.split()
    n = len(tokens)
    if n < _GARBLE_MIN_TOKENS:
        return False
    word_like = single_char = short = 0
    broken_punct = False
    for tok in tokens:
        core = tok.strip(string.punctuation)
        if _WORD_RUN.search(tok) or _LONG_ALNUM.search(tok):
            word_like += 1
        if len(core) == 1 and core.isalnum():
            single_char += 1
        if len(core) <= 2:
            short += 1
        if _PUNCT_RUN.search(tok):
            broken_punct = True
    if word_like / n >= _GARBLE_WORD_FLOOR:
        return False
    return (
        single_char / n >= _GARBLE_SINGLE_CHAR_MIN
        or short / n >= _GARBLE_SHORT_MIN
        or broken_punct
    )


def _pages_with_garbled_spans(doc: DoclingDocument) -> set:
    """Page numbers carrying at least one garbled embedded TextItem span (the reconcile trigger).
    FORMULA leaves are excluded — their text is CodeFormula LaTeX, not an embedded-layer span."""
    pages: set = set()
    for item, _ in doc.iterate_items():
        if not isinstance(item, TextItem) or item.label == DocItemLabel.FORMULA:
            continue
        prov = getattr(item, "prov", None)
        if prov and _is_garbled_text(item.text or ""):
            pages.add(prov[0].page_no)
    return pages


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
# CodeFormula is a VLM (one LaTeX generation per formula crop) on the SHARED Metal GPU, so a
# formula-DENSE paper (hundreds of equations) can dominate a conversion. Bound it: enrich at most
# _FORMULA_MAX_LEAVES formulas, in sub-batches of _FORMULA_BATCH, stopping once _FORMULA_TIME_BUDGET_S
# is exceeded. Excess formulas keep their embedded-layer text (digital) or OCR text (scanned). Tunable.
_FORMULA_MAX_LEAVES = int(os.environ.get("SHUBO_DOCLING_FORMULA_MAX_LEAVES", "60"))
_FORMULA_BATCH = max(1, int(os.environ.get("SHUBO_DOCLING_FORMULA_BATCH", "8")))
_FORMULA_TIME_BUDGET_S = float(os.environ.get("SHUBO_DOCLING_FORMULA_TIME_BUDGET_S", "90"))
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

# Recovery-pass guards (see convert_to_contract). The OCR model occasionally loops, emitting many
# degenerate (zero/near-zero area) or placeholder regions; those must NOT become recovered nodes.
_MIN_REGION_DIM = 0.004  # a recoverable region must span >0.4% of the page in BOTH dimensions
_PLACEHOLDER_RE = re.compile(r"^\[(?:non[-\s]?text|image|figure|photo|graphic|table|chart)\]$", re.I)


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
                # A bare picture carries no OCR text, but it MUST still become a PictureItem so the
                # downstream visual-description pipeline can describe it. Dropping it here is why
                # detected figures/signatures never reached doc.pictures on image-only PDFs.
                doc.add_picture(prov=prov)
                continue
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


# A single-formula crop is one equation, but CodeFormula's greedy decode (temperature 0) sometimes
# LOOPS — re-emitting the equation as a second `align` row with the SAME left-hand side (often via a
# junk \intertext), and the repeat is frequently corrupted (e.g. LC/8 → 1/8). Drop such repeated rows.
_INTERTEXT_RE = re.compile(r"\\intertext\s*\{[^}]*\}")


def _dedup_formula_latex(latex: str) -> str:
    r"""Collapse a CodeFormula repetition artifact: keep only the FIRST row for each distinct non-empty
    left-hand side (text before the first `&`). Conservative — genuine multi-row systems have distinct
    LHSs (or empty-LHS `& = …` continuations), which are preserved; only a row whose non-trivial LHS
    already appeared is dropped. Junk `\intertext{…}` fragments (the loop's seam) are stripped."""
    s = (latex or "").strip()
    if "\\\\" not in s:  # no row separator → single row, nothing to dedup
        return s
    rows = s.split("\\\\")
    kept: list = []
    seen_lhs: list = []
    dropped = 0
    for row in rows:
        r = _INTERTEXT_RE.sub("", row).strip()
        if not r:
            continue
        lhs = r.split("&", 1)[0].strip()
        if len(lhs) > 3 and lhs in seen_lhs:
            dropped += 1  # a re-emission of an already-seen equation — the looped duplicate
            continue
        if lhs:
            seen_lhs.append(lhs)
        kept.append(r)
    if dropped == 0 or not kept:
        return s  # no duplicate row removed → leave the original untouched (incl. spacing)
    return " \\\\ ".join(kept)


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
    total = len(items)
    if total > _FORMULA_MAX_LEAVES:  # formula-dense doc: cap so it can't dominate the conversion
        logger.warning(
            "formula-dense doc: %d formula leaves, enriching the first %d (rest keep embedded/OCR text)",
            total, _FORMULA_MAX_LEAVES,
        )
        items = items[:_FORMULA_MAX_LEAVES]
    model = _get_formula_model()
    if model is None:
        return harvested

    from docling.datamodel.base_models import ItemAndImageEnrichmentElement

    elements = []
    meta = []
    for it, prov, page in items:
        crop, box = _crop_formula_region(
            page.image.pil_image, prov.bbox, float(page.size.width), float(page.size.height)
        )
        if crop is None:
            continue
        elements.append(ItemAndImageEnrichmentElement(item=it, image=crop))
        meta.append((it, prov.page_no, box))
    if not elements:
        return harvested
    # Sub-batch with a wall-clock budget: CodeFormula is one VLM generation per crop on the shared
    # GPU, so a budget bounds a slow/contended run (the model() call mutates each item.text → LaTeX).
    start = time.monotonic()
    done = 0
    for i in range(0, len(elements), _FORMULA_BATCH):
        if done and time.monotonic() - start > _FORMULA_TIME_BUDGET_S:
            logger.warning(
                "formula enrichment hit the %.0fs budget; %d/%d enriched (rest keep their text)",
                _FORMULA_TIME_BUDGET_S, done, len(elements),
            )
            break
        try:
            list(model(doc, elements[i:i + _FORMULA_BATCH]))  # mutates item.text → LaTeX
        except Exception:  # noqa: BLE001
            logger.exception("CodeFormula enrichment failed; leaving remaining formula text as-is")
            break
        done += len(elements[i:i + _FORMULA_BATCH])
    for it, page_no, box in meta[:done]:
        latex = _dedup_formula_latex((it.text or "").strip())
        it.text = latex  # persist the de-looped LaTeX on the node, not just the harvested copy
        if latex:
            harvested.setdefault(page_no, []).append((box, latex))
    logger.debug("formula enrichment: %d/%d formula leaf/leaves → LaTeX", done, total)
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


def _detect_visual_regions(gray, erase_rects, page_w_px, page_h_px):
    """Residual-ink detector. `gray` is a HxW uint8 page render; `erase_rects` are pixel boxes of
    detected text/table/picture elements to blank out (their ink is already explained). Returns tight
    pixel boxes `(x0, y0, x1, y1)` of the ink that remains — the non-text graphics on the page.

    Pure + dependency-light (numpy + cv2, both already Docling deps) so it unit-tests on a synthetic
    array without the OCR weights.
    """
    import cv2  # local import: heavy native dep, only needed on scanned pages
    import numpy as np

    H, W = gray.shape
    ink = (gray < _VIS_INK_THRESHOLD).astype(np.uint8) * 255
    band = int(_VIS_BAND_FRAC * H)
    if band:
        ink[:band, :] = 0          # header band (page furniture: envelope IDs, titles)
        ink[H - band:, :] = 0      # footer band
    pad = int(_VIS_ERASE_PAD_FRAC * W)
    for (x0, y0, x1, y1) in erase_rects:
        ink[max(0, int(y0) - pad):min(H, int(y1) + pad),
            max(0, int(x0) - pad):min(W, int(x1) + pad)] = 0
    # Close gaps so a multi-stroke graphic (cursive signature) becomes one component.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    page_area = float(W * H)
    min_dim = _VIS_MIN_DIM_FRAC
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < _VIS_MIN_AREA_FRAC * page_area:
            continue                                  # specks / scan noise
        if w * h > _VIS_MAX_AREA_FRAC * page_area:
            continue                                  # page-scale blob, not a discrete graphic
        if w < min_dim * W and h < min_dim * H:
            continue                                  # tiny
        if h < min_dim * H and w > 0.30 * W:
            continue                                  # long thin rule / underline, not a figure
        out.append((x, y, x + w, y + h))
    return out


def _element_erase_rects(doc: DoclingDocument, page_no: int, sx: float, sy: float, page_h: float):
    """Pixel boxes of every detected text/table/picture element on `page_no` (already-explained ink),
    converting each bbox to TOP-LEFT pixel space regardless of its stored coord origin."""
    rects = []
    for item, _ in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov or prov[0].page_no != page_no:
            continue
        b = prov[0].bbox
        if getattr(b.coord_origin, "value", b.coord_origin) == CoordOrigin.BOTTOMLEFT.value:
            top, bot = page_h - max(b.t, b.b), page_h - min(b.t, b.b)
        else:
            top, bot = min(b.t, b.b), max(b.t, b.b)
        rects.append((b.l * sx, top * sy, b.r * sx, bot * sy))
    return rects


def _add_scanned_page_pictures(
    doc: DoclingDocument, native_chars_by_page: dict, page_images: dict
) -> int:
    """For every image-dominated (scanned) page, emit a tight PictureItem per non-text graphic found by
    residual-ink detection, so signatures/stamps/figures flow into the visual-description pipeline.

    A page is image-dominated when its NATIVE (pre-OCR) text layer is sparse
    (≤ `_PAGE_PICTURE_MAX_NATIVE_CHARS`) — on such a page the meaning lives in the pixels, and the
    residual ink (everything not already a detected text/table/picture box) is the graphic. Boxes are
    tight (just the signature, not the whole page). `page_images` maps page_no → PIL image (captured
    before any grounded rebuild, which drops page images). Returns the number of pictures added.
    """
    if not _PAGE_PICTURE_ENABLED:
        return 0
    import numpy as np

    added = 0
    for page_no, page in doc.pages.items():
        if page.size is None:
            continue
        if native_chars_by_page.get(page_no, 0) > _PAGE_PICTURE_MAX_NATIVE_CHARS:
            continue  # rich native text → digital page; its content is the text, not the image
        pil = page_images.get(page_no)
        if pil is None:
            continue
        gray = np.asarray(pil.convert("L"))
        H, W = gray.shape
        pw, ph = float(page.size.width), float(page.size.height)
        sx, sy = W / pw, H / ph
        erase = _element_erase_rects(doc, page_no, sx, sy, ph)
        for (x0, y0, x1, y1) in _detect_visual_regions(gray, erase, W, H):
            doc.add_picture(
                prov=ProvenanceItem(
                    page_no=page_no,
                    bbox=BoundingBox(
                        l=x0 / sx, t=y0 / sy, r=x1 / sx, b=y1 / sy,
                        coord_origin=CoordOrigin.TOPLEFT,
                    ),
                    charspan=(0, 0),
                )
            )
            added += 1
    return added


def _image_backed_pages(source: Union[str, bytes]) -> set:
    """Page numbers (1-based) rendered from a (near) full-page raster image — a SCANNED page, even when
    it also carries a digital text layer. Detected via a single image XObject covering
    ≥ `_IMAGE_BACKED_MIN_COVER` of the page. Best-effort: empty on any parse failure (degrades to the
    pre-fix behaviour). Gated by `_SCAN_VISUAL_BBOX`."""
    if not _SCAN_VISUAL_BBOX:
        return set()
    try:
        import fitz  # PyMuPDF (a Docling dependency)

        doc = (
            fitz.open(stream=bytes(source), filetype="pdf")
            if isinstance(source, (bytes, bytearray))
            else fitz.open(source)
        )
    except Exception:  # noqa: BLE001
        return set()
    out = set()
    for i in range(doc.page_count):
        pg = doc[i]
        page_area = float(pg.rect.width * pg.rect.height) or 1.0
        for im in pg.get_images(full=True):
            try:
                rects = pg.get_image_rects(im[0])
            except Exception:  # noqa: BLE001
                continue
            if any((r.width * r.height) >= _IMAGE_BACKED_MIN_COVER * page_area for r in rects):
                out.add(i + 1)
                break
    return out


# Docling's PDF reading-order sometimes threads a multi-column row (a 3-up author block, a 2-column
# header) into ONE text node — the node carries text from several columns under a single left-column
# bbox, so the other columns get no box (the dashboard "missing author" regression). PyMuPDF's own
# line geometry segments those columns correctly, so we re-split such a node per column. Killswitch +
# the min x-gap (pt) between columns that counts as a reading-order jump.
_RESEGMENT_MULTICOL = (
    os.environ.get("SHUBO_DOCLING_RESEGMENT_MULTICOL", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
_RESEGMENT_COL_GAP = float(os.environ.get("SHUBO_DOCLING_RESEGMENT_COL_GAP", "40"))


def _page_text_lines(mupage) -> list:
    """PyMuPDF text lines for a page as (norm_text, x0, y0, x1, y1) in TOP-LEFT page coords."""
    out = []
    for blk in mupage.get_text("dict").get("blocks", []):
        if blk.get("type") != 0:  # 0 == text block
            continue
        for ln in blk.get("lines", []):
            txt = " ".join("".join(s.get("text", "") for s in ln.get("spans", [])).split())
            if txt:
                x0, y0, x1, y1 = ln["bbox"]
                out.append((txt, x0, y0, x1, y1))
    return out


def _resegment_overflow_nodes(doc: DoclingDocument, source: Union[str, bytes]) -> int:
    """Re-split a text node that Docling threaded across COLUMNS back into one node per column, using
    PyMuPDF line geometry as ground truth. ONLY nodes whose own constituent lines span >1 column are
    touched (rare + clearly broken), so single-column prose — which Docling reads correctly — is never
    altered. A scanned page has no text layer, so it yields no lines and is a natural no-op. Best-effort
    and fully gated; returns the number of extra nodes created."""
    if not _RESEGMENT_MULTICOL:
        return 0
    try:
        import fitz

        from docling_core.types.doc import BoundingBox, CoordOrigin, ProvenanceItem

        pdf = (
            fitz.open(stream=bytes(source), filetype="pdf")
            if isinstance(source, (bytes, bytearray))
            else fitz.open(source)
        )
    except Exception:  # noqa: BLE001
        return 0

    page_lines: dict = {}
    created = 0
    for it in list(doc.texts):  # snapshot — we append while iterating
        if not it.prov or not it.text:
            continue
        pno = it.prov[0].page_no
        if pno < 1 or pno > pdf.page_count:
            continue
        node_norm = " ".join(it.text.split())
        if len(node_norm) < 12:
            continue
        if pno not in page_lines:
            page_lines[pno] = _page_text_lines(pdf[pno - 1])
        lines = page_lines[pno]
        page_h = float(pdf[pno - 1].rect.height)

        # Greedily reconstruct the node text from page lines, consuming from the front. When several
        # lines match the next prefix (repeats like "MIT CSAIL" recur in every column), pick the one
        # spatially CLOSEST to the previously-consumed line — that keeps each column's lines together.
        remaining = node_norm
        chosen: list = []
        # seed the spatial anchor at the node's own (top-left) position so the first pick is local
        ob = it.prov[0].bbox
        prev_xy = (ob.l, page_h - ob.t)
        ok = True
        while remaining:
            cands = [L for L in lines if remaining.startswith(L[0])]
            if not cands:
                ok = False
                break
            px, py = prev_xy
            pick = min(cands, key=lambda L: abs(L[1] - px) + abs(L[2] - py))
            chosen.append(pick)
            prev_xy = (pick[1], pick[2])
            remaining = remaining[len(pick[0]):].lstrip()
        if not ok or len(chosen) < 2:
            continue
        xs = sorted(L[1] for L in chosen)
        if xs[-1] - xs[0] <= _RESEGMENT_COL_GAP:
            continue  # single column — Docling read it fine

        # Cluster the chosen lines into columns by x0 gaps.
        cols: list = []
        for L in sorted(chosen, key=lambda L: (L[1], L[2])):
            if cols and L[1] - cols[-1]["x0"] <= _RESEGMENT_COL_GAP:
                cols[-1]["lines"].append(L)
            else:
                cols.append({"x0": L[1], "lines": [L]})
        if len(cols) < 2:
            continue

        segs = []
        for c in cols:
            cl = sorted(c["lines"], key=lambda L: L[2])  # top→bottom within the column
            txt = " ".join(L[0] for L in cl)
            x0 = min(L[1] for L in cl)
            y0 = min(L[2] for L in cl)
            x1 = max(L[3] for L in cl)
            y1 = max(L[4] for L in cl)
            bb = BoundingBox(l=x0, r=x1, t=page_h - y0, b=page_h - y1, coord_origin=CoordOrigin.BOTTOMLEFT)
            segs.append((txt, bb))

        # Mutate the original node into the first column; append the rest as sibling text nodes.
        ft, fb = segs[0]
        it.text = ft
        it.orig = ft
        it.prov[0].bbox = fb
        it.prov[0].charspan = (0, len(ft))
        for txt, bb in segs[1:]:
            doc.add_text(
                label=it.label,
                text=txt,
                orig=txt,
                prov=ProvenanceItem(page_no=pno, bbox=bb, charspan=(0, len(txt))),
            )
            created += 1
    return created


def _page_text_blocks(mupage) -> list:
    """PyMuPDF text blocks as (norm_text, x0, y0, x1, y1) in TOP-LEFT page coords. A block is PyMuPDF's
    paragraph-level grouping — it keeps an author's name+affiliation+email together and keeps a heading
    separate from the body that follows."""
    out = []
    for b in mupage.get_text("blocks"):
        if len(b) < 7 or b[6] != 0:  # b[6] == 0 → text block
            continue
        t = " ".join(b[4].split())
        if t:
            out.append((t, b[0], b[1], b[2], b[3]))
    return out


def _coalesce_text_blocks(doc: DoclingDocument, source: Union[str, bytes]) -> int:
    """Merge plain-TEXT nodes that one PyMuPDF block groups together but Docling OVER-SPLIT (e.g. an
    author name kept separate from its affiliation/email). Each such block collapses to ONE node = the
    block's text + bbox. Strictly scoped: only DocItemLabel.TEXT nodes whose text is fully inside the
    block AND whose center sits in it, and only when ≥2 share a block — single-paragraph blocks (1
    member) and all non-text structure (headers, lists, tables, formulas, captions) are untouched. Run
    AFTER _resegment_overflow_nodes so a cross-column merge is first split into clean per-column pieces,
    which this then re-joins with their same-column name fragment. Returns the number of nodes removed."""
    if not _RESEGMENT_MULTICOL:
        return 0
    try:
        import fitz

        from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel

        pdf = (
            fitz.open(stream=bytes(source), filetype="pdf")
            if isinstance(source, (bytes, bytearray))
            else fitz.open(source)
        )
    except Exception:  # noqa: BLE001
        return 0

    by_page: dict = {}
    for it in doc.texts:
        if it.label == DocItemLabel.TEXT and it.prov and it.text:
            by_page.setdefault(it.prov[0].page_no, []).append(it)

    to_delete: list = []
    deleted: set = set()
    merged = 0
    for pno, nodes in by_page.items():
        if pno < 1 or pno > pdf.page_count:
            continue
        page_h = float(pdf[pno - 1].rect.height)
        for (bt, bx0, by0, bx1, by1) in _page_text_blocks(pdf[pno - 1]):
            members = []
            for it in nodes:
                if id(it) in deleted:
                    continue
                nt = " ".join((it.text or "").split())
                if not nt or nt not in bt:
                    continue
                b = it.prov[0].bbox
                cx = (b.l + b.r) / 2.0
                cy = page_h - (b.t + b.b) / 2.0  # Docling bbox is BOTTOMLEFT → top-left for comparison
                if bx0 - 1 <= cx <= bx1 + 1 and by0 - 1 <= cy <= by1 + 1:
                    members.append(it)
            if len(members) < 2:
                continue
            # Collapse to one node carrying the block's full, correctly-ordered text + bbox.
            keep = members[0]
            keep.text = bt
            keep.orig = bt
            keep.prov[0].bbox = BoundingBox(
                l=bx0, r=bx1, t=page_h - by0, b=page_h - by1, coord_origin=CoordOrigin.BOTTOMLEFT
            )
            keep.prov[0].charspan = (0, len(bt))
            for extra in members[1:]:
                to_delete.append(extra)
                deleted.add(id(extra))
            merged += len(members) - 1
    if to_delete:
        try:
            doc.delete_items(node_items=to_delete)
        except Exception:  # noqa: BLE001
            pass
    return merged


# Docling scatters a multi-column author/byline row — the band between the paper title and the first
# heading that follows (ABSTRACT/INTRODUCTION) — across THREE places: body.children (the left column),
# a key_value_area group (Docling reads a name/affiliation/email column as key-value), and, once
# _resegment_overflow_nodes splits a threaded column, the TAIL of the document (add_text appends there).
# The dashboard then shows a co-author's affiliation as the last node in the paper. Collect every text
# node in that band on page 1 and re-thread them contiguously in body.children, in spatial reading order
# (row top→bottom, then column left→right), right after the title. No-op unless a clear title→heading
# band with ≥2 horizontally-separated text nodes exists, so single-column bylines and non-academic docs
# are never touched. Killswitch + the min horizontal spread (pt) that counts as multi-column.
_AUTHOR_REORDER = (
    os.environ.get("SHUBO_DOCLING_AUTHOR_REORDER", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
_AUTHOR_COL_SPREAD = float(os.environ.get("SHUBO_DOCLING_AUTHOR_COL_SPREAD", "60"))
_AUTHOR_ROW_TOL = float(os.environ.get("SHUBO_DOCLING_AUTHOR_ROW_TOL", "16"))


def _reorder_author_block(doc: DoclingDocument) -> int:
    """Re-thread a scattered multi-column author/byline row into correct reading order (see the note
    above for why Docling scatters it). Pulls the band nodes out of whatever body/group they landed in
    and re-inserts them, spatially ordered, right after the title. Returns the number of nodes moved
    (0 = no multi-column author band found, or it was single-column and already ordered)."""
    if not _AUTHOR_REORDER:
        return 0
    try:
        from docling_core.types.doc import CoordOrigin, DocItemLabel, RefItem
    except Exception:  # noqa: BLE001
        return 0

    def _vscore(item) -> float:  # larger = higher on the page, regardless of coord origin
        b = item.prov[0].bbox
        o = getattr(b.coord_origin, "value", b.coord_origin)
        cy = (b.t + b.b) / 2.0
        return cy if o == CoordOrigin.BOTTOMLEFT.value else -cy

    title_label = getattr(DocItemLabel, "TITLE", DocItemLabel.SECTION_HEADER)
    header_labels = {DocItemLabel.SECTION_HEADER, title_label}
    headers = [t for t in doc.texts if t.prov and t.prov[0].page_no == 1 and t.label in header_labels]
    if len(headers) < 2:
        return 0
    headers.sort(key=_vscore, reverse=True)
    title, stop = headers[0], headers[1]
    top_v, bot_v = _vscore(title), _vscore(stop)
    if top_v <= bot_v:
        return 0

    band = [
        t
        for t in doc.texts
        if t.prov
        and t.prov[0].page_no == 1
        and t.label == DocItemLabel.TEXT
        and bot_v < _vscore(t) < top_v
    ]
    if len(band) < 2:
        return 0
    lefts = [t.prov[0].bbox.l for t in band]
    if max(lefts) - min(lefts) < _AUTHOR_COL_SPREAD:
        return 0  # single column — Docling read it in order; leave it

    # reading order: rows top→bottom (bucket vscore so a row's slight skew stays one row), then left→right
    band.sort(key=lambda t: (-round(_vscore(t) / _AUTHOR_ROW_TOL), t.prov[0].bbox.l))
    band_refs = {t.self_ref for t in band}

    # unlink the band nodes from wherever they currently live, emptying (and later dropping) any group
    # that held only band nodes.
    emptied = []
    for grp in doc.groups:
        kids = [c for c in grp.children if c.cref not in band_refs]
        if len(kids) != len(grp.children):
            grp.children = kids
            if not kids:
                emptied.append(grp.self_ref)
    drop = band_refs | set(emptied)
    doc.body.children = [c for c in doc.body.children if c.cref not in drop]

    # re-insert contiguously right after the title, in spatial order
    try:
        ti = next(i for i, c in enumerate(doc.body.children) if c.cref == title.self_ref)
    except StopIteration:
        ti = -1
    doc.body.children[ti + 1 : ti + 1] = [RefItem(cref=t.self_ref) for t in band]
    body_ref = RefItem(cref=doc.body.self_ref)
    for t in band:
        t.parent = body_ref
    return len(band)


# Docling models Table.footnotes (like Table.captions) but its layout leaves it EMPTY, so a table
# footnote ("∗ Self-consistency measured on Qwen2-7B only") floats as an unparented text leaf, detached
# from the table it explains. Detect + attach it. Markers a footnote opens with; gap below the table.
_TABLE_FOOTNOTES = (
    os.environ.get("SHUBO_DOCLING_TABLE_FOOTNOTES", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
_TABLE_FOOTNOTE_MARKERS = ("∗", "*", "†", "‡", "§", "¶", "⁎", "⋆")
_TABLE_FOOTNOTE_GAP = float(os.environ.get("SHUBO_DOCLING_TABLE_FOOTNOTE_GAP", "24"))


def _attach_table_footnotes(doc: DoclingDocument) -> int:
    """Attach a table's footnote(s) to Table.footnotes (mirroring Docling's Table.captions edge), which
    Docling's layout leaves empty. A text leaf is a footnote of a table when it is DIRECTLY BELOW the
    table (its top within `_TABLE_FOOTNOTE_GAP` of the table's bottom), its center is WITHIN the table's
    x-span, and it OPENS WITH A FOOTNOTE MARKER (∗ † ‡ § ¶ / leading *) or Docling already labels it
    `footnote`. Strictly scoped by marker + geometry so ordinary prose below a table is never attached.
    Returns the number of footnotes attached. Gated by `_TABLE_FOOTNOTES`."""
    if not _TABLE_FOOTNOTES:
        return 0
    from docling_core.types.doc import DocItemLabel, RefItem

    def _top(bb):
        return max(bb.t, bb.b)

    def _bot(bb):
        return min(bb.t, bb.b)

    attached = 0
    for table in doc.tables:
        if not table.prov:
            continue
        tp = table.prov[0]
        tb = tp.bbox
        if table.footnotes is None:
            table.footnotes = []
        seen = {r.cref for r in table.footnotes}
        for it in doc.texts:
            if not it.prov or it.prov[0].page_no != tp.page_no or it.self_ref in seen:
                continue
            txt = (it.text or "").lstrip()
            is_footnote = it.label == DocItemLabel.FOOTNOTE or (bool(txt) and txt[0] in _TABLE_FOOTNOTE_MARKERS)
            if not is_footnote:
                continue
            fb = it.prov[0].bbox
            # directly below the table (footnote top near/just below the table's bottom edge)
            if not (_bot(tb) - _TABLE_FOOTNOTE_GAP <= _top(fb) <= _bot(tb) + 2.0):
                continue
            fcx = (fb.l + fb.r) / 2.0  # horizontally within the table's column span
            if not (tb.l - 2.0 <= fcx <= tb.r + 2.0):
                continue
            table.footnotes.append(RefItem(cref=it.self_ref))
            seen.add(it.self_ref)
            attached += 1
    return attached


# CS papers frame ALGORITHM (LaTeX algorithm/algorithmic) and BOXED-CALLOUT (tcolorbox/mdframed)
# environments as a titled block, but Docling's layout mislabels the "Algorithm N …" / "Box N …" title
# as a SECTION_HEADER (which then pollutes every following chunk's section_path breadcrumb, like the
# "list" leak) and scatters the numbered pseudocode as bare list_items. Recognize the environment and
# relabel: the title → caption (no longer a fake section), the pseudocode steps → code. Gated.
_STRUCTURED_ENV = (
    os.environ.get("SHUBO_DOCLING_STRUCTURED_ENV", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
_ENV_CAPTION_RE = re.compile(r"^\s*(Algorithm|Procedure|Listing|Box|Sidebar|Panel)\s+\d+\b", re.IGNORECASE)
_ALGO_CAPTION_RE = re.compile(r"^\s*(Algorithm|Procedure|Listing)\s+\d+\b", re.IGNORECASE)
_ALGO_STEP_NUM_RE = re.compile(r"^\s*\d+\s*[:.)]")  # "1:", "15: end if", "1.", "1)"
_ALGO_STEP_KW_RE = re.compile(r"^\s*(Require|Ensure|Input|Output)\s*:", re.IGNORECASE)


def _recognize_structured_environments(doc: DoclingDocument):
    """Relabel algorithm / boxed-callout environments Docling mislabels. An "Algorithm N …" / "Box N …"
    title that Docling tagged SECTION_HEADER is demoted to CAPTION (so it stops acting as a document
    section + polluting section_path); for an ALGORITHM, the contiguous pseudocode steps that follow
    (numbered list_items, or Require:/Ensure:/Input:/Output: text) are relabelled CODE. Strictly scoped
    by the title regex + a contiguous, pattern-matched step run; ordinary prose is never touched.

    Returns ``(changed, blocks)`` where ``changed`` is the number of nodes relabelled and ``blocks`` is
    a list of ``(caption_item, [step_items])`` in reading order (steps empty for a box/callout) — the
    single source of truth for the Phase-2 grouping pass, so it never re-scans. Gated by `_STRUCTURED_ENV`."""
    if not _STRUCTURED_ENV:
        return 0, []
    from docling_core.types.doc import DocItemLabel

    texts = list(doc.texts)
    n = len(texts)
    changed = 0
    blocks = []
    for i, it in enumerate(texts):
        if it.label != DocItemLabel.SECTION_HEADER:
            continue
        s = it.text or ""
        if not _ENV_CAPTION_RE.match(s):
            continue
        it.label = DocItemLabel.CAPTION  # a fake section header → the environment's caption
        changed += 1
        steps = []
        if _ALGO_CAPTION_RE.match(s):
            # a Box/callout title needs no step relabelling (its body is a normal list) → steps stays []
            # Relabel the CONTIGUOUS pseudocode steps that follow (in reading order) → code.
            for j in range(i + 1, n):
                nt = texts[j]
                ns = nt.text or ""
                # A step is a numbered line ("1:", "1.", "1)") or a Require/Ensure/Input/Output keyword
                # line. Docling tags these inconsistently as list_item OR text (a "Require:" line can land
                # as either), so accept both labels for both step shapes rather than pairing shape to label.
                is_step = nt.label in (DocItemLabel.LIST_ITEM, DocItemLabel.TEXT) and (
                    bool(_ALGO_STEP_NUM_RE.match(ns)) or bool(_ALGO_STEP_KW_RE.match(ns))
                )
                if not is_step:
                    break  # first non-step ends the algorithm body
                nt.label = DocItemLabel.CODE
                steps.append(nt)
                changed += 1
        blocks.append((it, steps))
    return changed, blocks


# Phase 2 (ADR-0029 Part C): wrap each relabeled algorithm block (its CAPTION + contiguous CODE steps)
# into ONE structural GroupItem so the block is a single addressable, hierarchical unit — like a Table
# owning its caption + body — instead of loose siblings scattered in reading order. Independent sub-gate
# so the (surgical, label-only) relabel above can ship / be A-B tested without the tree-restructuring.
_STRUCTURED_ENV_GROUP = (
    os.environ.get("SHUBO_DOCLING_STRUCTURED_ENV_GROUP", "1").strip().lower()
    not in ("0", "false", "no", "off")
)


def _group_structured_environments(doc: DoclingDocument, blocks) -> int:
    """Wrap each algorithm block from ``blocks`` (see _recognize_structured_environments) into one
    GroupLabel.UNSPECIFIED GroupItem that owns the caption + CODE steps as children, in reading order.
    UNSPECIFIED is deliberate — a neutral container with NO section semantics (SECTION/CHAPTER would
    re-pollute section_path, the very thing the relabel fixes). GroupItem has no prov, so the block is
    bounded purely by membership (orthogonal to every geometry pass). Re-parents the EXISTING nodes via
    the direct-tree idiom (cf. _reorder_author_block); never renumbers #/texts/N. A box/callout (no CODE
    steps) is left ungrouped. Returns the number of groups created. Gated by `_STRUCTURED_ENV_GROUP`."""
    if not (_STRUCTURED_ENV_GROUP and blocks):
        return 0
    try:
        from docling_core.types.doc import GroupItem, GroupLabel, RefItem
    except Exception:  # noqa: BLE001
        return 0

    made = 0
    for caption, steps in blocks:
        if not steps:
            continue  # box/callout: caption only, nothing to group
        members = [caption, *steps]
        step_refs = {s.self_ref for s in steps}
        # the caption must be a direct body child to keep the block in place; if it is nested (rare),
        # skip the wrap rather than guess a parent (the nodes stay relabeled, just ungrouped).
        cap_idx = next((k for k, c in enumerate(doc.body.children) if c.cref == caption.self_ref), None)
        if cap_idx is None:
            continue
        gref = f"#/groups/{len(doc.groups)}"
        grp = GroupItem(
            self_ref=gref,
            parent=RefItem(cref=doc.body.self_ref),
            label=GroupLabel.UNSPECIFIED,
            name=(caption.text or "algorithm").strip()[:120],
        )
        doc.groups.append(grp)
        # keep the block's reading position: the caption's slot in body.children becomes the group ref,
        # then strip the step refs from body.children.
        doc.body.children[cap_idx] = RefItem(cref=gref)
        doc.body.children = [c for c in doc.body.children if c.cref not in step_refs]
        # steps may have lived in a ListGroup; pull them out and drop any list group left empty.
        for g in list(doc.groups):
            if g.self_ref == gref:
                continue
            kept = [c for c in g.children if c.cref not in step_refs]
            if len(kept) != len(g.children):
                g.children = kept
                if not kept:
                    doc.body.children = [c for c in doc.body.children if c.cref != g.self_ref]
        grp.children = [RefItem(cref=m.self_ref) for m in members]
        gpref = RefItem(cref=gref)
        for m in members:
            m.parent = gpref
        made += 1
    return made


def _set_prov_bbox_from_norm(item, norm_box: Tuple[float, float, float, float], pw: float, ph: float) -> None:
    """Replace a mapped element's geometry with a normalized 0..1 TOP-LEFT box (the matched OCR
    region's), converted to page coordinates — aligns a scanned-page element with the rendered page
    image (Docling's embedded-text-cell box clips the visual ink there)."""
    prov = getattr(item, "prov", None)
    if not prov:
        return
    l, t, r, b = norm_box
    prov[0].bbox = BoundingBox(
        l=l * pw, t=t * ph, r=r * pw, b=b * ph, coord_origin=CoordOrigin.TOPLEFT
    )


def _snap_scanned_geometry(doc: DoclingDocument, page_data: dict, image_backed: set) -> int:
    """Unified scanned-page geometry pass: replace each element's bbox with its best-matching OCR
    region's bbox so the overlay box hugs the rendered ink (Docling's embedded-text-cell boxes clip it).

    Iterates `doc.texts` (which — unlike `iterate_items` — INCLUDES furniture: page header/footer) and
    `doc.tables` (cells re-derive from the snapped table box). FORMULA leaves keep their geometry. Runs
    AFTER residual-ink picture detection so a grown text box can't erase a detected figure. Returns the
    number of elements snapped."""
    if not _SCAN_VISUAL_BBOX:
        return 0
    texts = [(it, False) for it in (getattr(doc, "texts", None) or [])]
    tables = [(it, True) for it in (getattr(doc, "tables", None) or [])]
    count = 0
    # Per-page vertical reduction (page points) measured from each body-text snap: how far the OCR
    # region extends ABOVE / BELOW Docling's clipped box. Reused to expand table cells (below), which
    # carry their own clipped box and have no per-cell OCR region to snap to.
    pads: dict = {}
    unmatched: list = []  # scanned text/furniture that found NO OCR region → expand by page pad below
    for item, is_table in texts + tables:
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        page_no = prov[0].page_no
        if page_no not in image_backed or page_no not in page_data:
            continue
        if not is_table and getattr(item, "label", None) == DocItemLabel.FORMULA:
            continue
        regions, pw, ph = page_data[page_no]
        box = _norm_tl(prov[0].bbox, pw, ph)
        # Tables snap to the visual table region; text/furniture to a non-table region. Fall back to
        # any region so an element is never left on its clipped box when a same-kind match is missing.
        same_kind = [r for r in regions if (r[0] == "table") == is_table]
        reg = _best_region(box, same_kind) or _best_region(box, regions)
        if not reg:
            if not is_table:
                unmatched.append((item, page_no))  # e.g. a footer the OCR emitted no region for
            continue
        if not is_table:  # box=[l,t,r,b] 0..1 TL; reg[1] likewise — record how much the box must grow
            tp, bp = (box[1] - reg[1][1]) * ph, (reg[1][3] - box[3]) * ph
            if tp > 0 or bp > 0:
                pads.setdefault(page_no, []).append((max(0.0, tp), max(0.0, bp)))
        _set_prov_bbox_from_norm(item, reg[1], pw, ph)
        count += 1
    if _TABLE_CELL_EXPAND:
        count += _expand_table_cells(tables, image_backed, pads)
        # Furniture/text with no OCR region (footer-snap is flaky across pages) → un-clip by the same
        # page-median expansion, so every scanned-page box hugs its ink regardless of OCR coverage.
        for item, page_no in unmatched:
            plist = pads.get(page_no)
            if not plist:
                continue
            b = item.prov[0].bbox
            h = abs(b.t - b.b)
            tp, bp = min(_median([p[0] for p in plist]), h), min(_median([p[1] for p in plist]), h)
            if tp <= 0 and bp <= 0:
                continue
            is_bl = getattr(b.coord_origin, "value", b.coord_origin) == CoordOrigin.BOTTOMLEFT.value
            item.prov[0].bbox = BoundingBox(
                l=b.l, r=b.r,
                t=b.t + tp if is_bl else b.t - tp,
                b=b.b - bp if is_bl else b.b + bp,
                coord_origin=b.coord_origin,
            )
            count += 1
    return count


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _expand_table_cells(tables, image_backed: set, pads: dict) -> int:
    """Expand each scanned-table cell's reduced bbox vertically by the page's MEDIAN body-text reduction
    (`pads`), so the overlay box hugs the cell text. Cells carry Docling's clipped box and have no
    per-cell OCR region to snap to; the reduction is the same per-line font margin as the body, so it's
    added as an absolute point pad (capped at the cell's own height so a multi-line cell isn't blown up)."""
    n = 0
    for item, _is_table in tables:
        prov = getattr(item, "prov", None)
        if not prov or prov[0].page_no not in image_backed:
            continue
        plist = pads.get(prov[0].page_no)
        if not plist:
            continue
        top_pad = _median([p[0] for p in plist])
        bot_pad = _median([p[1] for p in plist])
        if top_pad <= 0 and bot_pad <= 0:
            continue
        for cell in (getattr(getattr(item, "data", None), "table_cells", None) or []):
            b = getattr(cell, "bbox", None)
            if b is None:
                continue
            h = abs(b.t - b.b)
            tp, bp = min(top_pad, h), min(bot_pad, h)
            is_bl = getattr(b.coord_origin, "value", b.coord_origin) == CoordOrigin.BOTTOMLEFT.value
            cell.bbox = BoundingBox(
                l=b.l,
                t=b.t + tp if is_bl else b.t - tp,
                r=b.r,
                b=b.b - bp if is_bl else b.b + bp,
                coord_origin=b.coord_origin,
            )
            n += 1
    return n


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

    # Repair multi-column reading-order merges (a 3-up author block threaded into one node) using
    # PyMuPDF line geometry, BEFORE density/garble are measured so each split column counts on its own.
    # Then coalesce any same-block fragments Docling over-split (an author name kept apart from its
    # affiliation/email) so each author is ONE consistent node — split first, then re-join.
    _resegment_overflow_nodes(doc, source)
    _coalesce_text_blocks(doc, source)
    # Re-thread a scattered multi-column author/byline row into correct reading order (Docling flings
    # the columns across body.children, key_value_area groups, and — after the re-split above — the
    # document tail). Runs after coalesce so each author is one node before it is placed.
    _reorder_author_block(doc)
    # Attach detached table footnotes to their table (Docling leaves Table.footnotes empty).
    _attach_table_footnotes(doc)

    # NATIVE (pre-OCR) digital-text density per page, captured on the original layout doc before any
    # grounded rebuild replaces `doc`. Drives both the digital-text fast path and the scanned-page
    # picture fallback (a sparse native layer ⇒ an image-dominated/scanned page).
    native_chars_by_page = _digital_text_chars_by_page(doc)
    # Page renders, captured here too: the grounded rebuild (`_build_doc_from_grounded`) makes a fresh
    # doc WITHOUT page images, so the residual-ink picture detector below would otherwise lose them.
    page_images = {
        p: pg.image.pil_image
        for p, pg in doc.pages.items()
        if pg.image is not None and pg.image.pil_image is not None
    }
    # Scanned (image-backed) pages: never digital-fast-pathed (their embedded-text geometry clips), and
    # their mapped elements adopt the visual OCR-region bbox below.
    image_backed = _image_backed_pages(source)

    # Formula enrichment: turn every layout-detected FORMULA leaf into deterministic LaTeX. Run on the
    # original layout doc (formula regions + page images exist for BOTH digital and image-only PDFs);
    # `formula_regions` carries them forward so the image-PDF grounded rebuild can re-inject them.
    formula_regions = _enrich_formulas(doc)

    # OCR each page exactly once (Unlimited-OCR is full-page); cache the grounded regions.
    # Digital-text fast path: skip the (expensive, GPU-bound) OCR for any page whose embedded text
    # layer already covers it — Docling's extracted text is exact and free, so those pages keep it
    # and are simply absent from page_data (the mapping below is a no-op for them). Scanned / empty
    # -layer pages fall through to OCR exactly as before.
    digital_chars = native_chars_by_page if _DIGITAL_FASTPATH else {}
    # Hybrid-page reconcile: pages classified digital that ALSO carry a garbled embedded span (e.g. a
    # handwritten date an upstream tool OCR'd into the text layer as garbage). Those are NOT skipped —
    # they get the whole-page OCR so the garbled span(s) can be replaced, while clean spans are kept.
    # GATED on image_backed: only a SCANNED page can have an OCR-mangled embedded span worth fixing; a
    # BORN-DIGITAL page's text layer is ground truth (never reconcile it). Without this gate the garble
    # heuristic false-positives on dense math/notation (single-char vars, spaced symbols, "( x 1 , ...")
    # in a digital paper and needlessly OCRs most pages — the 13-page-paper slowdown.
    garbled_pages = (
        {p for p in _pages_with_garbled_spans(doc) if p in image_backed}
        if (_DIGITAL_FASTPATH and _RECONCILE_GARBLED)
        else set()
    )
    page_data: dict = {}
    reconcile_pages: set = set()  # digital pages OCR'd ONLY to fix garbled spans (clean spans kept)
    skipped_digital = 0
    for page_no in sorted(doc.pages):
        page = doc.pages[page_no]
        img = page.image.pil_image if (page.image is not None) else None
        if img is None or page.size is None:
            continue
        is_digital = _DIGITAL_FASTPATH and digital_chars.get(page_no, 0) >= _MIN_DIGITAL_TEXT_CHARS
        if is_digital and page_no not in garbled_pages and page_no not in image_backed:
            skipped_digital += 1
            continue  # clean digital text layer → keep Docling's exact text, skip MLX OCR (zero cost)
        page_data[page_no] = (
            parse_grounded_regions(ocr_raw(img)),
            float(page.size.width),
            float(page.size.height),
        )
        if is_digital:
            reconcile_pages.add(page_no)  # hybrid page: OCR only to overwrite the garbled span(s)
    logger.debug(
        "digital-text fast path: %d/%d page(s) used the embedded layer (OCR skipped); %d OCR'd "
        "(%d hybrid reconcile)",
        skipped_digital, len(doc.pages), len(page_data), len(reconcile_pages),
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
        # First pass: map each OCR region onto Docling's existing layout elements. Track which
        # regions get consumed and collect each element's box (per page) so the recovery pass below
        # can tell which regions matched no element.
        consumed: set[int] = set()
        elem_boxes: dict[int, List[Tuple[float, float, float, float]]] = {}
        for item, _ in doc.iterate_items():
            prov = getattr(item, "prov", None)
            if not prov or prov[0].page_no not in page_data:
                continue
            page_no = prov[0].page_no
            regions, pw, ph = page_data[page_no]
            box = _norm_tl(prov[0].bbox, pw, ph)
            elem_boxes.setdefault(page_no, []).append(box)
            reconcile = page_no in reconcile_pages
            if isinstance(item, TableItem):
                # On a hybrid reconcile page the table cells are clean digital text → keep them; the
                # reconcile only targets garbled body spans. Scanned pages fill the grid from OCR.
                if reconcile:
                    continue
                treg = _best_region(box, [r for r in regions if r[0] == "table"]) or _best_region(
                    box, regions
                )
                if treg:
                    _fill_table_from_html(item, treg[2])
                    consumed.add(id(treg))
            elif isinstance(item, TextItem):
                # Keep CodeFormula's LaTeX on formula leaves — never overwrite it with the (garbled)
                # full-page-OCR text that overlaps the formula region. The formula leaf's box is still
                # recorded in elem_boxes above, so the recovery pass treats it like any other element.
                if item.label == DocItemLabel.FORMULA:
                    continue
                reg = _best_region(box, [r for r in regions if r[0] != "table"])
                if reg and reg[2]:
                    # Hybrid reconcile page: overwrite ONLY garbled embedded spans with the OCR text;
                    # keep clean digital text / IDs verbatim. Consume the region either way so the
                    # recovery pass below never re-emits a clean kept element as a duplicate node.
                    if not reconcile or _is_garbled_text(item.text or ""):
                        item.text = reg[2]
                        item.orig = reg[2]
                    # NOTE: scanned-page bbox snapping is NOT done here — it runs as a unified pass
                    # (`_snap_scanned_geometry`) AFTER residual-ink picture detection, so it covers
                    # furniture (header/footer, which `iterate_items` skips) and tables too, and can't
                    # erase a residual-ink figure by pre-growing a text box over it.
                    consumed.add(id(reg))

        # Second pass: recover regions Docling's layout under-segmented. Full-page OCR sometimes
        # finds text that maps to NO Docling element (its centre lies in no element box and its IoU
        # with every element is below the floor), so the first pass dropped it (e.g. an author column
        # the layout model failed to detect). Emit a text node for each so the content isn't silently
        # lost, skipping degenerate/placeholder regions the OCR emits when it loops. This branch only
        # runs for OCR'd pages (those in page_data); digital fast-path pages are absent → no-op.
        recovered = 0
        for page_no in sorted(page_data):
            if page_no in reconcile_pages:
                # Hybrid digital page: trust Docling's layout completely (only garbled spans were
                # fixed in place). Recovering "unmapped" OCR regions here would duplicate the clean
                # embedded text wherever the OCR segmented the page differently from the layout.
                continue
            regions, pw, ph = page_data[page_no]
            boxes = elem_boxes.get(page_no, [])
            # Boxes of formulas Docling DETECTED and CodeFormula rendered to LaTeX (kept on their
            # leaves in this structured path). A region transcribing one of these is the garbled OCR
            # of math we already have as LaTeX → skip it so we don't emit a duplicate raw-text node.
            # A formula Docling MISSED has no such box, so its region is still recovered below.
            fboxes = [b for b, _ in formula_regions.get(page_no, [])]
            for reg in regions:
                label, box, text = reg
                if id(reg) in consumed or label == "table" or not text:
                    continue
                if _PLACEHOLDER_RE.match(text.strip()):
                    continue
                if box[2] - box[0] < _MIN_REGION_DIM or box[3] - box[1] < _MIN_REGION_DIM:
                    continue  # zero/near-zero area → an OCR loop artifact, not placeable content
                resolved = _resolve_label(label, box)
                if resolved == DocItemLabel.PICTURE:
                    # An unmapped picture region (a figure the layout model missed): keep it as a
                    # PictureItem so the visual-description pipeline can describe it, rather than
                    # dropping it silently. Other non-text regions (table/chart/form/kv) are built
                    # elsewhere or have no describe path, so they stay skipped.
                    doc.add_picture(
                        prov=ProvenanceItem(
                            page_no=page_no,
                            bbox=BoundingBox(
                                l=box[0] * pw, t=box[1] * ph, r=box[2] * pw, b=box[3] * ph,
                                coord_origin=CoordOrigin.TOPLEFT,
                            ),
                            charspan=(0, 0),
                        )
                    )
                    continue
                if resolved in _NON_TEXT_LABELS:
                    continue
                cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                in_element = any(
                    eb[0] - 1e-6 <= cx <= eb[2] + 1e-6 and eb[1] - 1e-6 <= cy <= eb[3] + 1e-6
                    for eb in boxes
                )
                if in_element or any(_iou(box, eb) >= 0.1 for eb in boxes):
                    continue
                if _center_in_any(box, fboxes):
                    continue  # overlaps a detected formula → CodeFormula LaTeX already carries it
                prov = ProvenanceItem(
                    page_no=page_no,
                    bbox=BoundingBox(
                        l=box[0] * pw, t=box[1] * ph, r=box[2] * pw, b=box[3] * ph,
                        coord_origin=CoordOrigin.TOPLEFT,
                    ),
                    charspan=(0, len(text)),
                )
                doc.add_text(label=resolved, text=text, orig=text, prov=prov)
                recovered += 1
        logger.debug("recovered %d unmapped OCR region(s) from full-page OCR", recovered)

    # Correct any body line Docling's layout mislabeled as page furniture — runs on the FINAL doc so
    # it covers both the grounded rebuild and the digital/structured path (which keeps layout labels
    # verbatim). Without this, a mislabeled content line is dropped from chunks by furniture chunking.
    _correct_furniture_labels(doc)

    # Recognize algorithm / boxed-callout environments Docling mislabels (an "Algorithm/Box N ..." title
    # tagged section_header; pseudocode steps tagged list_items). MUST run on the FINAL doc — like the
    # furniture pass above — so it covers BOTH the digital/structured path and the grounded-rebuild
    # (scanned/image-only) path: before the rebuild a scanned page has no body text, so the pass is a
    # no-op there and the wholesale rebuild would then discard it.
    _changed, _algo_blocks = _recognize_structured_environments(doc)
    # Phase 2: wrap each relabeled algorithm block into one group container (hierarchical unit).
    _group_structured_environments(doc, _algo_blocks)

    # Scanned/image-dominated pages: box each non-text graphic (signature/stamp/figure) via residual-
    # ink detection, so it reaches the visual-description pipeline with a TIGHT bbox. Runs on the FINAL
    # doc → covers the grounded rebuild and the digital/structured path.
    page_pics = _add_scanned_page_pictures(doc, native_chars_by_page, page_images)
    if page_pics:
        logger.debug("added %d residual-ink picture(s) on scanned/image-dominated page(s)", page_pics)

    # Now (after picture detection) align every scanned-page element — body text, furniture
    # (header/footer), and tables — to its visual OCR-region geometry so the overlay boxes hug the ink.
    snapped = _snap_scanned_geometry(doc, page_data, image_backed)
    if snapped:
        logger.debug("snapped %d scanned-page element(s) to visual OCR geometry", snapped)

    page_nos = sorted(doc.pages)
    markdown_pages = (
        [doc.export_to_markdown(page_no=p) for p in page_nos]
        if page_nos
        else [doc.export_to_markdown()]
    )
    return {"markdown_pages": markdown_pages, "structured_document": doc.export_to_dict()}
