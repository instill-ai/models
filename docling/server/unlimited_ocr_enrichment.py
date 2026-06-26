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
    TableCell,
    TableData,
    TableItem,
    TextItem,
)

OcrRaw = Callable[[Image.Image], str]

# A grounded region: (label, (l, t, r, b) normalized 0..1 top-left, text).
Region = Tuple[str, Tuple[float, float, float, float], str]


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


def _grid_to_table_data(grid: List[List[str]]) -> Optional[TableData]:
    if not grid:
        return None
    n_cols = max(len(r) for r in grid)
    cells = []
    for ri, row in enumerate(grid):
        for ci, txt in enumerate(row):
            cells.append(
                TableCell(
                    text=txt,
                    start_row_offset_idx=ri,
                    end_row_offset_idx=ri + 1,
                    start_col_offset_idx=ci,
                    end_col_offset_idx=ci + 1,
                    column_header=(ri == 0),
                )
            )
    return TableData(table_cells=cells, num_rows=len(grid), num_cols=n_cols)


_HEADING_LABELS = {"title", "section_header", "header", "subtitle"}


def _build_doc_from_grounded(page_data: dict) -> DoclingDocument:
    """Build a DoclingDocument purely from Unlimited-OCR's grounded regions — the fallback for
    image-only PDFs, where Docling's layout (no embedded text, OCR off) produces nothing. Each
    region becomes a text/heading/table node with a page-coordinate provenance bbox.

    page_data: {page_no: (regions, page_width, page_height)}.
    """
    doc = DoclingDocument(name="document")
    for page_no in sorted(page_data):
        regions, pw, ph = page_data[page_no]
        doc.add_page(page_no=page_no, size=Size(width=pw, height=ph))
        for label, box, text in regions:
            bbox = BoundingBox(
                l=box[0] * pw,
                t=box[1] * ph,
                r=box[2] * pw,
                b=box[3] * ph,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            prov = ProvenanceItem(page_no=page_no, bbox=bbox, charspan=(0, len(text)))
            if label == "table":
                td = _grid_to_table_data(_html_to_grid(text))
                if td is not None:
                    doc.add_table(data=td, prov=prov)
            else:
                doc_label = (
                    DocItemLabel.SECTION_HEADER if label in _HEADING_LABELS else DocItemLabel.TEXT
                )
                doc.add_text(label=doc_label, text=text, orig=text, prov=prov)
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

    # OCR each page exactly once (Unlimited-OCR is full-page); cache the grounded regions.
    page_data: dict = {}
    for page_no in sorted(doc.pages):
        page = doc.pages[page_no]
        img = page.image.pil_image if (page.image is not None) else None
        if img is None or page.size is None:
            continue
        page_data[page_no] = (
            parse_grounded_regions(ocr_raw(img)),
            float(page.size.width),
            float(page.size.height),
        )

    # Does Docling's layout carry usable structure? Digital PDFs: yes. Image-only PDFs: no text
    # elements and empty tables (layout needs embedded text / a working OCR) → build from the
    # grounded OCR instead.
    text_items = [it for it, _ in doc.iterate_items() if isinstance(it, TextItem)]
    table_items = [it for it, _ in doc.iterate_items() if isinstance(it, TableItem)]
    docling_has_structure = bool(text_items) or any(t.data.table_cells for t in table_items)

    if not docling_has_structure and page_data:
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
                reg = _best_region(box, [r for r in regions if r[0] != "table"])
                if reg and reg[2]:
                    item.text = reg[2]
                    item.orig = reg[2]

    page_nos = sorted(doc.pages)
    markdown_pages = (
        [doc.export_to_markdown(page_no=p) for p in page_nos]
        if page_nos
        else [doc.export_to_markdown()]
    )
    return {"markdown_pages": markdown_pages, "structured_document": doc.export_to_dict()}
