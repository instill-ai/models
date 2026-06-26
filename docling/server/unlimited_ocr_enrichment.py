"""Unlimited-OCR as a Docling enrichment — high-accuracy text on a real DoclingDocument.

The shubo MLX docling host server (granite_docling.py) trades Docling's layout pipeline for a
single VLM. That works for granite-docling (which emits DocTags directly) but NOT for a pure OCR
model like Unlimited-OCR, which emits flat Markdown — so the structure (tables, headers, per-element
bboxes / DocTags) is lost (models #72 regression).

This module keeps BOTH: Docling's real `DocumentConverter` runs the **layout + table-structure
models** (do_ocr disabled — RapidOCR isn't needed, and is broken on this stack), producing the
DoclingDocument structure + per-element bboxes + DocTags. Then a `BaseItemAndImageEnrichmentModel`
re-OCRs **each text / formula element and each table cell** from its bbox crop with Unlimited-OCR,
writing the high-accuracy text back onto the element. Result: Unlimited-OCR accuracy AND DocTags.

`ocr_image` is injected (a `PIL.Image -> str` callable) so the enrichment is testable without the
~3.7 GB MLX model; the server passes the real Unlimited-OCR call.
"""
from __future__ import annotations

import io
from collections.abc import Callable, Iterable
from typing import Union

from PIL import Image

from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import DoclingDocument, NodeItem, TableItem, TextItem

OcrImage = Callable[[Image.Image], str]


class UnlimitedOcrEnrichmentModel(BaseItemAndImageEnrichmentModel):
    """Re-OCR every citable text leaf, formula, and table cell with Unlimited-OCR.

    `is_processable` accepts any TextItem (text / section_header / list_item / caption / formula)
    and any TableItem; `__call__` receives each element already cropped to its bbox (`elem.image`)
    and overwrites the element's text with the OCR output. Table cells are cropped out of the table
    image by their own bbox (coord-origin aware) and OCR'd individually.
    """

    # 2x the layout resolution keeps small glyphs legible for the OCR model without bloating crops.
    images_scale: float = 2.0
    # Pad the crop slightly so edge glyphs/ascenders aren't clipped.
    expansion_factor: float = 0.02

    def __init__(self, enabled: bool, ocr_image: OcrImage):
        self.enabled = enabled
        self.ocr_image = ocr_image

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, (TextItem, TableItem))

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            return
        for enriched in element_batch:
            item = enriched.item
            if isinstance(item, TableItem):
                self._enrich_table(doc, item, enriched.image)
            elif isinstance(item, TextItem):
                text = self._ocr(enriched.image)
                if text:
                    item.text = text
                    item.orig = text
            yield item

    def _ocr(self, image: Image.Image) -> str:
        try:
            return (self.ocr_image(image) or "").strip()
        except Exception:  # noqa: BLE001 — one bad element must not abort the whole document
            return ""

    def _enrich_table(
        self, doc: DoclingDocument, item: TableItem, table_image: Image.Image
    ) -> None:
        """OCR each cell from its crop within the table image, writing back cell.text.

        The table image spans the table's bbox (expanded by expansion_factor) at images_scale. The
        table bbox and the cell bboxes can be in DIFFERENT coord origins (the table is PDF-native
        BOTTOMLEFT; cells come from the table-structure model), so both are normalized to TOPLEFT
        via the page height before computing the cell's pixel offset within the table image.
        """
        prov = item.prov[0] if item.prov else None
        if prov is None:
            return
        page = doc.pages.get(prov.page_no)
        if page is None or page.size is None:
            return
        page_h = page.size.height
        table_tl = prov.bbox.to_top_left_origin(page_h).expand_by_scale(
            self.expansion_factor, self.expansion_factor
        )
        scale = self.images_scale
        img_w, img_h = table_image.size
        for cell in item.data.table_cells:
            cb = getattr(cell, "bbox", None)
            if cb is None:
                continue
            ctl = cb.to_top_left_origin(page_h)
            box = (
                max(0, int(round((ctl.l - table_tl.l) * scale))),
                max(0, int(round((ctl.t - table_tl.t) * scale))),
                min(img_w, int(round((ctl.r - table_tl.l) * scale))),
                min(img_h, int(round((ctl.b - table_tl.t) * scale))),
            )
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            text = self._ocr(table_image.crop(box))
            if text:
                cell.text = text


def make_unlimited_ocr_pipeline(ocr_image: OcrImage) -> type[StandardPdfPipeline]:
    """A StandardPdfPipeline whose enrichment stage is the Unlimited-OCR re-OCR, capturing the
    injected OCR callable (DocumentConverter instantiates the pipeline with options only)."""

    class _UnlimitedOcrPipeline(StandardPdfPipeline):
        def __init__(self, pipeline_options: PdfPipelineOptions):
            super().__init__(pipeline_options)
            self.enrichment_pipe = [
                UnlimitedOcrEnrichmentModel(
                    enabled=getattr(pipeline_options, "do_unlimited_ocr", True),
                    ocr_image=ocr_image,
                )
            ]
            # The enrichment crops element images from the page, so the page backend must stay open.
            self.keep_backend = True

    return _UnlimitedOcrPipeline


def build_converter(ocr_image: OcrImage) -> DocumentConverter:
    """A DocumentConverter that yields a structured DoclingDocument (layout + tables → DocTags +
    bboxes) with every text/formula/cell re-OCR'd by Unlimited-OCR. Docling's own OCR is OFF."""
    opts = PdfPipelineOptions()
    opts.do_ocr = False            # text comes from Unlimited-OCR (enrichment), not RapidOCR
    opts.do_table_structure = True  # the structure we were missing
    opts.generate_page_images = True  # required so the enrichment can crop element images
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=make_unlimited_ocr_pipeline(ocr_image),
                pipeline_options=opts,
            )
        }
    )


def convert_to_contract(source: Union[str, bytes], ocr_image: OcrImage) -> dict:
    """Convert a PDF (path or raw bytes) into the docling host-server contract:
    `{markdown_pages: [...], structured_document: {...}}` (schema_name "DoclingDocument"),
    exactly what routedConvertResultParser consumes — but now with real structure + DocTags
    and Unlimited-OCR text. `structured_document` is the DoclingDocument tree the artifact
    persists to evidence_tree; `markdown_pages` is one Markdown string per page.
    """
    from docling.datamodel.base_models import DocumentStream

    if isinstance(source, bytes):
        src: Union[str, DocumentStream] = DocumentStream(
            name="document.pdf", stream=io.BytesIO(source)
        )
    else:
        src = source
    doc: DoclingDocument = build_converter(ocr_image).convert(src).document
    page_nos = sorted(doc.pages)
    markdown_pages = (
        [doc.export_to_markdown(page_no=p) for p in page_nos]
        if page_nos
        else [doc.export_to_markdown()]
    )
    return {"markdown_pages": markdown_pages, "structured_document": doc.export_to_dict()}
