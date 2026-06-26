"""Integration test for the Unlimited-OCR Docling enrichment.

Runs the REAL Docling layout/table pipeline on a generated PDF with a MOCK OCR callable, so it
validates end-to-end that (a) Docling restores structure (section_header / text / table with cells
+ DocTags + bboxes) and (b) the enrichment re-OCRs every text element and table cell — without
needing the ~3.7 GB Unlimited-OCR MLX weights. The server injects the real OCR call in production.

Skips cleanly where the heavy Docling stack (or fpdf2 for the fixture) isn't installed.
"""
import pytest

docling = pytest.importorskip("docling")
fpdf = pytest.importorskip("fpdf")

from docling_core.types.doc import TableItem, TextItem  # noqa: E402

from unlimited_ocr_enrichment import build_converter, convert_to_contract  # noqa: E402


def _make_pdf(path: str) -> None:
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "1. BACKGROUND", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, "The Director needs to approve the proposed investment.")
    pdf.ln(3)
    cols, widths = ["Subscriber", "No. of Shares", "Monies"], [70, 40, 50]
    rows = [["RTP Global Partners III", "22,757", "$2,355,000"],
            ["Berlin Innovation Ventures", "7,731", "$800,000"]]
    pdf.set_font("Helvetica", "B", 11)
    for i, c in enumerate(cols):
        pdf.cell(widths[i], 9, c, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", size=10)
    for r in rows:
        for i, c in enumerate(r):
            pdf.cell(widths[i], 9, c, border=1)
        pdf.ln()
    pdf.output(path)


MARKER = "OCR_ENRICHED"


def test_structure_restored_and_every_element_reocred(tmp_path):
    pdf_path = str(tmp_path / "doc.pdf")
    _make_pdf(pdf_path)

    calls = {"n": 0}

    def mock_ocr(image):
        calls["n"] += 1
        return MARKER

    doc = build_converter(mock_ocr).convert(pdf_path).document

    # (a) Docling restored real structure — not one flat text node per page.
    labels = {}
    tables = []
    for item, _ in doc.iterate_items():
        labels[str(getattr(item, "label", None))] = labels.get(str(getattr(item, "label", None)), 0) + 1
        if isinstance(item, TableItem):
            tables.append(item)
    assert "section_header" in str(labels), f"missing heading structure: {labels}"
    assert len(tables) == 1, f"expected exactly one table, got {labels}"
    table = tables[0]
    assert table.data.num_rows >= 2 and table.data.num_cols == 3, "table grid not detected"
    assert len(table.data.table_cells) >= 6, "table cells not detected"

    # DocTags carry per-element bounding boxes (<loc_...>) and the table (OTSL) — the grounding.
    doctags = doc.export_to_doctags()
    assert "<loc_" in doctags and "<otsl>" in doctags

    # (b) The enrichment re-OCR'd every text element AND every table cell.
    text_items = [it for it, _ in doc.iterate_items() if isinstance(it, TextItem)]
    assert text_items and all(it.text == MARKER for it in text_items), "text not re-OCR'd"
    assert all(c.text == MARKER for c in table.data.table_cells if c.bbox), "cells not re-OCR'd"

    # One OCR call per text element + per table cell (proves the wiring fans out to cells).
    assert calls["n"] >= len(text_items) + sum(1 for c in table.data.table_cells if c.bbox)


def test_convert_to_contract_shape(tmp_path):
    pdf_path = str(tmp_path / "doc.pdf")
    _make_pdf(pdf_path)
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    out = convert_to_contract(pdf_bytes, lambda image: MARKER)

    # The host-server contract the parsing-router consumes.
    assert set(out) >= {"markdown_pages", "structured_document"}
    assert isinstance(out["markdown_pages"], list) and out["markdown_pages"]
    sd = out["structured_document"]
    assert sd.get("schema_name") == "DoclingDocument"
    assert len(sd.get("texts", [])) >= 2 and len(sd.get("tables", [])) == 1
    # Structure carries provenance bboxes (the grounding the dashboard overlays).
    assert sd["texts"][0]["prov"][0]["bbox"]
