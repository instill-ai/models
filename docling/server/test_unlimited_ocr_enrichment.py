"""Tests for the Docling-structure + full-page Unlimited-OCR mapping.

Unit tests for the byte-BPE decode + grounded-region parser + HTML-table alignment run anywhere.
The integration test runs the REAL Docling layout/table pipeline on a generated PDF with a MOCK
full-page OCR (grounded regions), validating that structure is restored AND each element/table is
filled from the mapped OCR text — without the ~3.7 GB Unlimited-OCR weights.
"""
import pytest

from unlimited_ocr_enrichment import (
    decode_bpe,
    parse_grounded_regions,
    _build_doc_from_grounded,
    _html_to_grid,
    _resolve_label,
)


def test_resolve_label_covers_full_doctags_vocabulary_and_furniture():
    from docling_core.types.doc import DocItemLabel

    mid = (0.1, 0.4, 0.9, 0.45)
    # Body text in the middle stays TEXT; top/bottom bands become furniture.
    assert _resolve_label("text", mid) == DocItemLabel.TEXT
    assert _resolve_label("text", (0.1, 0.02, 0.9, 0.05)) == DocItemLabel.PAGE_HEADER
    assert _resolve_label("text", (0.1, 0.95, 0.9, 0.98)) == DocItemLabel.PAGE_FOOTER
    # Aliases for the model's label strings.
    assert _resolve_label("title", mid) == DocItemLabel.SECTION_HEADER
    assert _resolve_label("header", mid) == DocItemLabel.PAGE_HEADER
    assert _resolve_label("footer", mid) == DocItemLabel.PAGE_FOOTER
    assert _resolve_label("equation", mid) == DocItemLabel.FORMULA
    # Full vocabulary passes straight through (the official parser can't emit these).
    for name in ("list_item", "formula", "code", "footnote", "caption", "reference", "section_header"):
        assert _resolve_label(name, mid) == DocItemLabel(name)


def test_decode_bpe_undoes_byte_artifacts():
    # Ġ→space, Ċ→newline (GPT-2 byte-level).
    assert decode_bpe("1.ĠBACKGROUND") == "1. BACKGROUND"
    assert decode_bpe("aĊb") == "a\nb"


def test_parse_grounded_regions_decodes_and_normalizes():
    raw = (
        "çŃĨnoise"  # leading hallucinated preamble before the first box — must be dropped
        "<|det|>titleĠ[0,Ġ0,Ġ999,Ġ100]<|/det|>1.ĠBACKGROUND"
        "Ċ<|det|>textĠ[0,Ġ200,Ġ999,Ġ300]<|/det|>TheĠDirectorĠapproves."
    )
    regions = parse_grounded_regions(raw)
    assert [r[0] for r in regions] == ["title", "text"]
    assert regions[0][2] == "1. BACKGROUND"
    assert regions[1][2] == "The Director approves."
    # 0..999 → 0..1 top-left.
    assert regions[0][1] == pytest.approx((0.0, 0.0, 1.0, 100 / 999), abs=1e-6)


def test_html_to_grid_parses_rows_and_cells():
    grid = _html_to_grid(
        "<table><tr><td>Subscriber</td><td>Shares</td></tr>"
        "<tr><td>RTP Global</td><td>22,757</td></tr></table>"
    )
    assert grid == [["Subscriber", "Shares"], ["RTP Global", "22,757"]]


def test_build_doc_from_grounded_makes_structure_for_image_pdfs():
    # The image-only fallback: grounded regions → DoclingDocument covering the FULL label
    # vocabulary, furniture by position, and tables via the official robust HTML parser.
    regions = [
        ("text", (0.05, 0.02, 0.50, 0.04), "CONFIDENTIAL"),  # top band → page_header
        ("title", (0.05, 0.06, 0.30, 0.08), "1. BACKGROUND"),  # → section_header
        ("text", (0.05, 0.09, 0.90, 0.11), "The Director approves the investment."),
        ("list_item", (0.07, 0.13, 0.90, 0.15), "(a) first condition"),
        ("formula", (0.07, 0.16, 0.50, 0.18), "E = mc^2"),
        ("table", (0.05, 0.20, 0.86, 0.32),
         "<table><tr><td>Subscriber</td><td>Shares</td></tr><tr><td>RTP</td><td>22757</td></tr></table>"),
        ("text", (0.05, 0.96, 0.50, 0.98), "Page 1 of 5"),  # bottom band → page_footer
    ]
    doc = _build_doc_from_grounded({1: (regions, 1240.0, 1754.0)})
    sd = doc.export_to_dict()
    labels = {t["label"] for t in sd["texts"]}
    assert {"section_header", "text", "page_header", "page_footer", "list_item", "formula"} <= labels
    assert sd["texts"][0]["prov"][0]["bbox"]  # page-coordinate provenance
    assert len(sd["tables"]) == 1
    tb = sd["tables"][0]["data"]
    assert {c["text"] for c in tb["table_cells"]} >= {"Subscriber", "RTP", "22757"}


# ── integration: real Docling structure + mocked full-page OCR ────────────────────────────────
docling = pytest.importorskip("docling")
fpdf = pytest.importorskip("fpdf")

from docling_core.types.doc import TableItem, TextItem  # noqa: E402

from unlimited_ocr_enrichment import convert_to_contract  # noqa: E402


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


def _grounded(label, l, t, r, b, text):
    # Emit a region in the model's raw byte-BPE grounded format (spaces as Ġ).
    coords = f"{l},Ġ{t},Ġ{r},Ġ{b}".replace(" ", "")
    return f"<|det|>{label}Ġ[{coords}]<|/det|>{text.replace(' ', 'Ġ')}"


def test_structure_restored_and_text_mapped_from_full_page_ocr(tmp_path, monkeypatch):
    # This case validates the OCR-mapping path (scanned-page behavior). The generated PDF HAS a
    # digital text layer, so disable the digital-text fast path here — otherwise OCR is (correctly)
    # skipped for it; the fast path itself is covered by the test below.
    monkeypatch.setattr("unlimited_ocr_enrichment._DIGITAL_FASTPATH", False)
    pdf_path = str(tmp_path / "doc.pdf")
    _make_pdf(pdf_path)
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    # One mock full-page OCR: grounded regions spanning the heading / intro / table bands.
    HTML = (
        "<table><tr><td>Subscriber</td><td>No. of Shares</td><td>Monies</td></tr>"
        "<tr><td>OCR RTP</td><td>22757</td><td>2355000</td></tr>"
        "<tr><td>OCR Berlin</td><td>7731</td><td>800000</td></tr></table>"
    )
    # The content is short and top-stacked on the A4 page: heading ~y0.07, intro ~0.095,
    # table ~0.10-0.19. Bands are sized so each element's center falls in the right region.
    raw = (
        _grounded("title", 30, 0, 970, 85, "1. BACKGROUND (ocr)")
        + _grounded("text", 30, 85, 970, 110, "The Director approves the investment. (ocr)")
        + _grounded("table", 20, 110, 980, 250, HTML)
    )
    calls = {"n": 0}

    def ocr_raw(_image):
        calls["n"] += 1
        return raw

    out = convert_to_contract(pdf_bytes, ocr_raw)

    # Page OCR'd exactly once (not per element).
    assert calls["n"] == 1

    sd = out["structured_document"]
    assert sd.get("schema_name") == "DoclingDocument"
    assert len(sd.get("tables", [])) == 1 and len(sd.get("texts", [])) >= 2
    # DocTags structure with bboxes survived.
    full_md = "\n".join(out["markdown_pages"])
    assert "(ocr)" in full_md  # text came from the mapped OCR, not Docling's (empty) layer
    # Table cells were filled from the OCR region's HTML, aligned to Docling's grid.
    assert "OCR RTP" in full_md and "OCR Berlin" in full_md
    assert sd["texts"][0]["prov"][0]["bbox"]  # provenance bboxes intact


def test_digital_text_fast_path_skips_ocr_for_digital_pdf(tmp_path, monkeypatch):
    # A digital PDF (embedded text layer) must use Docling's exact text and SKIP the expensive MLX
    # OCR entirely — the per-page fast path. Force the fast path on with a low char floor so the
    # test is independent of the ambient env / exact extracted-char count.
    monkeypatch.setattr("unlimited_ocr_enrichment._DIGITAL_FASTPATH", True)
    monkeypatch.setattr("unlimited_ocr_enrichment._MIN_DIGITAL_TEXT_CHARS", 10)
    pdf_path = str(tmp_path / "doc.pdf")
    _make_pdf(pdf_path)
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    calls = {"n": 0}

    def ocr_raw(_image):
        calls["n"] += 1
        return "SHOULD NOT BE CALLED"

    out = convert_to_contract(pdf_bytes, ocr_raw)

    # OCR was skipped for the digital page (this is the whole point of the fast path).
    assert calls["n"] == 0
    full_md = "\n".join(out["markdown_pages"])
    # Docling's own exact digital text is what lands — not the OCR fake.
    assert "BACKGROUND" in full_md and "Director" in full_md
    assert "SHOULD NOT BE CALLED" not in full_md
    sd = out["structured_document"]
    assert sd.get("schema_name") == "DoclingDocument"
    assert len(sd.get("texts", [])) >= 2  # structure (heading + intro + …) preserved
    assert len(sd.get("tables", [])) == 1  # the table structure survived (cells from the text layer)
