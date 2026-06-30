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
    _add_scanned_page_pictures,
    _detect_visual_regions,
    _image_backed_pages,
    _set_prov_bbox_from_norm,
    _snap_scanned_geometry,
    _expand_table_cells,
    _median,
    _build_doc_from_grounded,
    _center_in_any,
    _correct_furniture_labels,
    _enrich_formulas,
    _html_to_grid,
    _inject_formulas,
    _is_garbled_text,
    _PAGE_PICTURE_MAX_NATIVE_CHARS,
    _resolve_label,
)


def test_garble_detector_flags_ocr_cursive_not_clean_text_or_ids():
    # Tuned against real DocuSign-page spans. The handwritten date an upstream tool glyph-split into
    # the text layer is garbled; clean digital prose and a long no-space alphanumeric ID are NOT.
    assert _is_garbled_text("Dated:  I 2- D e lo cf.,r  Zo  ZZ") is True
    assert _is_garbled_text("Name of Director: Ping-Lin Chang") is False
    # A long alphanumeric ID must never be flagged (it is a single, clearly-good token).
    assert _is_garbled_text("DocuSign Envelope ID: A7E30573-7EC3-4F1A-9C2D-1234567890AB") is False
    assert _is_garbled_text("The Director approves the proposed investment.") is False
    assert _is_garbled_text("") is False
    # Too few tokens to judge → never flagged (protects short clean labels).
    assert _is_garbled_text("I 2 D") is False


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


def test_build_doc_from_grounded_emits_picture_for_image_region():
    # A region the OCR model labels "image"/"figure" must become a PictureItem (so the visual
    # description pipeline can describe it) — not be silently dropped.
    regions = [
        ("text", (0.05, 0.05, 0.50, 0.07), "Figure 1: the system"),
        ("image", (0.10, 0.20, 0.80, 0.60), ""),
    ]
    doc = _build_doc_from_grounded({1: (regions, 1240.0, 1754.0)})
    assert len(doc.pictures) == 1
    bbox = doc.pictures[0].prov[0].bbox
    # bbox carried through in page coordinates (0.10..0.80 * 1240).
    assert bbox.l == pytest.approx(0.10 * 1240.0, abs=1)
    assert bbox.r == pytest.approx(0.80 * 1240.0, abs=1)


def test_detect_visual_regions_boxes_residual_ink_and_skips_erased():
    # Residual-ink detector: ink OUTSIDE any detected-element box is a graphic and is boxed tightly;
    # ink INSIDE an erase rect (already-explained text) is removed and never surfaces.
    import numpy as np

    W, H = 1240, 1754
    gray = np.full((H, W), 255, np.uint8)
    gray[400:470, 300:520] = 0   # a graphic (signature-like) blob in the body
    gray[800:860, 300:520] = 0   # an "ink" blob that lives under a detected text box
    erase = [(290, 790, 530, 870)]  # px box around the second blob
    regions = _detect_visual_regions(gray, erase, W, H)
    assert len(regions) == 1
    x0, y0, x1, y1 = regions[0]
    assert 250 <= x0 <= 305 and 380 <= y0 <= 405   # tight around the first blob
    assert 515 <= x1 <= 560 and 465 <= y1 <= 490


def test_add_scanned_page_pictures_boxes_graphic_on_sparse_page_only():
    # A sparse (scanned) page: the residual graphic becomes a tight PictureItem; a text-dense page is
    # treated as digital and skipped.
    import numpy as np
    from PIL import Image

    W, H = 1240, 1754
    arr = np.full((H, W), 255, np.uint8)
    arr[400:470, 300:520] = 0  # graphic outside the (sparse) text box
    img = Image.fromarray(arr)
    text = [("text", (0.1, 0.6, 0.4, 0.62), "Signed:")]

    doc = _build_doc_from_grounded({1: (text, 595.0, 842.0)})
    added = _add_scanned_page_pictures(doc, {1: 30}, {1: img})
    assert added == 1
    sx = W / 595.0
    bbox = doc.pictures[0].prov[0].bbox
    assert bbox.l == pytest.approx(300 / sx, abs=8)  # tight, not the whole page
    assert bbox.r == pytest.approx(520 / sx, abs=8)

    dense = _build_doc_from_grounded({1: (text, 595.0, 842.0)})
    added2 = _add_scanned_page_pictures(dense, {1: _PAGE_PICTURE_MAX_NATIVE_CHARS + 1}, {1: img})
    assert added2 == 0


def test_image_backed_pages_detects_full_page_image_only():
    # A page rendered from a (near) full-page raster image is "scanned"; a pure-vector page is not.
    import io
    import fitz
    import numpy as np
    from PIL import Image

    doc = fitz.open()
    pg = doc.new_page(width=595, height=842)
    buf = io.BytesIO()
    Image.fromarray((np.ones((200, 140, 3)) * 255).astype("uint8")).save(buf, format="PNG")
    pg.insert_image(pg.rect, stream=buf.getvalue())  # cover the whole page
    assert _image_backed_pages(doc.tobytes()) == {1}

    vec = fitz.open()
    vec.new_page(width=595, height=842).insert_text((50, 50), "vector text only")
    assert _image_backed_pages(vec.tobytes()) == set()


def test_set_prov_bbox_from_norm_writes_topleft_page_coords():
    doc = _build_doc_from_grounded({1: ([("text", (0.1, 0.1, 0.5, 0.12), "x")], 595.0, 842.0)})
    item = doc.texts[0]
    _set_prov_bbox_from_norm(item, (0.2, 0.3, 0.6, 0.34), 595.0, 842.0)
    b = item.prov[0].bbox
    assert (b.l, b.t, b.r, b.b) == pytest.approx((0.2 * 595, 0.3 * 842, 0.6 * 595, 0.34 * 842))
    assert str(b.coord_origin).upper().endswith("TOPLEFT")


def test_snap_scanned_geometry_aligns_text_and_furniture_to_ocr_regions():
    # Both a body line AND a top-band furniture line (page_header) must be snapped to their OCR
    # region bbox on a scanned page — furniture is the case the in-loop snap missed.
    doc = _build_doc_from_grounded(
        {1: ([
            ("text", (0.10, 0.30, 0.50, 0.33), "body line"),
            ("text", (0.10, 0.02, 0.50, 0.04), "HEADER LINE"),  # top band → page_header
        ], 595.0, 842.0)}
    )
    from docling_core.types.doc import DocItemLabel

    assert any(t.label == DocItemLabel.PAGE_HEADER for t in doc.texts)  # furniture exists
    regions = [
        ("text", (0.08, 0.28, 0.55, 0.35), "body line"),
        ("header", (0.08, 0.01, 0.55, 0.05), "HEADER LINE"),
    ]
    n = _snap_scanned_geometry(doc, {1: (regions, 595.0, 842.0)}, image_backed={1})
    assert n >= 2
    for t in doc.texts:
        assert str(t.prov[0].bbox.coord_origin).upper().endswith("TOPLEFT")  # snapped to visual
    body = next(t for t in doc.texts if t.label != DocItemLabel.PAGE_HEADER)
    assert body.prov[0].bbox.t == pytest.approx(0.28 * 842.0, abs=2)  # the OCR region's top, not 0.30

    # Gate: a non-scanned page (image_backed empty) is untouched.
    doc2 = _build_doc_from_grounded({1: ([("text", (0.1, 0.3, 0.5, 0.33), "x")], 595.0, 842.0)})
    assert _snap_scanned_geometry(doc2, {1: (regions, 595.0, 842.0)}, image_backed=set()) == 0


def test_expand_table_cells_grows_clipped_cells_by_page_median_pad():
    # A scanned table cell carries Docling's reduced (clipped) bbox; expand it by the page's median
    # body-text reduction so the overlay box hugs the cell text.
    from types import SimpleNamespace
    from docling_core.types.doc import BoundingBox, CoordOrigin

    assert _median([5.0, 4.0, 1.0]) == 4.0  # sanity

    cell = SimpleNamespace(
        bbox=BoundingBox(l=10, t=100, r=50, b=107, coord_origin=CoordOrigin.TOPLEFT)  # 7pt clipped
    )
    table = SimpleNamespace(
        prov=[SimpleNamespace(page_no=1)], data=SimpleNamespace(table_cells=[cell])
    )
    pads = {1: [(4.0, 1.0), (5.0, 1.0)]}  # median top=4.5, bot=1.0
    n = _expand_table_cells([(table, True)], image_backed={1}, pads=pads)
    assert n == 1
    assert cell.bbox.t == pytest.approx(100 - 4.5)  # top grew UP
    assert cell.bbox.b == pytest.approx(107 + 1.0)  # bottom grew DOWN

    # Non-scanned (no page pad) → untouched.
    cell2 = SimpleNamespace(bbox=BoundingBox(l=10, t=100, r=50, b=107, coord_origin=CoordOrigin.TOPLEFT))
    table2 = SimpleNamespace(prov=[SimpleNamespace(page_no=1)], data=SimpleNamespace(table_cells=[cell2]))
    assert _expand_table_cells([(table2, True)], image_backed={1}, pads={}) == 0
    assert cell2.bbox.t == 100


def test_correct_furniture_labels_demotes_body_mislabels():
    # Docling's digital/structured path keeps layout labels verbatim, and the layout model sometimes
    # tags a real content line as page_header/page_footer. The correction pass demotes a CONTENT-LIKE
    # line that sits OUTSIDE the margin band back to TEXT, while keeping genuine furniture (in-band) and
    # short page-number-like furniture displaced into the body.
    from docling_core.types.doc import (
        BoundingBox,
        CoordOrigin,
        DocItemLabel,
        DoclingDocument,
        ProvenanceItem,
        Size,
    )

    doc = DoclingDocument(name="t")
    doc.add_page(page_no=1, size=Size(width=100.0, height=100.0))

    def add(label, text, top, bot):
        prov = ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=10.0, t=top, r=90.0, b=bot, coord_origin=CoordOrigin.TOPLEFT),
            charspan=(0, len(text)),
        )
        return doc.add_text(label=label, text=text, orig=text, prov=prov)

    # Mislabeled mid-page content (content-like, outside the bands) → demote to TEXT.
    body_footer = add(DocItemLabel.PAGE_FOOTER, "This is a real body sentence mislabeled as a footer.", 40.0, 44.0)
    body_header = add(DocItemLabel.PAGE_HEADER, "Another mislabeled mid-page content heading line.", 45.0, 48.0)
    # Genuine furniture in the margin bands → keep.
    real_footer = add(DocItemLabel.PAGE_FOOTER, "Journal of Foo, Vol 3, 2026", 95.0, 98.0)
    real_header = add(DocItemLabel.PAGE_HEADER, "Running head: FOO BAR", 2.0, 5.0)
    # Short page-number furniture displaced into the body (not content-like) → keep.
    page_num = add(DocItemLabel.PAGE_FOOTER, "12", 50.0, 52.0)

    _correct_furniture_labels(doc)

    assert body_footer.label == DocItemLabel.TEXT
    assert body_header.label == DocItemLabel.TEXT
    assert real_footer.label == DocItemLabel.PAGE_FOOTER
    assert real_header.label == DocItemLabel.PAGE_HEADER
    assert page_num.label == DocItemLabel.PAGE_FOOTER


# ── formula enrichment: pure helpers (no model) ───────────────────────────────────────────────
def test_center_in_any_detects_containment():
    boxes = [(0.10, 0.10, 0.40, 0.20)]
    assert _center_in_any((0.20, 0.12, 0.30, 0.18), boxes) is True  # centre inside
    assert _center_in_any((0.50, 0.50, 0.60, 0.60), boxes) is False  # disjoint
    assert _center_in_any((0.0, 0.0, 1.0, 1.0), []) is False  # no formula boxes


def test_inject_formulas_adds_latex_region_and_suppresses_overlapping_ocr():
    # OCR produced a garbled transcription of the formula plus a real body paragraph; injection must
    # drop the overlapping OCR region, add a `formula` region carrying the LaTeX, and keep the body.
    regions = [
        ("text", (0.10, 0.10, 0.40, 0.20), "E equals m c squared garbled"),  # overlaps the formula
        ("text", (0.10, 0.50, 0.90, 0.55), "Real body paragraph"),          # disjoint → kept
        ("table", (0.10, 0.12, 0.40, 0.18), "<table></table>"),             # table never suppressed
    ]
    page_data = {1: (regions, 1000.0, 1400.0)}
    formula_regions = {1: [((0.10, 0.10, 0.40, 0.20), r"E = mc^{2}")]}
    _inject_formulas(page_data, formula_regions)
    out = page_data[1][0]
    labels_texts = [(r[0], r[2]) for r in out]
    assert ("formula", r"E = mc^{2}") in labels_texts
    assert ("text", "Real body paragraph") in labels_texts
    assert ("text", "E equals m c squared garbled") not in labels_texts  # suppressed
    assert ("table", "<table></table>") in labels_texts  # tables survive overlap


def test_inject_formulas_then_build_yields_formula_node_with_latex():
    # End-to-end of the image-PDF formula path WITHOUT the model: inject + grounded-build →
    # a FORMULA node whose text is the LaTeX (what the persisted formula chunk will carry).
    regions = [("text", (0.05, 0.50, 0.90, 0.55), "Some body text about attention.")]
    page_data = {1: (list(regions), 1000.0, 1400.0)}
    _inject_formulas(page_data, {1: [((0.10, 0.10, 0.60, 0.20), r"\frac{QK^{T}}{\sqrt{d_k}}")]})
    doc = _build_doc_from_grounded(page_data)
    formulas = [t for t in doc.export_to_dict()["texts"] if t["label"] == "formula"]
    assert len(formulas) == 1
    assert formulas[0]["text"] == r"\frac{QK^{T}}{\sqrt{d_k}}"


def test_enrich_formulas_killswitch_is_noop(monkeypatch):
    # With the killswitch off, enrichment must NOT touch the model or the doc (returns empty harvest).
    monkeypatch.setattr("unlimited_ocr_enrichment._FORMULA_ENRICHMENT", False)

    def _boom():  # pragma: no cover - must never be called
        raise AssertionError("model must not load when formula enrichment is disabled")

    monkeypatch.setattr("unlimited_ocr_enrichment._get_formula_model", _boom)
    assert _enrich_formulas(_build_doc_from_grounded({1: ([], 100.0, 100.0)})) == {}


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


# ── hybrid-page reconcile: a clean digital page carrying one garbled embedded span ─────────────
def _make_hybrid_pdf(path: str, lines) -> None:
    # A clean digital PDF whose embedded text layer is written at known vertical positions (mm on an
    # A4 page) so the mock OCR's grounded bands can be aimed at each line's center.
    pdf = fpdf.FPDF()  # A4 portrait, units mm → page is 210 x 297 mm
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    for y_mm, text in lines:
        pdf.set_xy(15.0, y_mm)
        pdf.cell(0, 8, text)
    pdf.output(path)


def test_hybrid_page_reconciles_garbled_span_keeps_clean_digital_text(tmp_path, monkeypatch):
    # The core bug: a digital page (DocuSign-style) sails over the char floor but carries ONE garbled
    # embedded span (a handwritten date an upstream tool glyph-split). The fast path must NOT blindly
    # trust it: it OCR's the page and replaces ONLY the garbled element with the whole-page OCR text,
    # while keeping every clean digital element (prose + the long alphanumeric ID) verbatim.
    monkeypatch.setattr("unlimited_ocr_enrichment._DIGITAL_FASTPATH", True)
    monkeypatch.setattr("unlimited_ocr_enrichment._RECONCILE_GARBLED", True)
    monkeypatch.setattr("unlimited_ocr_enrichment._MIN_DIGITAL_TEXT_CHARS", 10)

    pdf_path = str(tmp_path / "hybrid.pdf")
    _make_hybrid_pdf(
        pdf_path,
        [
            (20.0, "1. BACKGROUND"),                                              # center ~0.081
            (60.0, "Name of Director: Ping-Lin Chang"),                           # center ~0.215
            (100.0, "DocuSign Envelope ID: A7E30573-7EC3-4F1A-9C2D-1234567890AB"),  # center ~0.350
            (160.0, "Dated:  I 2- D e lo cf.,r  Zo  ZZ"),                         # center ~0.552 GARBLED
        ],
    )
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    # Whole-page OCR: a band over each line. The clean lines carry DELIBERATELY WRONG text — if the
    # reconcile mistakenly overwrote a clean element, the wrong text would leak into the output.
    raw = (
        _grounded("title", 30, 40, 970, 120, "1. BACKGROUND")
        + _grounded("text", 30, 180, 970, 260, "WRONG OCR NAME")
        + _grounded("text", 30, 310, 970, 390, "WRONG OCR ID")
        + _grounded("text", 30, 510, 970, 600, "Dated: 12 December 2022")  # corrects the garbled span
    )
    calls = {"n": 0}

    def ocr_raw(_image):
        calls["n"] += 1
        return raw

    out = convert_to_contract(pdf_bytes, ocr_raw)
    full_md = "\n".join(out["markdown_pages"])

    # The page WAS OCR'd (it has a garbled span) — exactly once, whole-page.
    assert calls["n"] == 1
    # Garbled span replaced by the correct OCR reading.
    assert "Dated: 12 December 2022" in full_md
    assert "Zo  ZZ" not in full_md and "cf.,r" not in full_md
    # Clean digital elements kept verbatim — NOT overwritten with the wrong OCR text.
    assert "Ping-Lin Chang" in full_md
    assert "A7E30573-7EC3-4F1A-9C2D-1234567890AB" in full_md
    assert "WRONG OCR NAME" not in full_md
    assert "WRONG OCR ID" not in full_md


def test_clean_digital_page_does_not_trigger_reconcile_ocr(tmp_path, monkeypatch):
    # A fully-clean digital page (no garbled span) must keep the zero-cost fast path: the MLX OCR is
    # NEVER called, even with reconcile enabled.
    monkeypatch.setattr("unlimited_ocr_enrichment._DIGITAL_FASTPATH", True)
    monkeypatch.setattr("unlimited_ocr_enrichment._RECONCILE_GARBLED", True)
    monkeypatch.setattr("unlimited_ocr_enrichment._MIN_DIGITAL_TEXT_CHARS", 10)

    pdf_path = str(tmp_path / "clean.pdf")
    _make_hybrid_pdf(
        pdf_path,
        [
            (20.0, "1. BACKGROUND"),
            (60.0, "Name of Director: Ping-Lin Chang"),
            (100.0, "DocuSign Envelope ID: A7E30573-7EC3-4F1A-9C2D-1234567890AB"),
            (160.0, "The Director approves the proposed investment on the stated terms."),
        ],
    )
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    calls = {"n": 0}

    def ocr_raw(_image):
        calls["n"] += 1
        return "SHOULD NOT BE CALLED"

    out = convert_to_contract(pdf_bytes, ocr_raw)
    full_md = "\n".join(out["markdown_pages"])

    assert calls["n"] == 0  # clean page → no OCR pass at all
    assert "Ping-Lin Chang" in full_md
    assert "SHOULD NOT BE CALLED" not in full_md


# ── OCR-recovery: unmapped-region pass (scanned / under-segmented pages) ───────────────────────
# These exercise the recovery branch in convert_to_contract's structured (else) path. They use the
# digital test PDF, so the digital-text fast path is disabled — otherwise OCR is skipped and the
# page is absent from page_data (the recovery is correctly a no-op for digital fast-path pages,
# which is the composition guarantee; here we want the OCR'd-page behaviour).
def test_unmapped_ocr_region_is_recovered_as_text_node(tmp_path, monkeypatch):
    # An OCR region in an area Docling's layout left empty (no element) must be recovered as its own
    # text node instead of silently dropped — the fix for author columns the layout under-segments.
    monkeypatch.setattr("unlimited_ocr_enrichment._DIGITAL_FASTPATH", False)
    pdf_path = str(tmp_path / "doc.pdf")
    _make_pdf(pdf_path)
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    # One region maps onto the heading; a second sits low on the page where the short test PDF has
    # no text element at all → no Docling element to map onto.
    raw = (
        _grounded("title", 30, 0, 970, 85, "1. BACKGROUND (ocr)")
        + _grounded("text", 50, 700, 950, 740, "Recovered Author kraska@example.edu")
    )

    out = convert_to_contract(pdf_bytes, lambda _img: raw)
    full_md = "\n".join(out["markdown_pages"])
    assert "Recovered Author kraska@example.edu" in full_md
    # No duplication: the recovered text appears exactly once.
    assert full_md.count("Recovered Author kraska@example.edu") == 1


def test_recovery_does_not_duplicate_a_mapped_region(tmp_path, monkeypatch):
    # A region that maps onto an existing Docling element is consumed in the first pass; the recovery
    # pass must add NOTHING for it (no duplicate node). Recovered nodes are created with TOPLEFT
    # provenance, whereas mapped Docling elements keep the layout's own (BOTTOMLEFT) provenance — so a
    # recovered duplicate is identifiable regardless of how Docling segments the page.
    monkeypatch.setattr("unlimited_ocr_enrichment._DIGITAL_FASTPATH", False)
    pdf_path = str(tmp_path / "doc.pdf")
    _make_pdf(pdf_path)
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    raw = _grounded("title", 30, 0, 970, 85, "1. BACKGROUND (ocr)")  # maps onto Docling's heading

    out = convert_to_contract(pdf_bytes, lambda _img: raw)
    sd = out["structured_document"]
    recovered_texts = [
        t["text"]
        for t in sd["texts"]
        if t["prov"] and t["prov"][0]["bbox"].get("coord_origin") == "TOPLEFT"
    ]
    # The heading region mapped onto an element → it must NOT also appear as a recovered node.
    assert "1. BACKGROUND (ocr)" not in recovered_texts


def test_recovery_skips_degenerate_and_placeholder_regions(tmp_path, monkeypatch):
    # The OCR model loops, emitting zero-area boxes and bracketed placeholders in empty page space;
    # those must NOT become recovered nodes (no garbage), while a genuine unmapped line is kept.
    monkeypatch.setattr("unlimited_ocr_enrichment._DIGITAL_FASTPATH", False)
    pdf_path = str(tmp_path / "doc.pdf")
    _make_pdf(pdf_path)
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    raw = (
        _grounded("title", 30, 0, 970, 85, "1. BACKGROUND (ocr)")
        + _grounded("text", 50, 700, 950, 740, "Genuine recovered line")  # placeable → kept
        + _grounded("ref_text", 171, 930, 826, 930, "Looped zero height junk")  # zero height
        + _grounded("text", 0, 0, 3, 3, "Tiny corner junk")  # below min dim
        + _grounded("text", 50, 760, 950, 800, "[Non-Text]")  # placeholder marker
    )

    out = convert_to_contract(pdf_bytes, lambda _img: raw)
    full_md = "\n".join(out["markdown_pages"])
    assert "Genuine recovered line" in full_md
    assert "Looped zero height junk" not in full_md
    assert "Tiny corner junk" not in full_md
    assert "[Non-Text]" not in full_md


def test_recovery_skips_region_overlapping_detected_formula(tmp_path, monkeypatch):
    # A region whose centre sits inside a Docling-DETECTED formula box (CodeFormula already rendered
    # it to LaTeX) is the garbled OCR of that same math → it must be SKIPPED, not recovered as a
    # duplicate raw-text node. A genuine unmapped line elsewhere is still recovered. We stub
    # _enrich_formulas to inject a formula box without needing the GPU model or a real formula PDF.
    monkeypatch.setattr("unlimited_ocr_enrichment._DIGITAL_FASTPATH", False)
    fbox = (0.05, 0.70, 0.95, 0.74)  # normalized 0..1 formula box, low on the (empty) page body
    monkeypatch.setattr(
        "unlimited_ocr_enrichment._enrich_formulas",
        lambda _doc: {1: [(fbox, r"E = mc^{2}")]},
    )
    pdf_path = str(tmp_path / "doc.pdf")
    _make_pdf(pdf_path)
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    # The garbled-formula region's centre (≈0.50, 0.72) lies inside fbox; the footnote (≈0.86) does
    # not. Both are in empty page space (no Docling element), so absent the formula guard BOTH would
    # be recovered — the assertion proves only the formula-overlapping one is suppressed.
    raw = (
        _grounded("title", 30, 0, 970, 85, "1. BACKGROUND (ocr)")
        + _grounded("text", 50, 710, 950, 730, "E equals m c squared garbled ocr")
        + _grounded("text", 50, 850, 950, 880, "Genuine recovered footnote line")
    )

    out = convert_to_contract(pdf_bytes, lambda _img: raw)
    full_md = "\n".join(out["markdown_pages"])
    assert "Genuine recovered footnote line" in full_md  # genuine unmapped region recovered
    assert "E equals m c squared garbled ocr" not in full_md  # formula transcription suppressed
