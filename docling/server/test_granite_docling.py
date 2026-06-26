import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import granite_docling


class DoclingModelDefaultsTest(unittest.TestCase):
    def test_default_model_is_unlimited_ocr_mxfp8(self):
        self.assertEqual(
            granite_docling.DEFAULT_MODEL,
            "sahilchachra/unlimited-ocr-mxfp8-mlx",
        )

    def test_unlimited_ocr_uses_native_prompt_without_chat_template(self):
        self.assertTrue(
            granite_docling._is_unlimited_ocr_model(
                "sahilchachra/unlimited-ocr-mxfp8-mlx"
            )
        )
        self.assertEqual(
            granite_docling._default_prompt(
                "sahilchachra/unlimited-ocr-mxfp8-mlx"
            ),
            "<image>document parsing.",
        )

    def test_prompt_override_wins(self):
        with patch.dict(os.environ, {"SHUBO_DOCLING_PROMPT": "<image>Free OCR."}):
            self.assertEqual(
                granite_docling._default_prompt(
                    "sahilchachra/unlimited-ocr-mxfp8-mlx"
                ),
                "<image>Free OCR.",
            )


class MarkdownDoclingDocumentTest(unittest.TestCase):
    def test_markdown_pages_are_wrapped_as_docling_text_nodes(self):
        image = Image.new("RGB", (640, 480), "white")
        doc = granite_docling._markdown_to_docling_document(
            ["# Invoice\n\nTotal: $12"],
            [image],
            "sahilchachra/unlimited-ocr-mxfp8-mlx",
        )

        self.assertEqual(doc["schema_name"], "DoclingDocument")
        self.assertEqual(doc["body"]["children"], [{"$ref": "#/texts/0"}])
        self.assertEqual(doc["texts"][0]["text"], "# Invoice\n\nTotal: $12")
        self.assertEqual(doc["texts"][0]["label"], "text")
        self.assertEqual(doc["texts"][0]["prov"][0]["page_no"], 1)
        self.assertEqual(
            doc["texts"][0]["prov"][0]["bbox"],
            {"l": 0, "t": 0, "r": 640, "b": 480, "coord_origin": "TOPLEFT"},
        )
        self.assertEqual(doc["pages"]["1"]["size"], {"width": 640, "height": 480})
        self.assertTrue(
            doc["pages"]["1"]["image"]["uri"].startswith("data:image/png;base64,")
        )


if __name__ == "__main__":
    unittest.main()
