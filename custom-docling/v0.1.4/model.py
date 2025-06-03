from typing import List, Tuple
import io
import base64
import os
import asyncio

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import parse_custom_input, construct_custom_output

from utils import process_document

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    PowerpointFormatOption,
    WordFormatOption,
)
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.pipeline.simple_pipeline import SimplePipeline
from docling_core.types.doc import PictureItem


@instill_deployment
class DocumentProcessor:
    def __init__(self):
        """Initialize document processing pipeline"""

        settings.perf.doc_batch_size = 1  # Process one document at a time
        settings.perf.doc_batch_concurrency = 1  # Keep at 1 for thread safety
        settings.perf.page_batch_size = 8  # Increase page batch size
        settings.perf.page_batch_concurrency = 4  # Increase page batch concurrency
        settings.perf.elements_batch_size = 16

        self.pipeline_options = PdfPipelineOptions(artifacts_path="docling-models")
        self.pipeline_options.generate_page_images = False
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.do_picture_description = False
        self.pipeline_options.do_formula_enrichment = False
        self.pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.CUDA
        )
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.PPTX,
                InputFormat.DOCX
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options
                ),
                InputFormat.PPTX: PowerpointFormatOption(
                    pipeline_cls=SimplePipeline
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline
                ),
            },
        )
        self.unit_sep = '\x1f-Instill-Unit-Seperator-\x1f'

    async def __call__(self, request):
        """Process document inference request"""
        # Parse inputs using custom input parser
        inputs = await parse_custom_input(request)

        outputs = []
        for input_data in inputs:
            # Process the document
            markdown_pages, extracted_images, pages_with_images = (
                await self._process_document(input_data["data"]["doc_content"])
            )

            # Structure the output
            output = {
                "data": {
                    "markdown_pages": markdown_pages,
                    "extracted_images": extracted_images,
                    "pages_with_images": pages_with_images,
                }
            }
            outputs.append(output)

        # Return formatted output
        return construct_custom_output(request, outputs)

    async def _process_document(
        self, base64_content: str
    ) -> Tuple[List[str], List[str], List[int]]:

        # Decode the base64 content and create a DocumentStream
        source = DocumentStream(
            name="doc",
            stream=io.BytesIO(
                base64.b64decode(
                    await process_document(base64_content)
                )
            )
        )

        # Convert using DocumentStream
        result = await asyncio.to_thread(self.converter.convert, source)

        extracted_images = []
        pages_with_images = []

        # Extract images
        for element, _level in result.document.iterate_items():
            if isinstance(element, PictureItem):
                # For DOCX files, we'll use a default page number of 1 since the number of pages aren't properly registered (Docling bug)
                page_no = element.prov[0].page_no if element.prov else 1
                if hasattr(element, "image") and element.image is not None:
                    extracted_images.append(str(element.image.uri))
                    pages_with_images.append(page_no)

        # Export to markdown using the unit separator between pages
        markdown = result.document.export_to_markdown(
            page_break_placeholder=self.unit_sep
        )

        markdown = markdown.split(self.unit_sep)

        return markdown, extracted_images, pages_with_images


os.environ["RAY_IS_HIGH_SCALE_MODEL"] = "true"
entrypoint = InstillDeployable(DocumentProcessor).get_deployment_handle()
