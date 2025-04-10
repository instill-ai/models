from typing import List, Tuple
import io
import base64
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
from docling.pipeline.simple_pipeline import SimplePipeline
import os


@instill_deployment
class DocumentProcessor:
    def __init__(self):
        """Initialize document processing pipeline"""
        self.pipeline_options = PdfPipelineOptions(
            artifacts_path="docling-models"
        )
        self.pipeline_options.generate_page_images = False
        self.pipeline_options.generate_picture_images = False
        self.pipeline_options.do_formula_enrichment = True
        self.pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.CUDA
        )
        self.pipeline_options.do_picture_description = False
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.PPTX,
                InputFormat.DOCX,
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
        self.unit_sep = '\x1f'

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
        """Internal method to process a single document"""

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
        result = await asyncio.to_thread(
            self.converter.convert,
            source)

        # For DOCX files, we need to handle the case where num_pages() returns 0
        markdown = await asyncio.to_thread(
            result.document.export_to_markdown,
            page_break_placeholder=self.unit_sep
        )

        markdown = markdown.split(self.unit_sep)

        return markdown, [], []


os.environ["RAY_IS_HIGH_SCALE_MODEL"] = "true"

deployable = InstillDeployable(DocumentProcessor)
entrypoint = deployable.get_deployment_handle()
