from typing import List, Tuple
import io
import base64

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_custom_input,
    construct_custom_output
)

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
from docling_core.types.doc import PictureItem
from docling.pipeline.simple_pipeline import SimplePipeline


@instill_deployment
class DocumentProcessor:
    def __init__(self):
        """Initialize document processing pipeline"""
        self.pipeline_options = PdfPipelineOptions(artifacts_path="docling-models")
        self.pipeline_options.images_scale = 150/72.0
        self.pipeline_options.generate_page_images = True
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.do_formula_enrichment = True
        self.pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CUDA
        )
        self.pipeline_options.do_picture_description = False
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
                )
            }
        )

    async def __call__(self, request):
        """Process document inference request"""
        # Parse inputs using custom input parser
        inputs = await parse_custom_input(request)

        outputs = []
        for input_data in inputs:
            # Process the document
            markdown_pages, extracted_images, pages_with_images = self._process_document(
                input_data["doc_content"]
            )

            # Structure the output
            output = {
                "markdown_pages": markdown_pages,
                "extracted_images": extracted_images,
                "pages_with_images": pages_with_images
            }
            outputs.append(output)

        # Return formatted output
        return construct_custom_output(request, outputs)

    def _process_document(self, base64_content: str) -> Tuple[List[str], List[str], List[int]]:
        """Internal method to process a single document"""
        # Decode base64 content to bytes
        doc_content = base64.b64decode(base64_content)

        # Create BytesIO object and DocumentStream
        doc_stream = io.BytesIO(doc_content)
        source = DocumentStream(name="doc", stream=doc_stream)

        # Convert using DocumentStream
        result = self.converter.convert(source)

        extracted_images = []
        markdown_pages = []
        pages_with_images = []
        # Extract images
        for element, _level in result.document.iterate_items():
            if isinstance(element, PictureItem):
                # For DOCX files, we'll use a default page number of 1 since the number of pages aren't properly registered (Docling bug)
                page_no = element.prov[0].page_no if element.prov else 1
                if hasattr(element, 'image') and element.image is not None:
                    extracted_images.append(str(element.image.uri))
                    pages_with_images.append(page_no)
        # For DOCX files, we need to handle the case where num_pages() returns 0
        if result.document.num_pages() == 0:
            # Get markdown for the entire document as a single page
            full_md = result.document.export_to_markdown()
            markdown_pages.append(full_md)
        else:
            # Process markdown pages normally for PDF and PPTX
            for page in range(result.document.num_pages()):
                page_no = page + 1
                page_md = result.document.export_to_markdown(page_no=page_no)
                markdown_pages.append(page_md)
        return markdown_pages, extracted_images, pages_with_images


entrypoint = InstillDeployable(DocumentProcessor).get_deployment_handle()
