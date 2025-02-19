from typing import List, Tuple
import io
import base64

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_custom_input,
    construct_custom_output
)

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    smolvlm_picture_description
)
from docling_core.types.doc import PictureItem


@instill_deployment
class DocumentProcessor:
    def __init__(self):
        """Initialize document processing pipeline"""
        self.pipeline_options = PdfPipelineOptions(artifacts_path="docling-models")
        self.pipeline_options.images_scale = 300/72.0
        self.pipeline_options.generate_page_images = True
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CUDA
        )
        self.pipeline_options.do_picture_description = False
        prompt = "Describe what you can see in the image. Do not make anything up that is not in the image. Respond with three sentences."
        self.pipeline_options.picture_description_options = smolvlm_picture_description
        self.pipeline_options.picture_description_options.prompt = prompt
        self.pipeline_options.picture_description_options.generation_config = {
            "max_new_tokens": 500,
            "do_sample": False,
        }

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options
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
            markdown_pages, extracted_images, page_images = self._process_document(
                input_data["pdf_content"]
            )

            # Structure the output
            output = {
                "markdown_pages": markdown_pages,
                "extracted_images": extracted_images,
                "page_images": page_images
            }
            outputs.append(output)

        # Return formatted output
        return construct_custom_output(request, outputs)

    def _process_document(self, base64_content: str) -> Tuple[List[str], List[str], List[str]]:
        """Internal method to process a single document"""
        # Decode base64 content to bytes
        pdf_content = base64.b64decode(base64_content)

        # Create BytesIO object and DocumentStream
        pdf_stream = io.BytesIO(pdf_content)
        source = DocumentStream(name="doc.pdf", stream=pdf_stream)

        # Convert using DocumentStream
        result = self.converter.convert(source)

        page_images = []
        extracted_images = []
        markdown_pages = []
        picture_descriptions = {}

        # Get page images from the conversion result
        for page_no, page in result.document.pages.items():
            if hasattr(page, 'image') and page.image is not None:
                page_images.append(str(page.image.uri))

        # Collect images and descriptions
        for element, _level in result.document.iterate_items():
            if isinstance(element, PictureItem):
                page_no = element.prov[0].page_no

                if hasattr(element, 'image') and element.image is not None:
                    extracted_images.append(str(element.image.uri))

                if page_no not in picture_descriptions:
                    picture_descriptions[page_no] = []

                if element.annotations:
                    ann = element.annotations[0]
                    desc = f"**AI-Generated Image Description:** {ann.text}\n<!-- end image description -->"
                    picture_descriptions[page_no].append(desc)

        # Process markdown pages
        for i in range(result.document.num_pages()):
            page_no = i + 1
            page_md = result.document.export_to_markdown(page_no=page_no)

            if page_no in picture_descriptions:
                parts = page_md.split("<!-- image -->")
                new_page_md = parts[0]

                for idx, part in enumerate(parts[1:]):
                    if idx < len(picture_descriptions[page_no]):
                        description = picture_descriptions[page_no][idx]
                        new_page_md += f"<!-- image -->\n{description}\n{part}"
                    else:
                        new_page_md += f"<!-- image -->{part}"

                page_md = new_page_md

            markdown_pages.append(page_md)

        return markdown_pages, extracted_images, page_images


entrypoint = InstillDeployable(DocumentProcessor).get_deployment_handle()
