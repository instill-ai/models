import asyncio
import aiofiles
import base64
from pathlib import Path


async def convert_file(input_path: str, output_format: str) -> Path:
    """
    Convert a file to the specified format using LibreOffice.

    Args:
        input_path: Path to input file
        output_format: Target format (e.g., 'docx', 'pptx')

    Returns:
        Path to converted file
    """
    print(f"Converting {input_path} to {output_format}")
    input_path = Path(input_path)
    output_dir = input_path.parent

    command = [
        "soffice",
        "--headless",
        "--convert-to",
        output_format,
        str(input_path),
        "--outdir",
        str(output_dir),
    ]

    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    _, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Conversion failed: {stderr.decode()}")

    return output_dir / f"{input_path.stem}.{output_format}"


async def get_base64_content(file_path: str) -> str:
    """
    Convert file to base64 with proper MIME type prefix.

    Args:
        file_path: Path to the file

    Returns:
        Base64 encoded string with MIME type prefix
    """
    file_path = Path(file_path)
    # mime_type, _ = mimetypes.guess_type(str(file_path))

    async with aiofiles.open(file_path, "rb") as file:
        encoded_string = base64.b64encode(await file.read()).decode("utf-8")

    return encoded_string


async def process_document(base64_string: str) -> str:
    """
    Process document file, converting if necessary, and return base64 content.

    Args:
        file_path: Path to the document file

    Returns:
        Base64 encoded content with MIME type prefix
    """
    prefix, base64_content = base64_string.split(",")

    # Define format mappings
    format_conversions = {
        "data:application/vnd.ms-powerpoint;base64": "pptx",
        "data:application/msword;base64": "docx",
    }

    # Check if conversion is needed
    if prefix in format_conversions:
        if format_conversions[prefix] == "pptx":
            file_path = "temp.ppt"
        elif format_conversions[prefix] == "docx":
            file_path = "temp.doc"
    else:
        return base64_content

    try:
        decoded_bytes = base64.b64decode(base64_content)
    except base64.binascii.Error as e:
        raise ValueError(f"Failed to decode base64 string: {e}")

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(decoded_bytes)

    if prefix in format_conversions:
        try:
            # Convert the file
            converted_path = await convert_file(file_path, format_conversions[prefix])
            # Get base64 of converted file
            base64_string = await get_base64_content(converted_path)
            # Clean up converted file
            converted_path.unlink()
        except Exception as e:
            raise RuntimeError(f"Failed to process document: {str(e)}")

    return base64_string
