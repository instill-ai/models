import base64
import mimetypes
import subprocess
from pathlib import Path


def convert_file(input_path: str, output_format: str) -> Path:
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

    try:
        subprocess.run(command, check=True, capture_output=True)
        return output_dir / f"{input_path.stem}.{output_format}"
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Conversion failed: {e.stderr.decode()}")


def get_base64_content(file_path: str) -> str:
    """
    Convert file to base64 with proper MIME type prefix.

    Args:
        file_path: Path to the file

    Returns:
        Base64 encoded string with MIME type prefix
    """
    file_path = Path(file_path)
    # mime_type, _ = mimetypes.guess_type(str(file_path))

    with open(file_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode("utf-8")

    return encoded_string


def process_document(base64_string: str) -> str:
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

    with open(file_path, "wb") as f:
        f.write(decoded_bytes)

    if prefix in format_conversions:
        try:
            # Convert the file
            converted_path = convert_file(file_path, format_conversions[prefix])
            # Get base64 of converted file
            base64_string = get_base64_content(converted_path)
            # Clean up converted file
            converted_path.unlink()
        except Exception as e:
            raise RuntimeError(f"Failed to process document: {str(e)}")

    return base64_string
