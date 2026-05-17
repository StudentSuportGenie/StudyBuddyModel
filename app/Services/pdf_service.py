import fitz
import requests
import os
import tempfile


def extract_text_from_pdf_url(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Use a proper temp file to avoid leaking files on error
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(response.content)

        doc = fitz.open(temp_path)
        text = "".join(page.get_text("text") for page in doc)
        doc.close()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return text
