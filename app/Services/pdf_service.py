import fitz
import requests
import os
import tempfile
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


PDF_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"


def normalize_pdf_url(url: str) -> str:
    parsed = urlparse(url)
    if "dropbox.com" in parsed.netloc:
        query = parse_qs(parsed.query, keep_blank_values=True)
        if query.get("dl") == ["0"]:
            query["dl"] = ["1"]
        elif "dl" not in query:
            query["dl"] = ["1"]
        normalized_query = urlencode(query, doseq=True)
        parsed = parsed._replace(query=normalized_query)
    return urlunparse(parsed)


def is_pdf_content(content: bytes, content_type: str) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return content.lstrip().startswith(b"%PDF")


def extract_text_from_pdf_url(url: str) -> str:
    url = normalize_pdf_url(url)

    with requests.Session() as session:
        response = session.get(
            url,
            headers={
                "User-Agent": PDF_USER_AGENT,
                "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            },
            timeout=30,
            allow_redirects=True,
        )

    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if not is_pdf_content(response.content, content_type):
        if "dropbox.com" in url:
            raise ValueError(
                "Dropbox share link did not resolve to a PDF file. "
                "Use a direct download link or include the full Dropbox query parameters such as ?dl=1 and rlkey=... "
                "before uploading."
            )
        raise ValueError(
            f"URL did not return PDF content (content-type={content_type}). "
            "Please provide a direct PDF download URL."
        )

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
