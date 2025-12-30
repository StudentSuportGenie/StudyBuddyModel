import fitz
import requests
import os


def extract_text_from_pdf_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()

    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    doc = fitz.open("temp.pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    os.remove("temp.pdf")

    return text
