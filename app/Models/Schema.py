from pydantic import BaseModel
from typing import List

class PDFRequest(BaseModel):
    url: List[str]
    useremail: str

class QARequest(BaseModel):
    useremail: str
    text: str
    question: str


class ImageRequest(BaseModel):
    image_url: str
    useremail: str
