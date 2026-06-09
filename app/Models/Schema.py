from pydantic import BaseModel
from typing import List, Union

class PDFRequest(BaseModel):
    url: Union[List[str], str]
    useremail: str

class QARequest(BaseModel):
    useremail: str
    text: str
    question: str


class ImageRequest(BaseModel):
    image_url: str
    useremail: str
