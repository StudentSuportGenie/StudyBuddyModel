import os
import sys
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API")

if not GOOGLE_API_KEY:
    print("Error: Google API key not found")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class PDFRequest(BaseModel):
    url: str
    useremail: str

# Extract text from a PDF URL
def extract_text_from_pdf_url(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open("temp.pdf", "wb") as f:
            f.write(response.content)

        doc = fitz.open("temp.pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        os.remove("temp.pdf")  # Clean up
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to fetch or process PDF: {e}")

# Store extracted text into Chroma DB for the given user
def store_text_to_chroma(text: str, useremail: str) -> int:
    chunks = [chunk.strip() for chunk in text.split(". ") if chunk.strip()]
    user_db_path = os.path.join("AnswerDB", useremail.replace("@", "_").replace(".", "_"))
    os.makedirs(user_db_path, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=user_db_path, embedding_function=embeddings)
    vector_store.add_texts(chunks)

    # No need to call vector_store.persist() — it saves automatically
    return len(chunks)

# Endpoint to process PDF and store embeddings
@app.post("/PdFChoose")
async def create_vector_db(request: PDFRequest):
    try:
        pdf_text = extract_text_from_pdf_url(request.url)
        chunks_stored = store_text_to_chroma(pdf_text, request.useremail)
        return {
            "message": "PDF processed and stored in ChromaDB.",
            "chunks_stored": chunks_stored,
            "useremail": request.useremail
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
