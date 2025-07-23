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
from typing import List



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

# Request models
class PDFRequest(BaseModel):
    url: List[str]
    useremail: str

class QARequest(BaseModel):
    useremail: str
    text: str
    question: str

# Extract text from PDF URL
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
        os.remove("temp.pdf")
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to fetch or process PDF from {url}: {e}")

# Store text in ChromaDB
def store_text_to_chroma(text: str, useremail: str) -> int:
    chunks = [chunk.strip() for chunk in text.split(". ") if chunk.strip()]
    user_db_path = os.path.join("AnswerDB", useremail.replace("@", "_").replace(".", "_"))
    os.makedirs(user_db_path, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=user_db_path, embedding_function=embeddings)
    vector_store.add_texts(chunks)
    return len(chunks)

# Route: Upload PDFs and store embeddings
@app.post("/PdFChoose")
async def create_vector_db(request: PDFRequest):
    try:
        total_chunks = 0
        for url in request.url:
            pdf_text = extract_text_from_pdf_url(url)
            chunks_stored = store_text_to_chroma(pdf_text, request.useremail)
            total_chunks += chunks_stored

        return {
            "message": f"{len(request.url)} PDF(s) processed and stored in ChromaDB.",
            "chunks_stored": total_chunks,
            "useremail": request.useremail
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route: Get answer from stored ChromaDB and Gemini
@app.post("/GetAnswer")
async def giveanswersusingPDF(request: QARequest):
    try:
        user_db_path = os.path.join("AnswerDB", request.useremail.replace("@", "_").replace(".", "_"))

        # Load embeddings and Chroma vector DB
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory=user_db_path, embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(request.question)
        context = "\n".join(doc.page_content for doc in docs)

        # Gemini model
        model = genai.GenerativeModel("gemma-3n-e2b-it")

        prompt = f"""
You are an assistant. Use the following extracted text and user input to answer the question clearly and accurately.

--- Extracted Text from PDF ---
{context}

--- Additional Input Text ---
{request.text}

--- Question ---
{request.question}

Please provide a detailed and relevant answer:
"""

        response = model.generate_content(prompt)
        return {
            "answer": response.text,
            "useremail": request.useremail
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
