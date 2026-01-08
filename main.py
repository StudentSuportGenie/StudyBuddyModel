import os
import sys
import fitz
import requests
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from google import genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union

# ------------------ ENV SETUP ------------------

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API")

if not GOOGLE_API_KEY:
    print("❌ Error: Google API key not found")
    sys.exit(1)

# ✅ CREATE CLIENT ONCE (IMPORTANT)
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

# ------------------ FASTAPI APP ------------------

app = FastAPI(title="StudyBuddy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ REQUEST MODELS ------------------

class PDFRequest(BaseModel):
    url: Union[str, List[str]]
    useremail: str

    def __init__(self, **data):
        super().__init__(**data)
        if isinstance(self.url, str):
            self.url = [self.url]

class QARequest(BaseModel):
    useremail: str
    text: str
    question: str

# ------------------ HELPERS ------------------

def extract_text_from_pdf_url(url: str) -> str:
    try:
        if "dropbox.com" in url and "dl=0" in url:
            url = url.replace("dl=0", "dl=1")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        temp_file = "temp.pdf"
        with open(temp_file, "wb") as f:
            f.write(response.content)

        doc = fitz.open(temp_file)
        text = ""

        for page in doc:
            text += page.get_text("text")

        doc.close()
        os.remove(temp_file)

        if not text.strip():
            raise RuntimeError("PDF has no extractable text")

        return text

    except Exception as e:
        raise RuntimeError(f"PDF Error: {e}")

def store_text_to_chroma(text: str, useremail: str) -> int:
    chunks = [c.strip() for c in text.split(". ") if len(c.strip()) > 40]

    if not chunks:
        raise RuntimeError("No valid text chunks created")

    user_db_path = os.path.join(
        "AnswerDB",
        useremail.replace("@", "_").replace(".", "_")
    )
    os.makedirs(user_db_path, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=user_db_path,
        embedding_function=embeddings
    )

    vector_store.add_texts(chunks)
    return len(chunks)

# ------------------ ROUTES ------------------

@app.post("/PdFChoose")
async def create_vector_db(request: PDFRequest):
    try:
        total_chunks = 0

        for url in request.url:
            text = extract_text_from_pdf_url(url)
            total_chunks += store_text_to_chroma(text, request.useremail)

        return {
            "message": "PDFs processed successfully",
            "chunks_stored": total_chunks,
            "useremail": request.useremail
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/GetAnswer")
async def giveanswersusingPDF(request: QARequest):
    try:
        user_db_path = os.path.join(
            "AnswerDB",
            request.useremail.replace("@", "_").replace(".", "_")
        )

        if not os.path.exists(user_db_path):
            raise HTTPException(
                status_code=400,
                detail="No PDFs uploaded for this user"
            )

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = Chroma(
            persist_directory=user_db_path,
            embedding_function=embeddings
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(request.question)

        if not docs:
            return {
                "answer": "No relevant information found.",
                "useremail": request.useremail
            }

        context = "\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are an intelligent assistant.

Context:
{context}

User notes:
{request.text}

Question:
{request.question}

Give a clear answer.
"""

        # ✅ USE EXISTING CLIENT (NO RE-CREATION)
        response = genai_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        return {
            "answer": response.text,
            "useremail": request.useremail
        }

    except Exception as e:
        print("❌ GetAnswer Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ HEALTH CHECK ------------------

@app.get("/")
async def home():
    return {"status": "StudyBuddy API running 🚀"}
