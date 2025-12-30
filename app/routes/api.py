from fastapi import APIRouter, HTTPException
from app.Models.Schema import PDFRequest, QARequest
from app.Services.pdf_service import extract_text_from_pdf_url
from app.Services.vector_service import store_text_to_chroma, load_user_vector_db
from app.Services.qa_service import get_answer

router = APIRouter()


@router.post("/PdFChoose")
async def create_vector_db(request: PDFRequest):
    total_chunks = 0
    for url in request.url:
        text = extract_text_from_pdf_url(url)
        total_chunks += store_text_to_chroma(text, request.useremail)

    return {"message": "PDFs processed", "chunks_stored": total_chunks}


@router.post("/GetAnswer")
async def answer(request: QARequest):
    db = load_user_vector_db(request.useremail)
    docs = db.as_retriever(
        search_kwargs={"k": 5}).get_relevant_documents(request.question)

    context = "\n".join(d.page_content for d in docs)
    answer = get_answer(context, request.text, request.question)

    return {"answer": answer}

@router.get("/")
async def home():
    return {"status": "StudyBuddy API is running 🚀"}
