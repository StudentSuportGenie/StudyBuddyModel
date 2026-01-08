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

    return {
        "message": "PDFs processed successfully",
        "chunks_stored": total_chunks
    }


@router.post("/GetAnswer")
async def answer(request: QARequest):
    db = load_user_vector_db(request.useremail)

    # ✅ FIX: use invoke() instead of get_relevant_documents()
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(request.question)

    context = "\n".join(doc.page_content for doc in docs)

    answer = get_answer(context, request.text, request.question)

    return {"answer": answer}


@router.get("/")
async def home():
    return {"status": "StudyBuddy API is running 🚀"}
