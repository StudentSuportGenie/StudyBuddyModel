import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Global cache to load the embedding model only once
_embeddings_instance = None

def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings_instance


def store_text_to_chroma(text: str, useremail: str) -> int:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    

    chunks = text_splitter.split_text(text)
    chunks = [c.strip() for c in chunks if c.strip()]

    user_db_path = os.path.join(
        "AnswerDB", useremail.replace("@", "_").replace(".", "_"))
    os.makedirs(user_db_path, exist_ok=True)

    db = Chroma(persist_directory=user_db_path, embedding_function=get_embeddings())
    db.add_texts(chunks)

    return len(chunks)


def load_user_vector_db(useremail: str):
    user_db_path = os.path.join(
        "AnswerDB", useremail.replace("@", "_").replace(".", "_"))
    return Chroma(persist_directory=user_db_path, embedding_function=get_embeddings())

