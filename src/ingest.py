"""
Ingest PDFs: parse → chunk → embed → store in ChromaDB.
"""
import os
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )


def ingest_pdf(pdf_path: str) -> int:
    """Parse a PDF, chunk it, and upsert into ChromaDB. Returns chunk count."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1. Extract text from all pages
    reader = PdfReader(str(path))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    if not full_text.strip():
        raise ValueError("Could not extract any text from the PDF.")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(full_text)

    # 3. Store in ChromaDB (upsert so re-ingesting is safe)
    collection = _get_collection()
    doc_id = path.stem  # filename without extension as document ID prefix

    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": path.name, "chunk": i} for i in range(len(chunks))]

    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

    return len(chunks)


def list_documents() -> list[str]:
    """Return unique source filenames currently in the collection."""
    collection = _get_collection()
    results = collection.get(include=["metadatas"])
    sources = {m["source"] for m in results["metadatas"]} if results["metadatas"] else set()
    return sorted(sources)
