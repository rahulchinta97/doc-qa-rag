"""
Query ChromaDB for the most relevant chunks given a user question.
"""
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
TOP_K = 5


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Return the top-k most relevant chunks for the query.
    Each item: {"text": str, "source": str, "chunk": int, "distance": float}
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )
    except Exception:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "source": meta.get("source", "unknown"),
            "chunk": meta.get("chunk", 0),
            "distance": round(dist, 4),
        })

    return chunks
