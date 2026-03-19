"""
Build the prompt from retrieved chunks and call Claude for an answer.
"""
import os
import anthropic
from src.retriever import retrieve

MODEL = "claude-opus-4-6"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document excerpts.
- Answer only from the context below. If the answer is not in the context, say so clearly.
- Cite the source filename when referencing information.
- Be concise and factual."""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] Source: {chunk['source']}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def answer(question: str, top_k: int = 5) -> dict:
    """
    Retrieve relevant chunks and generate an answer with Claude.
    Returns {"answer": str, "sources": list[str], "chunks": list[dict]}
    """
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "No documents have been ingested yet. Please upload a PDF first.",
            "sources": [],
            "chunks": [],
        }

    context = build_context(chunks)
    user_message = f"""Context:\n{context}\n\nQuestion: {question}"""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = response.content[0].text
    sources = list({chunk["source"] for chunk in chunks})

    return {
        "answer": answer_text,
        "sources": sources,
        "chunks": chunks,
    }
