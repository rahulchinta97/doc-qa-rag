"""
Streamlit UI for Document Q&A.
Run with: streamlit run app.py
"""
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.ingest import ingest_pdf, list_documents
from src.qa import answer

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
st.title("📄 Document Q&A")
st.caption("Upload PDFs and ask questions about them using Claude.")

# ── Sidebar: upload & ingestion ─────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Documents")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        if st.button("Ingest PDF"):
            with st.spinner("Ingesting…"):
                # Save to a temp file so pypdf can read it
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf", dir="uploads"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    n = ingest_pdf(tmp_path)
                    # Rename to the original filename in uploads/
                    dest = os.path.join("uploads", uploaded_file.name)
                    os.replace(tmp_path, dest)
                    st.success(f"Ingested {n} chunks from **{uploaded_file.name}**")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    st.divider()
    st.subheader("Loaded Documents")
    docs = list_documents()
    if docs:
        for d in docs:
            st.write(f"• {d}")
    else:
        st.info("No documents ingested yet.")

# ── Main: Q&A ────────────────────────────────────────────────────────────────
os.makedirs("uploads", exist_ok=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Chat history display
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

# Input
question = st.chat_input("Ask a question about your documents…")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = answer(question)
        st.markdown(result["answer"])
        if result["sources"]:
            st.caption(f"Sources: {', '.join(result['sources'])}")

        with st.expander("Retrieved chunks"):
            for i, chunk in enumerate(result["chunks"], 1):
                st.markdown(f"**[{i}] {chunk['source']}** (dist: {chunk['distance']})")
                st.text(chunk["text"][:400] + "…" if len(chunk["text"]) > 400 else chunk["text"])

    st.session_state.history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
