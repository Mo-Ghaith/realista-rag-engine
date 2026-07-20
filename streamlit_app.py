"""Streamlit interface for the complete course RAG pipeline."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import streamlit as st


APP_DIRECTORY = Path(__file__).resolve().parent
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))


documents_stage = importlib.import_module("01_documents")
store_stage = importlib.import_module("05_create_chroma_store")
retrieval_stage = importlib.import_module("06_retrieve_context")
rag = importlib.import_module("07_prompting")

try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass


st.set_page_config(page_title="Realista RAG Engine", page_icon="🔎")
st.title("Realista RAG Engine")
st.caption("Documents → preprocessing → chunks → vectors → Chroma → retrieval → answer")

uploaded_files = st.file_uploader(
    "Optional: add UTF-8 text documents",
    type=["txt", "md"],
    accept_multiple_files=True,
)
uploads = []
for uploaded in uploaded_files:
    try:
        uploads.append((uploaded.name, uploaded.getvalue().decode("utf-8")))
    except UnicodeDecodeError:
        st.warning(f"Skipped {uploaded.name}: the file is not valid UTF-8.")

documents = documents_stage.load_documents()
documents.extend(documents_stage.documents_from_uploads(uploads))
_, collection = store_stage.build_store_from_documents(documents)

question = st.text_input("Ask a question about the indexed documents")
top_k = st.slider("Retrieved chunks", min_value=1, max_value=6, value=3)

if st.button("Retrieve and answer", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        retrieved = retrieval_stage.retrieve_context(collection, question, top_k=top_k)
        try:
            result = rag.answer_question(question, retrieved)
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.subheader("Answer")
            st.write(result["answer"])
            st.caption(f"Mode: {result['mode']} | Retrieved context used: {result['used_retrieved_context']}")
            st.subheader("Sources")
            for source in result["sources"]:
                st.markdown(
                    f"- **[{source['citation']}] {source['source_name']}** "
                    f"— chunk `{source['chunk_id']}`"
                )
            with st.expander("Retrieved chunks"):
                for item in retrieved:
                    st.markdown(f"**[{item['citation']}] {item['source_name']}**")
                    st.write(item["text"])
