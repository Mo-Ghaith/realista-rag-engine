"""Stage 1: provide raw, source-labelled documents for the RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


DEFAULT_DOCUMENTS = [
    {
        "document_id": "realista_scope",
        "source_name": "Realista Project Source of Truth - included excerpt",
        "source_url": "local://PROJECT_SOURCE_OF_TRUTH.md",
        "text": (
            "Realista is an Egyptian real-estate market-intelligence and project-planning system. "
            "It combines trustworthy listing observations, Egyptian Arabic social comments, and "
            "developer-supplied project scenarios. Client-visible conclusions must be backed by "
            "traceable evidence, and unsupported facts must be reported as unavailable."
        ),
    },
    {
        "document_id": "realista_rag_rules",
        "source_name": "Realista evidence-bounding rules - included excerpt",
        "source_url": "local://PROJECT_SOURCE_OF_TRUTH.md",
        "text": (
            "A language model may explain validated results, but it may not calculate authoritative "
            "statistics, invent missing market facts, infer transaction prices from asking prices, "
            "or present social commenters as a representative population. Every numerical claim "
            "must link to a fact pack and every social claim must link to supporting comment ids."
        ),
    },
    {
        "document_id": "student_rag_guide",
        "source_name": "Student RAG Project Instructions - included excerpt",
        "source_url": "local://Student RAG Project Instructions.pdf",
        "text": (
            "The required course pipeline is documents, preprocessing, chunking, vector "
            "representation, vector store, context retrieval, prompting, and a Streamlit UI. "
            "The final answer must use retrieved context and cite its sources. Real API keys must "
            "not be stored in Python files or uploaded environment files."
        ),
    },
]


def load_documents(paths: Iterable[str | Path] | None = None) -> list[dict[str, str]]:
    """Return built-in documents plus optional UTF-8 text files."""

    documents = [dict(document) for document in DEFAULT_DOCUMENTS]
    for item in paths or []:
        path = Path(item)
        documents.append(
            {
                "document_id": path.stem,
                "source_name": path.name,
                "source_url": path.resolve().as_uri(),
                "text": path.read_text(encoding="utf-8"),
            }
        )
    return documents


def documents_from_uploads(uploads: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
    """Create source-labelled documents from ``(filename, text)`` upload pairs."""

    return [
        {
            "document_id": f"upload_{index}",
            "source_name": filename,
            "source_url": f"upload://{filename}",
            "text": text,
        }
        for index, (filename, text) in enumerate(uploads, start=1)
        if text.strip()
    ]


if __name__ == "__main__":
    print(f"Loaded {len(load_documents())} raw documents.")
