"""Stage 3: split preprocessed documents into overlapping, traceable chunks."""

from __future__ import annotations

import hashlib
import importlib


preprocessing_stage = importlib.import_module("02_preprocessing")


def chunk_documents(
    documents: list[dict[str, str]],
    chunk_size: int = 90,
    overlap: int = 20,
) -> list[dict[str, object]]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Use chunk_size > 0 and 0 <= overlap < chunk_size.")

    chunks: list[dict[str, object]] = []
    step = chunk_size - overlap
    for document in documents:
        words = str(document.get("text", "")).split()
        starts = [0] if document.get("atomic_document") else range(0, len(words), step)
        for start in starts:
            chunk_words = words if document.get("atomic_document") else words[start : start + chunk_size]
            if not chunk_words:
                continue
            text = " ".join(chunk_words)
            identity = f"{document['document_id']}:{start}:{text}"
            chunks.append(
                {
                    "chunk_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                    "document_id": document["document_id"],
                    "source_name": document["source_name"],
                    "source_url": document.get("source_url", ""),
                    "document_type": document.get("document_type", "text"),
                    "entity_type": document.get("entity_type", ""),
                    "entity_name": document.get("entity_name", ""),
                    "start_word": start,
                    "text": text,
                }
            )
            if document.get("atomic_document") or start + chunk_size >= len(words):
                break
    return chunks


if __name__ == "__main__":
    docs = preprocessing_stage.preprocess_documents(
        preprocessing_stage.documents_stage.load_documents()
    )
    print(f"Created {len(chunk_documents(docs))} chunks.")
