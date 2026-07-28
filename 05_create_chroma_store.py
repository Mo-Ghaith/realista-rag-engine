"""Stage 5: create and populate a Chroma vector store."""

from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
from typing import Any
import warnings


vector_stage = importlib.import_module("04_vector_representation")
COLLECTION_NAME = "rag_assignment_documents"


class InMemoryVectorCollection:
    """Small Chroma-compatible fallback for restricted deployment runtimes."""

    def __init__(self, vectorized_chunks: list[dict[str, object]]) -> None:
        self._rows = [dict(chunk) for chunk in vectorized_chunks]

    def count(self) -> int:
        return len(self._rows)

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str] | None = None,
    ) -> dict[str, list[list[Any]]]:
        del include
        query = query_embeddings[0] if query_embeddings else []
        ranked = sorted(
            self._rows,
            key=lambda row: _cosine_distance(
                query,
                list(row.get("embedding") or []),
            ),
        )[: max(0, min(n_results, len(self._rows)))]
        return {
            "ids": [[str(row["chunk_id"]) for row in ranked]],
            "documents": [[str(row["text"]) for row in ranked]],
            "metadatas": [
                [
                    {
                        "document_id": str(row["document_id"]),
                        "source_name": str(row["source_name"]),
                        "source_url": str(row.get("source_url", "")),
                        "document_type": str(row.get("document_type", "text")),
                        "entity_type": str(row.get("entity_type", "")),
                        "entity_name": str(row.get("entity_name", "")),
                        "start_word": int(row["start_word"]),
                    }
                    for row in ranked
                ]
            ],
            "distances": [
                [
                    _cosine_distance(query, list(row.get("embedding") or []))
                    for row in ranked
                ]
            ],
        }


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 1.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 1.0
    return max(0.0, min(2.0, 1.0 - dot / (left_norm * right_norm)))


def create_chroma_store(
    vectorized_chunks: list[dict[str, object]],
    persist_directory: str | Path | None = None,
    collection_name: str = COLLECTION_NAME,
) -> tuple[Any, Any]:
    import chromadb

    client = (
        chromadb.PersistentClient(path=str(persist_directory))
        if persist_directory
        else chromadb.Client()
    )
    existing_names = {
        item.name if hasattr(item, "name") else str(item)
        for item in client.list_collections()
    }
    if collection_name in existing_names:
        client.delete_collection(collection_name)
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "description": "Realista RAG evidence chunks"},
        embedding_function=None,
    )
    if vectorized_chunks:
        collection.upsert(
            ids=[str(chunk["chunk_id"]) for chunk in vectorized_chunks],
            embeddings=[chunk["embedding"] for chunk in vectorized_chunks],
            documents=[str(chunk["text"]) for chunk in vectorized_chunks],
            metadatas=[
                {
                    "document_id": str(chunk["document_id"]),
                    "source_name": str(chunk["source_name"]),
                    "source_url": str(chunk.get("source_url", "")),
                    "document_type": str(chunk.get("document_type", "text")),
                    "entity_type": str(chunk.get("entity_type", "")),
                    "entity_name": str(chunk.get("entity_name", "")),
                    "start_word": int(chunk["start_word"]),
                }
                for chunk in vectorized_chunks
            ],
        )
    return client, collection


def build_store_from_documents(
    documents: list[dict[str, str]],
    persist_directory: str | Path | None = None,
    chunk_size: int = 90,
    overlap: int = 20,
    collection_name: str | None = None,
) -> tuple[Any, Any]:
    preprocessing = importlib.import_module("02_preprocessing")
    chunking = importlib.import_module("03_chunking")
    processed = preprocessing.preprocess_documents(documents)
    chunks = chunking.chunk_documents(processed, chunk_size=chunk_size, overlap=overlap)
    if collection_name is None:
        identity = "|".join(
            f"{document.get('document_id')}:{len(str(document.get('text', '')))}"
            for document in documents
        )
        digest = hashlib.sha256(
            f"{chunk_size}:{overlap}:{identity}".encode("utf-8")
        ).hexdigest()[:16]
        collection_name = f"{COLLECTION_NAME}_{digest}"
    vectorized = vector_stage.vectorize_chunks(chunks)
    try:
        return create_chroma_store(
            vectorized,
            persist_directory,
            collection_name=collection_name,
        )
    except Exception as exc:
        warnings.warn(
            "Chroma is unavailable in this runtime; using the compatible "
            f"in-memory vector fallback ({type(exc).__name__}: {exc}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, InMemoryVectorCollection(vectorized)


if __name__ == "__main__":
    documents = importlib.import_module("01_documents").load_documents()
    _, store = build_store_from_documents(documents, ".chroma")
    print(f"Chroma collection contains {store.count()} chunks.")
