"""Stage 5: create and populate a Chroma vector store."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


vector_stage = importlib.import_module("04_vector_representation")
COLLECTION_NAME = "rag_assignment_documents"


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
    collection = client.get_or_create_collection(
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
    documents: list[dict[str, str]], persist_directory: str | Path | None = None
) -> tuple[Any, Any]:
    preprocessing = importlib.import_module("02_preprocessing")
    chunking = importlib.import_module("03_chunking")
    processed = preprocessing.preprocess_documents(documents)
    chunks = chunking.chunk_documents(processed)
    return create_chroma_store(vector_stage.vectorize_chunks(chunks), persist_directory)


if __name__ == "__main__":
    documents = importlib.import_module("01_documents").load_documents()
    _, store = build_store_from_documents(documents, ".chroma")
    print(f"Chroma collection contains {store.count()} chunks.")
