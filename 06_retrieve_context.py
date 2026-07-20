"""Stage 6: retrieve relevant context and assign stable citation labels."""

from __future__ import annotations

import importlib
from typing import Any


vector_stage = importlib.import_module("04_vector_representation")


def retrieve_context(collection: Any, question: str, top_k: int = 4) -> list[dict[str, object]]:
    question = str(question or "").strip()
    if not question:
        return []
    available = int(collection.count())
    if available == 0:
        return []
    result = collection.query(
        query_embeddings=[vector_stage.embed_query(question)],
        n_results=min(max(1, top_k), available),
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]
    return [
        {
            "citation": f"S{index}",
            "chunk_id": ids[index - 1],
            "text": text,
            "distance": float(distances[index - 1]),
            **(metadatas[index - 1] or {}),
        }
        for index, text in enumerate(documents, start=1)
    ]


def format_context(retrieved: list[dict[str, object]]) -> str:
    return "\n\n".join(
        f"[{item['citation']}] {item['source_name']}\n{item['text']}" for item in retrieved
    )
