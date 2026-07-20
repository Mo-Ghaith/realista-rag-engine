"""Stage 6: retrieve relevant context and assign stable citation labels."""

from __future__ import annotations

import importlib
import re
from typing import Any


vector_stage = importlib.import_module("04_vector_representation")
TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
COMMENT_TERMS = {
    "comment",
    "comments",
    "social",
    "buyer",
    "buyers",
    "objection",
    "objections",
    "intent",
    "sentiment",
    "delivery",
    "payment",
    "price",
    "pricing",
}
FACT_PACK_TERMS = {
    "aggregate",
    "aggregates",
    "count",
    "counts",
    "distribution",
    "distributions",
    "fact",
    "facts",
    "factpack",
    "pack",
    "packs",
    "summary",
    "summarize",
    "overview",
}


def retrieve_context(collection: Any, question: str, top_k: int = 4) -> list[dict[str, object]]:
    question = str(question or "").strip()
    if not question:
        return []
    available = int(collection.count())
    if available == 0:
        return []
    query_terms = set(TOKEN_PATTERN.findall(question.casefold()))
    wants_fact_pack = bool(query_terms & FACT_PACK_TERMS)
    wants_comment_evidence = bool(query_terms & COMMENT_TERMS) and not wants_fact_pack
    candidate_count = (
        available
        if wants_comment_evidence or wants_fact_pack
        else min(max(1, top_k * 8), available)
    )
    result = collection.query(
        query_embeddings=[vector_stage.embed_query(question)],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]
    retrieved = [
        {
            "citation": "",
            "chunk_id": ids[index - 1],
            "text": text,
            "distance": float(distances[index - 1]),
            **(metadatas[index - 1] or {}),
        }
        for index, text in enumerate(documents, start=1)
    ]
    ranked = _rerank_for_realista(question, retrieved)[: min(max(1, top_k), available)]
    for index, item in enumerate(ranked, start=1):
        item["citation"] = f"S{index}"
    return ranked


def format_context(retrieved: list[dict[str, object]]) -> str:
    return "\n\n".join(
        f"[{item['citation']}] {item['source_name']}\n{item['text']}" for item in retrieved
    )


def _rerank_for_realista(
    question: str, retrieved: list[dict[str, object]]
) -> list[dict[str, object]]:
    query_terms = set(TOKEN_PATTERN.findall(question.casefold()))
    wants_fact_pack = bool(query_terms & FACT_PACK_TERMS)
    wants_comment_evidence = bool(query_terms & COMMENT_TERMS) and not wants_fact_pack

    def score(item: dict[str, object]) -> tuple[float, float]:
        text = f"{item.get('source_name', '')} {item.get('text', '')}".casefold()
        text_terms = set(TOKEN_PATTERN.findall(text))
        lexical_overlap = len(query_terms & text_terms)
        source_name = str(item.get("source_name", "")).casefold()
        evidence_boost = 0.0
        if wants_fact_pack and "fact pack" in source_name:
            evidence_boost += 10.0
        if wants_comment_evidence and (
            "classified comment" in source_name
            or "evidence capsule social_comment" in source_name
        ):
            evidence_boost += 8.0
        if "comment id:" in text:
            evidence_boost += 2.0
        if wants_comment_evidence and "annotation_guidelines" in source_name:
            evidence_boost -= 2.0
        distance = float(item.get("distance", 1.0))
        return (evidence_boost + lexical_overlap, -distance)

    return sorted(retrieved, key=score, reverse=True)
