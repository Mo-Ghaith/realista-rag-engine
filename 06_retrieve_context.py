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
MARKET_TERMS = {
    "area",
    "areas",
    "developer",
    "developers",
    "location",
    "locations",
    "market",
    "mean",
    "average",
    "median",
    "price",
    "prices",
    "pricing",
    "project",
    "projects",
    "apartment",
    "apartments",
    "villa",
    "villas",
    "townhouse",
    "townhouses",
    "chalet",
    "chalets",
    "new",
    "cairo",
    "zayed",
    "october",
    "sahel",
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
    requested_units = query_terms & {
        "apartment",
        "apartments",
        "villa",
        "villas",
        "townhouse",
        "townhouses",
        "chalet",
        "chalets",
        "office",
        "retail",
        "penthouse",
        "duplex",
    }
    wants_comment_evidence = (
        bool(query_terms & COMMENT_TERMS)
        and not wants_fact_pack
        and not (query_terms & {"developer", "developers", "mean", "average", "median"})
    )
    wants_market_evidence = bool(query_terms & MARKET_TERMS) and not wants_comment_evidence
    candidate_count = (
        available
        if wants_comment_evidence or wants_fact_pack or wants_market_evidence
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
    requested_units = query_terms & {
        "apartment",
        "apartments",
        "villa",
        "villas",
        "townhouse",
        "townhouses",
        "chalet",
        "chalets",
        "office",
        "retail",
        "penthouse",
        "duplex",
    }
    wants_comment_evidence = (
        bool(query_terms & COMMENT_TERMS)
        and not wants_fact_pack
        and not (query_terms & {"developer", "developers", "mean", "average", "median"})
    )
    wants_market_evidence = bool(query_terms & MARKET_TERMS) and not wants_comment_evidence
    wants_developer_list = bool(query_terms & {"developer", "developers", "who"}) and not requested_units

    def score(item: dict[str, object]) -> tuple[float, float]:
        text = f"{item.get('source_name', '')} {item.get('text', '')}".casefold()
        text_terms = set(TOKEN_PATTERN.findall(text))
        lexical_overlap = len(query_terms & text_terms)
        source_name = str(item.get("source_name", "")).casefold()
        evidence_boost = 0.0
        if wants_market_evidence and "market fact pack" in source_name:
            evidence_boost += 12.0
            entity_type = str(item.get("entity_type", "")).casefold()
            entity_name = str(item.get("entity_name", "")).casefold()
            if wants_developer_list:
                evidence_boost += 5.0 if "unit type:" not in text else -4.0
                if entity_type == "location":
                    evidence_boost += 24.0
                elif entity_type in {"developer", "project"}:
                    evidence_boost -= 4.0
            if entity_name and entity_name in question.casefold():
                evidence_boost += 18.0
            for unit in requested_units:
                singular = unit[:-1] if unit.endswith("s") else unit
                if f"unit type: {singular}" in text:
                    evidence_boost += 8.0
                elif "unit type:" in text:
                    evidence_boost -= 2.0
        if wants_fact_pack and "fact pack" in source_name:
            evidence_boost += 10.0
        if wants_comment_evidence and (
            "classified comment" in source_name
            or "evidence capsule social_comment" in source_name
        ):
            evidence_boost += 8.0
            if "price" in query_terms and (
                "price_question" in text or "objection labels: price" in text
            ):
                evidence_boost += 6.0
            if "payment" in query_terms and (
                "payment_question" in text or "payment_plan" in text
            ):
                evidence_boost += 6.0
            if "delivery" in query_terms and (
                "delivery_question" in text or "delivery_time" in text
            ):
                evidence_boost += 6.0
            if "location" in query_terms and (
                "location_question" in text or "objection labels: location" in text
            ):
                evidence_boost += 6.0
            if "objection labels: none" in text and query_terms & {"price", "payment", "delivery", "location"}:
                evidence_boost -= 3.0
        if "comment id:" in text:
            evidence_boost += 2.0
        if wants_comment_evidence and "annotation_guidelines" in source_name:
            evidence_boost -= 2.0
        distance = float(item.get("distance", 1.0))
        return (evidence_boost + lexical_overlap, -distance)

    return sorted(retrieved, key=score, reverse=True)
