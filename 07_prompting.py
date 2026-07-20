"""Stage 7: answer only from retrieved context and expose source citations."""

from __future__ import annotations

import importlib
import json
import os
import re
from urllib import error, request


retrieval_stage = importlib.import_module("06_retrieve_context")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_prompt(question: str, retrieved: list[dict[str, object]]) -> str:
    context = retrieval_stage.format_context(retrieved)
    return f"""You are an evidence-bounded assistant.
Answer the question using only the retrieved context below.
If the context is insufficient, say so plainly.
Cite every factual statement with one or more labels such as [S1].
Do not invent facts, sources, prices, or statistics.

Question: {question}

Retrieved context:
{context}
""".strip()


def answer_question(
    question: str,
    retrieved: list[dict[str, object]],
    timeout_seconds: int = 45,
) -> dict[str, object]:
    if not retrieved:
        return {
            "answer": "I do not have enough retrieved context to answer that question.",
            "sources": [],
            "used_retrieved_context": False,
            "mode": "insufficient_context",
        }

    prompt = build_prompt(question, retrieved)
    if OPENROUTER_API_KEY:
        answer = _call_openrouter(prompt, timeout_seconds)
        mode = "openrouter"
    else:
        answer = _extractive_fallback(question, retrieved)
        mode = "local_extractive_fallback"

    sources = [
        {
            "citation": item["citation"],
            "source_name": item["source_name"],
            "source_url": item.get("source_url", ""),
            "chunk_id": item["chunk_id"],
        }
        for item in retrieved
    ]
    return {
        "answer": answer,
        "sources": sources,
        "used_retrieved_context": True,
        "mode": mode,
    }


def _call_openrouter(prompt: str, timeout_seconds: int) -> str:
    payload = json.dumps(
        {
            "model": OPENROUTER_MODEL,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    http_request = request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
    return str(data["choices"][0]["message"]["content"]).strip()


def _extractive_fallback(
    question: str, retrieved: list[dict[str, object]], max_sentences: int = 3
) -> str:
    """Produce a key-free answer from retrieved sentences, with citations."""

    fact_pack_summary = _summarize_fact_pack_evidence(retrieved)
    if fact_pack_summary:
        return fact_pack_summary

    comment_summary = _summarize_comment_evidence(retrieved)
    if comment_summary:
        return comment_summary

    query_terms = set(re.findall(r"[\w\u0600-\u06ff]+", question.lower()))
    candidates: list[tuple[int, int, str, str]] = []
    for rank, item in enumerate(retrieved):
        for sentence in re.split(r"(?<=[.!?؟])\s+", str(item["text"])):
            terms = set(re.findall(r"[\w\u0600-\u06ff]+", sentence.lower()))
            score = len(query_terms & terms)
            candidates.append((score, -rank, sentence.strip(), str(item["citation"])))
    chosen = sorted(candidates, reverse=True)[:max_sentences]
    statements = [f"{sentence} [{citation}]" for _, _, sentence, citation in chosen if sentence]
    return " ".join(statements) or "The retrieved context is insufficient for a specific answer."


def _summarize_comment_evidence(retrieved: list[dict[str, object]]) -> str:
    comment_items = [
        item
        for item in retrieved
        if str(item.get("source_name", "")).startswith("Classified comment")
        or str(item.get("source_name", "")).startswith("Evidence capsule social_comment")
    ]
    if not comment_items:
        return ""

    statements = []
    for item in comment_items[:4]:
        text = str(item.get("text", ""))
        comment_id = _field(text, "Comment ID") or str(item.get("source_name", "comment"))
        intent = _field(text, "Intent labels") or "unclear"
        objection = _field(text, "Objection labels") or "unclear"
        stage = _field(text, "Buyer stage") or "unclear"
        trust = _field(text, "Trust status") or "unknown"
        statements.append(
            f"{comment_id} is labelled with intent `{intent}`, objection `{objection}`, "
            f"buyer stage `{stage}`, and trust status `{trust}` [{item['citation']}]."
        )
    return " ".join(statements)


def _summarize_fact_pack_evidence(retrieved: list[dict[str, object]]) -> str:
    fact_items = [
        item for item in retrieved if str(item.get("source_name", "")).startswith("Fact pack")
    ]
    if not fact_items:
        return ""
    if not str(retrieved[0].get("source_name", "")).startswith("Fact pack") and len(fact_items) < 2:
        return ""

    grouped: list[dict[str, object]] = []
    by_source: dict[str, dict[str, object]] = {}
    for item in fact_items:
        source_name = str(item.get("source_name", ""))
        if source_name not in by_source:
            by_source[source_name] = {**item, "text": str(item.get("text", ""))}
            grouped.append(by_source[source_name])
        else:
            by_source[source_name]["text"] = (
                str(by_source[source_name].get("text", ""))
                + " "
                + str(item.get("text", ""))
            )

    statements = []
    for item in grouped[:2]:
        text = str(item.get("text", ""))
        fact_pack_id = _field(text, "Fact pack ID") or str(item.get("source_name", "fact pack"))
        row_count = _field(text, "Row Count")
        non_spam_count = _field(text, "Non Spam Count")
        review_required_count = _field(text, "Review Required Count")
        sentiment_counts = _field(text, "Sentiment Counts")
        intent_counts = _field(text, "Intent Counts")
        objection_counts = _field(text, "Objection Counts")
        limitations = _field(text, "Limitations")
        parts = [f"{fact_pack_id}"]
        if row_count:
            parts.append(f"covers {row_count} rows")
        if non_spam_count:
            parts.append(f"{non_spam_count} non-spam comments")
        if review_required_count:
            parts.append(f"{review_required_count} review-required labels")
        if sentiment_counts:
            parts.append(f"sentiment counts {sentiment_counts}")
        if intent_counts:
            parts.append(f"intent counts {intent_counts}")
        if objection_counts:
            parts.append(f"objection counts {objection_counts}")
        if limitations:
            parts.append(f"limitations {limitations}")
        statements.append("; ".join(parts) + f" [{item['citation']}].")
    return " ".join(statements)


def _field(text: str, label: str) -> str:
    labels = [
        "Comment ID",
        "Post ID",
        "Evidence type",
        "Fact pack ID",
        "Scope",
        "Row Count",
        "Non Spam Count",
        "Review Required Count",
        "Duplicate Count",
        "Sentiment Counts",
        "Intent Counts",
        "Objection Counts",
        "Buyer Stage Counts",
        "Evidence Comment Ids",
        "Examples By Label",
        "Limitations",
        "Source",
        "Sentiment",
        "Intent labels",
        "Objection labels",
        "Buyer stage",
        "Trust status",
        "Needs human review",
        "Duplicate group",
        "Broker spam",
        "Spam reasons",
        "Decision reason",
        "Usage note",
        "Original comment",
        "Cleaned comment",
        "Uncertainty",
        "Instruction",
    ]
    next_labels = "|".join(re.escape(item) for item in labels if item != label)
    match = re.search(
        rf"{re.escape(label)}:\s*(.*?)(?=\s+(?:{next_labels}):|\Z)",
        text,
        re.S,
    )
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""
