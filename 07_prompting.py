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
