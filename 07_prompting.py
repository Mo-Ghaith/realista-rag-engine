"""Stage 7: answer only from retrieved context and expose source citations."""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
from urllib import error, request


retrieval_stage = importlib.import_module("06_retrieve_context")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)


def build_prompt(question: str, retrieved: list[dict[str, object]]) -> str:
    context = retrieval_stage.format_context(retrieved)
    return f"""You are Realista's evidence-bounded RAG assistant.
Answer the user's question using only the retrieved context below.
Write a concise, useful answer for a real-estate analyst.
Do not dump raw JSON, Python dictionaries, or source fields unless the user asks for raw records.
If the evidence is aggregate, summarize the main pattern, then mention limitations.
If the evidence is comment-level, mention the relevant comment IDs and labels.
Every factual claim must cite one or more labels such as [S1].
If the context is insufficient, say so plainly.
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
        answer = _local_rag_answer(question, retrieved)
        mode = "local_synthesized_rag"

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


def _local_rag_answer(question: str, retrieved: list[dict[str, object]]) -> str:
    market_summary = _summarize_market_evidence(question, retrieved)
    if market_summary:
        return market_summary

    fact_pack_summary = _summarize_fact_pack_evidence(retrieved)
    if fact_pack_summary:
        return fact_pack_summary

    comment_summary = _summarize_comment_evidence(question, retrieved)
    if comment_summary:
        return comment_summary

    return _extractive_answer(question, retrieved)


def _summarize_market_evidence(
    question: str, retrieved: list[dict[str, object]]
) -> str:
    market_items = [
        item
        for item in retrieved
        if str(item.get("source_name", "")).startswith("Market fact pack")
    ]
    if not market_items:
        return ""

    grouped = _merge_by_source(market_items)
    question_terms = set(TOKEN_PATTERN.findall(question.casefold()))
    wants_developers = bool(question_terms & {"developer", "developers", "who"})
    wants_price = bool(question_terms & {"mean", "average", "median", "price", "prices", "pricing"})

    best = _best_market_pack(grouped, question)
    text = str(best.get("text", ""))
    citation = str(best["citation"])
    pack_id = _field(text, "Market Fact Pack ID") or str(best.get("source_name", "market fact pack"))
    location = _field(text, "Location")
    developer = _field(text, "Developer")
    unit_type = _field(text, "Unit Type")
    record_count = _field(text, "Record Count")
    price_stats = _parse_literal(_field(text, "Price Egp"))
    developers = _as_list(_field(text, "Developers"))
    projects = _as_list(_field(text, "Projects"))
    locations = _as_list(_field(text, "Locations"))
    limitations = _as_list(_field(text, "Limitations"))

    subject = _market_subject(location, developer, unit_type, pack_id)
    lines: list[str] = []

    if wants_developers and developers:
        shown = ", ".join(developers[:10])
        lines.append(
            f"For {location or subject}, Realista's Nawy market evidence shows "
            f"{len(developers)} developer(s), including {shown} [{citation}]."
        )
        if projects:
            lines.append(f"Example projects include {', '.join(projects[:8])} [{citation}].")
    elif wants_price and isinstance(price_stats, dict):
        lines.append(
            f"For {subject}, the available Nawy listed-price evidence contains "
            f"{price_stats.get('count', record_count)} observations [{citation}]."
        )
        lines.append(
            "Listed price stats: "
            f"mean {_money(price_stats.get('mean'))}, "
            f"median {_money(price_stats.get('median'))}, "
            f"minimum {_money(price_stats.get('min'))}, "
            f"maximum {_money(price_stats.get('max'))} [{citation}]."
        )
        if developers:
            lines.append(f"Developers represented include {', '.join(developers[:8])} [{citation}].")
        if projects:
            lines.append(f"Projects represented include {', '.join(projects[:8])} [{citation}].")
    else:
        lines.append(
            f"Realista found a market fact pack for {subject} with "
            f"{record_count or 'available'} observations [{citation}]."
        )
        if isinstance(price_stats, dict) and price_stats.get("count"):
            lines.append(
                f"Price range is {_money(price_stats.get('min'))} to "
                f"{_money(price_stats.get('max'))}, with mean {_money(price_stats.get('mean'))} "
                f"and median {_money(price_stats.get('median'))} [{citation}]."
            )
        if developers:
            lines.append(f"Developers: {', '.join(developers[:10])} [{citation}].")
        if projects:
            lines.append(f"Projects: {', '.join(projects[:10])} [{citation}].")
        if locations:
            lines.append(f"Locations: {', '.join(locations[:10])} [{citation}].")

    if limitations:
        lines.append(f"Important limitation: {limitations[0]} [{citation}].")
    return "\n".join(lines)


def _extractive_answer(
    question: str, retrieved: list[dict[str, object]], max_sentences: int = 3
) -> str:
    query_terms = set(TOKEN_PATTERN.findall(question.lower()))
    candidates: list[tuple[int, int, str, str]] = []
    for rank, item in enumerate(retrieved):
        for sentence in re.split(r"(?<=[.!?\u061f])\s+", str(item["text"])):
            clean = sentence.strip()
            if not clean:
                continue
            terms = set(TOKEN_PATTERN.findall(clean.lower()))
            score = len(query_terms & terms)
            candidates.append((score, -rank, clean, str(item["citation"])))
    chosen = sorted(candidates, reverse=True)[:max_sentences]
    statements = [f"{sentence} [{citation}]" for _, _, sentence, citation in chosen]
    return " ".join(statements) or "The retrieved context is insufficient for a specific answer."


def _summarize_comment_evidence(
    question: str, retrieved: list[dict[str, object]]
) -> str:
    comment_items = [
        item
        for item in retrieved
        if str(item.get("source_name", "")).startswith("Classified comment")
        or str(item.get("source_name", "")).startswith("Evidence capsule social_comment")
    ]
    if not comment_items:
        return ""

    wanted = _wanted_comment_labels(question)
    focused_items = _filter_comment_items(comment_items, wanted)
    if focused_items:
        comment_items = focused_items

    lines = ["The retrieved comment evidence shows:"]
    for item in comment_items[:4]:
        text = str(item.get("text", ""))
        comment_id = _field(text, "Comment ID") or str(item.get("source_name", "comment"))
        sentiment = _field(text, "Sentiment") or "unclear"
        intent = _field(text, "Intent labels") or "unclear"
        objection = _field(text, "Objection labels") or "unclear"
        stage = _field(text, "Buyer stage") or "unclear"
        trust = _field(text, "Trust status") or "unknown"
        lines.append(
            f"- `{comment_id}` is sentiment `{sentiment}`, with intent `{intent}`, objection `{objection}`, "
            f"buyer stage `{stage}`, and trust status `{trust}` [{item['citation']}]."
        )
    lines.append("")
    lines.append(
        "These are traceable classified-comment signals, but they are not a representative "
        "market survey and should not be treated as final truth while marked `review_required`."
    )
    return "\n".join(lines)


def _summarize_fact_pack_evidence(retrieved: list[dict[str, object]]) -> str:
    fact_items = [
        item for item in retrieved if str(item.get("source_name", "")).startswith("Fact pack")
    ]
    if not fact_items:
        return ""
    if not str(retrieved[0].get("source_name", "")).startswith("Fact pack") and len(fact_items) < 2:
        return ""

    grouped = _merge_by_source(fact_items)
    overview = next(
        (
            item
            for item in grouped
            if "social_comment_overview_all" in str(item.get("source_name", ""))
        ),
        grouped[0],
    )

    text = str(overview.get("text", ""))
    citation = str(overview["citation"])
    fact_pack_id = _field(text, "Fact pack ID") or "social_comment_overview_all"
    row_count = _field(text, "Row Count") or "the available"
    non_spam_count = _field(text, "Non Spam Count") or "the available"
    review_required_count = _field(text, "Review Required Count") or "the available"

    lines = [
        f"`{fact_pack_id}` summarizes {row_count} classified social comments, "
        f"including {non_spam_count} non-spam comments. All {review_required_count} "
        f"labels are marked review-required, so the result is useful for exploration "
        f"but not final market truth [{citation}]."
    ]

    for label, field_name, limit in [
        ("Sentiment", "Sentiment Counts", None),
        ("Main intents", "Intent Counts", 5),
        ("Main objections", "Objection Counts", 5),
        ("Buyer stages", "Buyer Stage Counts", None),
    ]:
        formatted = _format_counts(_field(text, field_name), limit=limit)
        if formatted:
            lines.append(f"- {label}: {formatted} [{citation}].")

    limitations = _as_list(_field(text, "Limitations"))
    if limitations:
        lines.append(f"- Key limitation: {limitations[0]} [{citation}].")

    example_pack = next(
        (
            item
            for item in grouped
            if "examples by label" in str(item.get("text", "")).casefold()
        ),
        None,
    )
    if example_pack:
        examples = _short_examples(_field(str(example_pack.get("text", "")), "Examples By Label"))
        if examples:
            lines.append(f"- Example comment IDs: {examples} [{example_pack['citation']}].")

    return "\n".join(lines)


def _merge_by_source(items: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: list[dict[str, object]] = []
    by_source: dict[str, dict[str, object]] = {}
    for item in items:
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
    return grouped


def _best_market_pack(items: list[dict[str, object]], question: str) -> dict[str, object]:
    terms = set(TOKEN_PATTERN.findall(question.casefold()))
    requested_units = {
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
    } & terms
    wants_developers = bool(terms & {"developer", "developers", "who"})
    wants_price = bool(terms & {"mean", "average", "median", "price", "prices", "pricing"})

    def score(item: dict[str, object]) -> tuple[int, int]:
        text = str(item.get("source_name", "")) + " " + str(item.get("text", ""))
        text_terms = set(TOKEN_PATTERN.findall(text.casefold()))
        pack_id = _field(text, "Market Fact Pack ID").casefold()
        unit_type = _field(text, "Unit Type")
        specificity = 0
        if "market_location_" in pack_id and not unit_type:
            specificity += 6 if wants_developers and not requested_units else 2
        if "market_location_" in pack_id and unit_type:
            specificity += 3
            if wants_price and requested_units:
                specificity += 5
            if wants_developers and not requested_units:
                specificity -= 4
        if "market_developer_" in pack_id:
            specificity += 1
        return (specificity, len(terms & text_terms))

    return sorted(items, key=score, reverse=True)[0]


def _market_subject(location: str, developer: str, unit_type: str, pack_id: str) -> str:
    parts = []
    if unit_type:
        parts.append(unit_type)
    if location:
        parts.append(f"in {location}")
    if developer:
        parts.append(f"by {developer}")
    return " ".join(parts) if parts else pack_id


def _money(value) -> str:
    try:
        return f"{float(value):,.0f} EGP"
    except (TypeError, ValueError):
        return "unavailable"


def _wanted_comment_labels(question: str) -> set[str]:
    terms = set(TOKEN_PATTERN.findall(question.casefold()))
    wanted: set[str] = set()
    if terms & {"price", "pricing", "سعر", "السعر"}:
        wanted.update({"price_question", "price"})
    if terms & {"payment", "payments", "installment", "installments", "تقسيط", "قسط"}:
        wanted.update({"payment_question", "payment_plan"})
    if terms & {"delivery", "handover", "استلام"}:
        wanted.update({"delivery_question", "delivery_time"})
    if terms & {"location", "locations", "لوكيشن", "مكان"}:
        wanted.update({"location_question", "location"})
    if terms & {"trust", "developer", "reputation", "ثقة", "مطور"}:
        wanted.update({"trust", "developer_reputation"})
    return wanted


def _filter_comment_items(
    comment_items: list[dict[str, object]], wanted: set[str]
) -> list[dict[str, object]]:
    if not wanted:
        return []
    focused = []
    for item in comment_items:
        text = str(item.get("text", ""))
        labels = {
            label.strip()
            for label in (
                _field(text, "Intent labels") + "," + _field(text, "Objection labels")
            ).split(",")
        }
        if labels & wanted:
            focused.append(item)
    return focused


def _field(text: str, label: str) -> str:
    labels = [
        "Comment ID",
        "Post ID",
        "Evidence type",
        "Fact pack ID",
        "Market Fact Pack ID",
        "Scope",
        "Location",
        "Developer",
        "Unit Type",
        "Row Count",
        "Record Count",
        "Location Count",
        "Developer Count",
        "Project Count",
        "Non Spam Count",
        "Review Required Count",
        "Duplicate Count",
        "Developers",
        "Projects",
        "Locations",
        "Sentiment Counts",
        "Intent Counts",
        "Objection Counts",
        "Buyer Stage Counts",
        "Unit Type Counts",
        "Price Egp",
        "Top Locations By Observations",
        "Top Developers By Observations",
        "Example Urls",
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


def _format_counts(raw: str, limit: int | None = None) -> str:
    value = _parse_literal(raw)
    if not isinstance(value, dict):
        return raw
    items = sorted(value.items(), key=lambda item: (-int(item[1]), str(item[0])))
    if limit:
        items = items[:limit]
    return ", ".join(f"{key}: {count}" for key, count in items)


def _short_examples(raw: str, limit: int = 3) -> str:
    value = _parse_literal(raw)
    if not isinstance(value, dict):
        return ""
    parts = []
    for label, ids in sorted(value.items())[:limit]:
        if isinstance(ids, list):
            parts.append(f"{label}: {', '.join(str(item) for item in ids[:2])}")
    return "; ".join(parts)


def _parse_literal(raw: str):
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    parsed = _parse_literal(str(value or ""))
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []
