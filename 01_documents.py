"""Stage 1: provide source-labelled Realista documents for the RAG pipeline."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Iterable


APP_DIRECTORY = Path(__file__).resolve().parent
REALISTA_ROOT = APP_DIRECTORY.parent
LOCAL_PROCESSED_DATA = APP_DIRECTORY / "data" / "processed"
PARENT_PROCESSED_DATA = REALISTA_ROOT / "data" / "processed"

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


REALISTA_TEXT_SOURCES = [
    REALISTA_ROOT / "PROJECT_SOURCE_OF_TRUTH.md",
    REALISTA_ROOT / "docs" / "annotation_guidelines_v2.md",
    REALISTA_ROOT / "docs" / "execution" / "DATA_LINEAGE.md",
    REALISTA_ROOT / "docs" / "execution" / "SECURITY_AND_PRIVACY.md",
]


def load_documents(paths: Iterable[str | Path] | None = None) -> list[dict[str, str]]:
    """Return Realista evidence documents plus optional UTF-8 text files.

    The assignment version remains self-contained, but when it sits inside the
    full Realista repository it also indexes the real workflow artifacts:
    project rules, lineage/security docs, evidence capsules, and classified
    comments with comment IDs and model labels.
    """

    documents = [dict(document) for document in DEFAULT_DOCUMENTS]
    documents.extend(load_realista_source_documents())
    documents.extend(load_realista_evidence_documents())
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


def load_realista_source_documents() -> list[dict[str, str]]:
    """Load canonical Realista docs when the parent repository is available."""

    documents: list[dict[str, str]] = []
    for path in REALISTA_TEXT_SOURCES:
        if not path.exists() or not path.is_file():
            continue
        documents.append(
            {
                "document_id": f"realista_doc_{path.stem}",
                "source_name": f"Realista document: {path.name}",
                "source_url": _path_uri(path),
                "text": path.read_text(encoding="utf-8"),
            }
        )
    return documents


def load_realista_evidence_documents() -> list[dict[str, str]]:
    """Load report-ready evidence capsules or classified comments.

    Evidence capsules are preferred because they are already shaped for RAG.
    If the capsule export is empty, classified comments are converted into
    evidence records so citations can still point to concrete comment IDs.
    """

    capsules = _first_non_empty(
        _load_evidence_capsules(path)
        for path in _candidate_processed_paths("evidence_capsules.jsonl")
    )
    if capsules:
        return capsules + load_realista_fact_pack_documents() + load_realista_market_documents()

    for csv_path in _candidate_processed_paths("classified_comments.csv"):
        if csv_path.exists():
            return (
                _load_classified_comments(csv_path)
                + load_realista_fact_pack_documents()
                + load_realista_market_documents()
            )

    for jsonl_path in _candidate_processed_paths("classified_comments.jsonl"):
        if jsonl_path.exists():
            return (
                _load_classified_comment_jsonl(jsonl_path)
                + load_realista_fact_pack_documents()
                + load_realista_market_documents()
            )

    return []


def load_realista_fact_pack_documents() -> list[dict[str, str]]:
    """Load deterministic aggregate fact packs for narrative grounding."""

    documents: list[dict[str, str]] = []
    for path in _candidate_processed_paths("fact_packs.jsonl"):
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                fact_pack_id = str(row.get("fact_pack_id") or f"fact_pack_{line_number}")
                documents.append(
                    {
                        "document_id": f"fact_pack_{fact_pack_id}",
                        "source_name": f"Fact pack {fact_pack_id}",
                        "source_url": f"realista://fact_packs/{fact_pack_id}",
                        "text": _fact_pack_text(row),
                    }
                )
        if documents:
            return documents
    return documents


def load_realista_market_documents() -> list[dict[str, str]]:
    """Load Realista market fact packs for Egyptian real-estate QA."""

    documents: list[dict[str, str]] = []
    for path in _candidate_processed_paths("market_facts.jsonl"):
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                pack_id = str(row.get("fact_pack_id") or f"market_pack_{line_number}")
                documents.append(
                    {
                        "document_id": f"market_fact_{pack_id}",
                        "source_name": f"Market fact pack {pack_id}",
                        "source_url": f"realista://market_facts/{pack_id}",
                        "text": _market_fact_text(row),
                    }
                )
        if documents:
            return documents
    return documents


def _load_evidence_capsules(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    documents: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            comment_id = str(row.get("comment_id") or f"capsule_{line_number}")
            capsule_id = str(row.get("capsule_id") or comment_id)
            documents.append(
                {
                    "document_id": f"evidence_capsule_{capsule_id}",
                    "source_name": f"Evidence capsule {capsule_id}",
                    "source_url": f"realista://evidence_capsules/{capsule_id}",
                    "text": _capsule_text(row),
                }
            )
    return documents


def _load_classified_comments(path: Path) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            comment_id = str(row.get("comment_id") or "").strip()
            if not comment_id:
                continue
            documents.append(_classified_comment_document(row, comment_id))
    return documents


def _load_classified_comment_jsonl(path: Path) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            comment_id = str(row.get("comment_id") or "").strip()
            if not comment_id:
                continue
            documents.append(_classified_comment_document(row, comment_id))
    return documents


def _classified_comment_document(row: dict, comment_id: str) -> dict[str, str]:
    source = str(row.get("source") or "classified_comments")
    text = "\n".join(
        item
        for item in [
            f"Comment ID: {comment_id}",
            f"Post ID: {row.get('post_id') or 'unavailable'}",
            f"Source: {source}",
            f"Sentiment: {row.get('sentiment') or 'unclear'}"
            f" (confidence {row.get('sentiment_confidence') or 'unavailable'})",
            f"Intent labels: {', '.join(_as_list(row.get('intent'))) or 'unclear'}",
            "Objection labels: "
            f"{', '.join(_as_list(row.get('objection_type'))) or 'unclear'}",
            f"Buyer stage: {row.get('buyer_stage') or 'unclear'}",
            f"Trust status: {row.get('annotation_trust_status') or 'unknown'}",
            f"Needs human review: {row.get('needs_review') or 'unknown'}",
            f"Duplicate group: {row.get('duplicate_group_id') or 'unavailable'}",
            f"Broker spam: {row.get('is_broker_spam') or 'False'}",
            f"Spam reasons: {', '.join(_as_list(row.get('spam_reasons'))) or 'none'}",
            f"Decision reason: {row.get('decision_reason') or row.get('uncertainty_reason') or ''}",
            f"Original comment: {row.get('original_text') or ''}",
            f"Cleaned comment: {row.get('cleaned_text') or ''}",
        ]
        if str(item).strip()
    )
    return {
        "document_id": f"classified_comment_{comment_id}",
        "source_name": f"Classified comment {comment_id}",
        "source_url": f"realista://classified_comments/{comment_id}",
        "text": text,
    }


def _capsule_text(row: dict) -> str:
    labels = row.get("soft_labels") or {}
    labels = labels or row.get("labels") or {}
    sentiment = labels.get("sentiment") or {}
    if not isinstance(sentiment, dict):
        sentiment = {
            "value": labels.get("sentiment", "unclear"),
            "confidence": labels.get("sentiment_confidence", "unavailable"),
        }
    intent = labels.get("intent") or {}
    if not isinstance(intent, dict):
        intent = {"value": labels.get("intent") or []}
    objection = labels.get("objection_type") or {}
    if not isinstance(objection, dict):
        objection = {"value": labels.get("objection_type") or []}
    stage = labels.get("buyer_stage") or {}
    if not isinstance(stage, dict):
        stage = {"value": labels.get("buyer_stage", "unclear")}
    trust = row.get("trust") or {}
    quality = row.get("quality") or {}
    return "\n".join(
        [
            f"Comment ID: {row.get('comment_id')}",
            f"Post ID: {row.get('post_id') or ''}",
            f"Evidence type: {row.get('evidence_type') or 'classified_social_comment'}",
            f"Sentiment: {sentiment.get('value', 'unclear')}"
            f" (confidence {sentiment.get('confidence', 'unavailable')})",
            f"Intent labels: {', '.join(_as_list(intent.get('value'))) or 'unclear'}",
            "Objection labels: "
            f"{', '.join(_as_list(objection.get('value'))) or 'unclear'}",
            f"Buyer stage: {stage.get('value', 'unclear')}",
            "Trust status: "
            f"{trust.get('annotation_trust_status') or row.get('annotation_trust_status') or 'unknown'}",
            f"Needs human review: {trust.get('needs_review', row.get('needs_review'))}",
            f"Duplicate group: {quality.get('duplicate_group_id') or ''}",
            f"Broker spam: {quality.get('is_broker_spam', False)}",
            f"Spam reasons: {', '.join(_as_list(quality.get('spam_reasons'))) or 'none'}",
            "Uncertainty: "
            f"{trust.get('decision_reason') or row.get('uncertainty_reason') or ''}",
            f"Usage note: {row.get('usage_note') or ''}",
            f"Instruction: {row.get('instruction') or ''}",
            f"Original comment: {row.get('text') or ''}",
            f"Cleaned comment: {row.get('cleaned_text') or ''}",
        ]
    )


def _fact_pack_text(row: dict) -> str:
    lines = [
        f"Fact pack ID: {row.get('fact_pack_id')}",
        f"Evidence type: {row.get('evidence_type')}",
        f"Scope: {row.get('scope') or row.get('label_field') or ''}",
    ]
    for key in [
        "row_count",
        "non_spam_count",
        "review_required_count",
        "duplicate_count",
        "sentiment_counts",
        "intent_counts",
        "objection_counts",
        "buyer_stage_counts",
        "examples_by_label",
        "limitations",
    ]:
        if key in row:
            lines.append(f"{key.replace('_', ' ').title()}: {row[key]}")
    if "evidence_comment_ids" in row:
        lines.append(f"Evidence Comment Ids: {row['evidence_comment_ids'][:10]}")
    return "\n".join(lines)


def _market_fact_text(row: dict) -> str:
    lines = [
        f"Market Fact Pack ID: {row.get('fact_pack_id')}",
        f"Evidence type: {row.get('evidence_type')}",
        f"Scope: {row.get('scope') or ''}",
    ]
    for key in [
        "location",
        "developer",
        "unit_type",
        "record_count",
        "location_count",
        "developer_count",
        "project_count",
        "developers",
        "projects",
        "locations",
        "unit_type_counts",
        "price_egp",
        "top_locations_by_observations",
        "top_developers_by_observations",
    ]:
        if key in row:
            lines.append(f"{key.replace('_', ' ').title()}: {row[key]}")
    if "limitations" in row:
        lines.append(f"Limitations: {str(row['limitations'][0]) if row['limitations'] else ''}")
    return "\n".join(lines)


def _candidate_processed_paths(filename: str) -> list[Path]:
    return [LOCAL_PROCESSED_DATA / filename, PARENT_PROCESSED_DATA / filename]


def _first_non_empty(groups: Iterable[list[dict[str, str]]]) -> list[dict[str, str]]:
    for group in groups:
        if group:
            return group
    return []


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
    return [text]


def _path_uri(path: Path) -> str:
    try:
        return path.resolve().as_uri()
    except ValueError:
        return str(path)


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
