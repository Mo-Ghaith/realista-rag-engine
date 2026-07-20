"""Build compact Realista RAG exports for Streamlit deployment.

The full Realista workspace can be larger than the assignment deployment.
This script copies the evidence that the RAG app needs into local JSONL files:

- `data/processed/evidence_capsules.jsonl`
- `data/processed/fact_packs.jsonl`

The exports deliberately preserve trust and review fields. Review-required
comments can be cited as evidence of model output, but not as final market truth.
"""

from __future__ import annotations

import ast
from collections import Counter
import csv
import json
from pathlib import Path


APP_DIRECTORY = Path(__file__).resolve().parent
REALISTA_ROOT = APP_DIRECTORY.parent
PARENT_PROCESSED = REALISTA_ROOT / "data" / "processed"
LOCAL_PROCESSED = APP_DIRECTORY / "data" / "processed"


def main() -> None:
    LOCAL_PROCESSED.mkdir(parents=True, exist_ok=True)
    rows = _load_classified_comments()
    capsules = [_row_to_capsule(row) for row in rows]
    fact_packs = _build_fact_packs(rows)

    _write_jsonl(LOCAL_PROCESSED / "evidence_capsules.jsonl", capsules)
    _write_jsonl(LOCAL_PROCESSED / "fact_packs.jsonl", fact_packs)
    print(
        "Wrote "
        f"{len(capsules)} evidence capsules and {len(fact_packs)} fact packs "
        f"to {LOCAL_PROCESSED}"
    )


def _load_classified_comments() -> list[dict[str, str]]:
    csv_path = PARENT_PROCESSED / "classified_comments.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing classified comments: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_to_capsule(row: dict[str, str]) -> dict:
    comment_id = row.get("comment_id") or "unknown_comment"
    return {
        "capsule_id": f"social_comment:{comment_id}",
        "evidence_type": "classified_social_comment",
        "comment_id": comment_id,
        "post_id": row.get("post_id") or "",
        "source": row.get("source") or "classified_comments",
        "text": row.get("original_text") or "",
        "cleaned_text": row.get("cleaned_text") or "",
        "labels": {
            "sentiment": row.get("sentiment") or "unclear",
            "sentiment_confidence": _float(row.get("sentiment_confidence")),
            "intent": _as_list(row.get("intent")),
            "objection_type": _as_list(row.get("objection_type")),
            "buyer_stage": row.get("buyer_stage") or "unclear",
        },
        "trust": {
            "annotation_trust_status": row.get("annotation_trust_status") or "unknown",
            "needs_review": _bool(row.get("needs_review")),
            "label_source": row.get("label_source") or "unknown",
            "committee_policy_version": row.get("committee_policy_version") or "",
            "decision_reason": row.get("decision_reason")
            or row.get("uncertainty_reason")
            or "",
        },
        "quality": {
            "is_duplicate": _bool(row.get("is_duplicate")),
            "duplicate_group_id": row.get("duplicate_group_id") or "",
            "is_broker_spam": _bool(row.get("is_broker_spam")),
            "spam_score": _int(row.get("spam_score")),
            "spam_reasons": _as_list(row.get("spam_reasons")),
        },
        "timestamps": {
            "source_created_at": row.get("source_created_at") or "",
            "labeling_batch_id": row.get("labeling_batch_id") or "",
            "updated_at": row.get("updated_at") or "",
        },
        "usage_note": (
            "Use as traceable classified-comment evidence. If trust status is "
            "review_required, describe it as model/committee output awaiting "
            "human review, not as final market truth."
        ),
    }


def _build_fact_packs(rows: list[dict[str, str]]) -> list[dict]:
    non_spam = [row for row in rows if not _bool(row.get("is_broker_spam"))]
    review_required = [
        row for row in non_spam if str(row.get("annotation_trust_status")) == "review_required"
    ]
    packs = [
        {
            "fact_pack_id": "social_comment_overview_all",
            "evidence_type": "aggregate_social_comment_labels",
            "scope": "all classified Realista comments in local processed export",
            "row_count": len(rows),
            "non_spam_count": len(non_spam),
            "review_required_count": len(review_required),
            "duplicate_count": sum(1 for row in rows if _bool(row.get("is_duplicate"))),
            "sentiment_counts": _count_scalar(non_spam, "sentiment"),
            "intent_counts": _count_list(non_spam, "intent"),
            "objection_counts": _count_list(non_spam, "objection_type"),
            "buyer_stage_counts": _count_scalar(non_spam, "buyer_stage"),
            "evidence_comment_ids": [
                row.get("comment_id") for row in non_spam if row.get("comment_id")
            ][:100],
            "limitations": [
                "This pack summarizes classified social comments, not a representative market survey.",
                "Review-required labels are model/committee outputs awaiting human validation.",
                "Do not infer transaction prices, sales velocity, ROI, or population-level demand from this pack.",
            ],
        }
    ]

    for field, pack_id in [
        ("intent", "social_comment_intent_examples"),
        ("objection_type", "social_comment_objection_examples"),
    ]:
        examples: dict[str, list[str]] = {}
        for row in non_spam:
            for label in _as_list(row.get(field)):
                if label in {"", "unclear", "none"}:
                    continue
                examples.setdefault(label, [])
                if len(examples[label]) < 5 and row.get("comment_id"):
                    examples[label].append(row["comment_id"])
        packs.append(
            {
                "fact_pack_id": pack_id,
                "evidence_type": "label_to_comment_examples",
                "label_field": field,
                "examples_by_label": dict(sorted(examples.items())),
                "limitations": [
                    "Example IDs support traceability only; they are not statistically sampled.",
                    "Open the matching evidence capsule for the original comment and trust status.",
                ],
            }
        )
    return packs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _count_scalar(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field) or "unclear") for row in rows)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _count_list(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(label for label in _as_list(row.get(field)) if label)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


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


def _bool(value) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
