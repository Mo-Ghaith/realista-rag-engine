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
import re
from statistics import mean, median


APP_DIRECTORY = Path(__file__).resolve().parent
REALISTA_ROOT = APP_DIRECTORY.parent
PARENT_PROCESSED = REALISTA_ROOT / "data" / "processed"
LOCAL_PROCESSED = APP_DIRECTORY / "data" / "processed"


def main() -> None:
    LOCAL_PROCESSED.mkdir(parents=True, exist_ok=True)
    rows = _load_classified_comments()
    capsules = [_row_to_capsule(row) for row in rows]
    fact_packs = _build_fact_packs(rows)
    market_facts = _build_market_facts()

    _write_jsonl(LOCAL_PROCESSED / "evidence_capsules.jsonl", capsules)
    _write_jsonl(LOCAL_PROCESSED / "fact_packs.jsonl", fact_packs)
    _write_jsonl(LOCAL_PROCESSED / "market_facts.jsonl", market_facts)
    print(
        "Wrote "
        f"{len(capsules)} evidence capsules, {len(fact_packs)} fact packs, "
        f"and {len(market_facts)} market fact packs "
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


def _build_market_facts() -> list[dict]:
    source_path = REALISTA_ROOT / "data" / "ingested" / "nawy_english_property500_controlled_20260718.json"
    if not source_path.exists():
        return []
    data = json.loads(source_path.read_text(encoding="utf-8"))
    records = _extract_market_records(data.get("items", []))
    packs: list[dict] = [_market_overview(records, source_path)]

    by_location: dict[str, list[dict]] = {}
    by_location_unit: dict[tuple[str, str], list[dict]] = {}
    by_developer: dict[str, list[dict]] = {}
    for record in records:
        by_location.setdefault(record["location"], []).append(record)
        by_location_unit.setdefault(
            (record["location"], record["unit_type"].casefold()), []
        ).append(record)
        if record["developer"] != "unknown":
            by_developer.setdefault(record["developer"], []).append(record)

    for location, rows in sorted(by_location.items()):
        packs.append(_location_pack(location, rows))
    for (location, unit_type), rows in sorted(by_location_unit.items()):
        if len(rows) >= 3:
            packs.append(_location_unit_pack(location, unit_type, rows))
    for developer, rows in sorted(by_developer.items()):
        if len(rows) >= 3:
            packs.append(_developer_pack(developer, rows))
    return packs


def _extract_market_records(items: list[dict]) -> list[dict]:
    records: list[dict] = []
    seen = set()
    for item in items:
        if item.get("item_type") != "property":
            continue
        version = (item.get("versions") or {}).get("en") or {}
        fields = version.get("fields") or {}
        facts = fields.get("typed_market_facts") or []
        title = str(item.get("title") or version.get("title") or "")
        title_parts = _infer_title_parts(title)
        for fact in facts:
            if fact.get("fact_type") != "unit_total_price":
                continue
            if fact.get("market_trust_status") != "valid":
                continue
            price = _float(fact.get("total_price_egp"))
            if price <= 0:
                continue
            unit_type = _clean_name(fact.get("unit_type") or title_parts.get("unit_type"))
            location = _clean_name(fact.get("area_name") or title_parts.get("location"))
            developer = _clean_name(fact.get("developer_name") or title_parts.get("developer"))
            project = _clean_name(fact.get("project_name") or title_parts.get("project"))
            if location.casefold() in {"area", "unknown", ""}:
                location = title_parts.get("location", "unknown")
            location = _canonical_location(location)
            if developer.casefold() in {"developer", "developers", "unknown", ""}:
                developer = title_parts.get("developer", "unknown")
            if project.casefold() in {"compound", "project", "unknown", ""}:
                project = title_parts.get("project", title[:80] or "unknown")
            key = (item.get("canonical_url"), unit_type, location, developer, project, price)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "property_url": item.get("canonical_url") or fact.get("source_url") or "",
                    "title": title,
                    "location": location,
                    "developer": developer,
                    "project": project,
                    "unit_type": unit_type or "unknown",
                    "total_price_egp": price,
                    "area_sqm": _float(fact.get("area_sqm")) or None,
                    "source": "nawy",
                    "trust_status": fact.get("market_trust_status") or "valid",
                }
            )
    return records


def _infer_title_parts(title: str) -> dict[str, str]:
    unit_types = [
        "Apartment",
        "Villa",
        "Townhouse",
        "Twinhouse",
        "Chalet",
        "Penthouse",
        "Duplex",
        "Studio",
        "Office",
        "Retail",
        "Medical",
        "Loft",
        "Cabin",
        "Family House",
    ]
    unit_type = next((kind for kind in unit_types if re.search(rf"\b{re.escape(kind)}\b", title, re.I)), "unknown")
    project = "unknown"
    location = "unknown"
    developer = "unknown"
    match = re.search(
        r"for sale in (?P<project>.+?)(?:\s+-\s+|\s+with\s+.*?\s+in\s+)(?P<location>.+?)\s+by\s+(?P<developer>.+?)(?:\.|$)",
        title,
        re.I,
    )
    if match:
        project = match.group("project")
        location = match.group("location")
        developer = match.group("developer")
    else:
        prefix = re.search(r"^(?:Delivery In \d{4}\s+)?(?P<location>.+?)\s+(?:Apartment|Villa|Townhouse|Twinhouse|Chalet|Penthouse|Duplex|Studio|Office|Retail|Medical|Loft|Cabin)", title, re.I)
        if prefix:
            location = prefix.group("location").split(",")[-1].strip()
        by_match = re.search(r"\bby\s+(?P<developer>[^.]+?)(?:\.|$)", title, re.I)
        if by_match:
            developer = by_match.group("developer")
        project_match = re.search(r"\b(?:Apartment|Villa|Townhouse|Twinhouse|Chalet|Penthouse|Duplex|Studio|Office|Retail|Medical|Loft|Cabin),\s*(?P<project>.+?)\s+\d+\s+Beds", title, re.I)
        if project_match:
            project = project_match.group("project")
    return {
        "unit_type": _clean_name(unit_type),
        "project": _clean_name(project),
        "location": _clean_name(location),
        "developer": _clean_name(developer),
    }


def _market_overview(records: list[dict], source_path: Path) -> dict:
    return {
        "fact_pack_id": "market_overview_nawy_property500",
        "evidence_type": "market_listing_aggregate",
        "scope": "validated Nawy property observations in controlled property500 crawl",
        "source_file": source_path.name,
        "record_count": len(records),
        "location_count": len({row["location"] for row in records}),
        "developer_count": len({row["developer"] for row in records if row["developer"] != "unknown"}),
        "unit_type_counts": _counter(row["unit_type"] for row in records),
        "price_egp": _stats(row["total_price_egp"] for row in records),
        "top_locations_by_observations": _top(_counter(row["location"] for row in records), 10),
        "top_developers_by_observations": _top(_counter(row["developer"] for row in records if row["developer"] != "unknown"), 10),
        "limitations": _market_limitations(),
    }


def _location_pack(location: str, rows: list[dict]) -> dict:
    return {
        "fact_pack_id": f"market_location_{_slug(location)}",
        "evidence_type": "market_location_aggregate",
        "location": location,
        "record_count": len(rows),
        "developer_count": len({row["developer"] for row in rows if row["developer"] != "unknown"}),
        "project_count": len({row["project"] for row in rows if row["project"] != "unknown"}),
        "developers": sorted({row["developer"] for row in rows if row["developer"] != "unknown"})[:25],
        "projects": sorted({row["project"] for row in rows if row["project"] != "unknown"})[:25],
        "unit_type_counts": _counter(row["unit_type"] for row in rows),
        "price_egp": _stats(row["total_price_egp"] for row in rows),
        "example_urls": [row["property_url"] for row in rows if row["property_url"]][:5],
        "limitations": _market_limitations(),
    }


def _location_unit_pack(location: str, unit_type: str, rows: list[dict]) -> dict:
    return {
        "fact_pack_id": f"market_location_{_slug(location)}_{_slug(unit_type)}",
        "evidence_type": "market_location_unit_aggregate",
        "location": location,
        "unit_type": unit_type,
        "record_count": len(rows),
        "developer_count": len({row["developer"] for row in rows if row["developer"] != "unknown"}),
        "project_count": len({row["project"] for row in rows if row["project"] != "unknown"}),
        "developers": sorted({row["developer"] for row in rows if row["developer"] != "unknown"})[:20],
        "projects": sorted({row["project"] for row in rows if row["project"] != "unknown"})[:20],
        "price_egp": _stats(row["total_price_egp"] for row in rows),
        "example_urls": [row["property_url"] for row in rows if row["property_url"]][:5],
        "limitations": _market_limitations(),
    }


def _developer_pack(developer: str, rows: list[dict]) -> dict:
    return {
        "fact_pack_id": f"market_developer_{_slug(developer)}",
        "evidence_type": "market_developer_aggregate",
        "developer": developer,
        "record_count": len(rows),
        "locations": sorted({row["location"] for row in rows if row["location"] != "unknown"})[:25],
        "projects": sorted({row["project"] for row in rows if row["project"] != "unknown"})[:25],
        "unit_type_counts": _counter(row["unit_type"] for row in rows),
        "price_egp": _stats(row["total_price_egp"] for row in rows),
        "example_urls": [row["property_url"] for row in rows if row["property_url"]][:5],
        "limitations": _market_limitations(),
    }


def _stats(values) -> dict:
    clean = sorted(float(value) for value in values if _float(value) > 0)
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(clean),
        "min": round(clean[0], 2),
        "max": round(clean[-1], 2),
        "mean": round(mean(clean), 2),
        "median": round(median(clean), 2),
    }


def _counter(values) -> dict:
    return dict(sorted(Counter(values).items(), key=lambda item: (-item[1], str(item[0]))))


def _top(counts: dict, limit: int) -> dict:
    return dict(list(counts.items())[:limit])


def _clean_name(value) -> str:
    text = re.sub(r"\s+", " ", str(value or "unknown")).strip(" .,-")
    return text or "unknown"


def _canonical_location(value: str) -> str:
    text = _clean_name(value)
    lowered = text.casefold()
    canonical_markers = [
        ("new cairo", "New Cairo"),
        ("6th of october", "6th of October City"),
        ("6 october", "6th of October City"),
        ("october gardens", "October Gardens"),
        ("el sheikh zayed", "El Sheikh Zayed"),
        ("sheikh zayed", "El Sheikh Zayed"),
        ("new capital", "New Capital City"),
        ("mostakbal", "Mostakbal City"),
        ("north coast", "North Coast-Sahel"),
        ("ras el hekma", "Ras El Hekma"),
        ("ain sokhna", "Ain Sokhna"),
        ("al dabaa", "Al Dabaa"),
        ("al alamein", "Al Alamein"),
    ]
    for marker, canonical in canonical_markers:
        if marker in lowered:
            return canonical
    return text


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return text or "unknown"


def _market_limitations() -> list[str]:
    return [
        "Prices are asking/listed prices from Nawy observations, not verified transaction prices.",
        "Statistics describe the local crawl/export only and may not cover the whole Egyptian market.",
        "Use row counts and source URLs when judging reliability.",
    ]


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
