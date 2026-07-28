"""Read-only audit of the Nawy Mongo source used by the standalone RAG app."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any


APP_DIRECTORY = Path(__file__).resolve().parent
WORKSPACE_ROOT = APP_DIRECTORY.parent


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(WORKSPACE_ROOT / ".env")


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _load_env()
    mongo_uri = os.getenv("MONGO_URI", "").strip()
    if not mongo_uri:
        raise SystemExit("MONGO_URI is not configured.")

    from pymongo import MongoClient

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    try:
        database = client[os.getenv("MONGO_DB", "Realista")]
        available = set(database.list_collection_names())
        collection_names = [
            name
            for name in ("listing_snapshots", "nawy_listing_snapshots")
            if name in available
        ]
        reports = [_audit_collection(database[name]) for name in collection_names]
        report = {
            "database": database.name,
            "snapshot_collections": reports,
            "document_collections": {
                name: _audit_document_collection(database[name])
                for name in ("nawy_items", "nawy_listings")
                if name in available
            },
            "rollup_counts": {
                name: database[name].count_documents({})
                for name in ("developers", "projects", "locations")
                if name in available
            },
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_value))
    finally:
        client.close()


def _audit_document_collection(collection: Any) -> dict[str, Any]:
    """Describe a source collection without printing bulky payload values."""

    sample = collection.find_one({}, {"_id": 0}) or {}
    nested_keys = {
        key: sorted(value.keys())
        for key, value in sample.items()
        if isinstance(value, dict)
    }
    return {
        "count": collection.count_documents({}),
        "sample_top_level_keys": sorted(sample.keys()),
        "sample_nested_keys": nested_keys,
    }


def _audit_collection(collection: Any) -> dict[str, Any]:
    """Return a compact coverage and field-population report."""

    projection = {
        "_id": 0,
        "crawl_batch_id": 1,
        "captured_at": 1,
        "market_trust_status": 1,
        "validation_status": 1,
        "entity_status": 1,
        "developer_id": 1,
        "project_id": 1,
        "location_id": 1,
        "unit_id": 1,
        "unit_type_id": 1,
        "total_price_egp": 1,
        "area_sqm": 1,
        "price_per_sqm_egp": 1,
        "bedrooms": 1,
        "bathrooms": 1,
        "delivery": 1,
        "availability_status": 1,
        "raw_fields": 1,
    }
    rows = list(collection.find({"source": "nawy"}, projection))
    batches: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "min": None, "max": None}
    )
    field_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    raw_field_counts: Counter[str] = Counter()

    for row in rows:
        batch = batches[str(row.get("crawl_batch_id") or "missing")]
        captured_at = row.get("captured_at")
        batch["count"] += 1
        if captured_at is not None:
            batch["min"] = (
                captured_at
                if batch["min"] is None or captured_at < batch["min"]
                else batch["min"]
            )
            batch["max"] = (
                captured_at
                if batch["max"] is None or captured_at > batch["max"]
                else batch["max"]
            )

        for field, value in row.items():
            if field != "raw_fields" and value not in (None, "", [], {}):
                field_counts[field] += 1
        for field, value in (row.get("raw_fields") or {}).items():
            if value not in (None, "", [], {}):
                raw_field_counts[field] += 1
        status_counts[
            "|".join(
                [
                    str(row.get("market_trust_status") or "missing"),
                    str(row.get("validation_status") or "missing"),
                    str(row.get("entity_status") or "missing"),
                ]
            )
        ] += 1

    ordered_batches = sorted(
        (
            {
                "crawl_batch_id": batch_id,
                **{key: _json_value(value) for key, value in summary.items()},
            }
            for batch_id, summary in batches.items()
        ),
        key=lambda item: str(item.get("max") or ""),
        reverse=True,
    )
    return {
        "collection": collection.name,
        "total_rows": len(rows),
        "valid_rows": status_counts.get("valid|valid|entity_resolved", 0),
        "capture_min": _json_value(
            min(
                (row.get("captured_at") for row in rows if row.get("captured_at")),
                default=None,
            )
        ),
        "capture_max": _json_value(
            max(
                (row.get("captured_at") for row in rows if row.get("captured_at")),
                default=None,
            )
        ),
        "batches": ordered_batches,
        "field_population": dict(field_counts.most_common()),
        "raw_field_population": dict(raw_field_counts.most_common()),
        "status_counts": dict(status_counts.most_common()),
        "sample_rows": [
            {
                key: _json_value(value)
                for key, value in row.items()
                if key != "raw_fields"
            }
            for row in rows[:3]
        ],
    }


if __name__ == "__main__":
    main()
