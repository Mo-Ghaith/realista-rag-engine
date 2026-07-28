"""Versioned record-level Nawy release export and loading helpers.

The Streamlit app consumes static files so it never needs MongoDB credentials.
Only validated, entity-resolved latest-unit observations enter a release.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


RELEASE_SCHEMA_VERSION = "realista_nawy_rag_release_v1"
LISTING_FIELDS = (
    "snapshot_id",
    "crawl_batch_id",
    "captured_at",
    "source_record_id",
    "source_url",
    "canonical_url",
    "language",
    "developer_id",
    "project_id",
    "location_id",
    "unit_id",
    "unit_type_id",
    "unit_type_source",
    "total_price_egp",
    "area_sqm",
    "price_per_sqm_egp",
    "bedrooms",
    "bathrooms",
    "delivery",
    "availability_status",
    "market_trust_status",
    "validation_status",
    "entity_status",
)


def export_release_from_database(
    database: Any,
    output_directory: str | Path,
    *,
    snapshot_collection: str | None = None,
    release_status: str = "complete",
) -> dict[str, Any]:
    """Export the latest validated row per Nawy unit plus a hashed manifest."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    available = set(database.list_collection_names())
    source_collection = snapshot_collection or (
        "listing_snapshots"
        if "listing_snapshots" in available
        else "nawy_listing_snapshots"
    )
    if source_collection not in available:
        raise RuntimeError("No Nawy listing snapshot collection is available.")

    entity_names = _load_entity_names(database)
    relationship_names = _load_rollup_names(database)
    query = {
        "source": "nawy",
        "market_trust_status": "valid",
        "validation_status": "valid",
        "entity_status": "entity_resolved",
    }
    projection = {"_id": 0, **{field: 1 for field in LISTING_FIELDS}}
    projection["raw_fields"] = 1
    latest_by_identity: dict[str, dict[str, Any]] = {}
    source_row_count = 0
    for row in database[source_collection].find(query, projection).sort("captured_at", 1):
        source_row_count += 1
        identity = _listing_identity(row)
        if not identity:
            continue
        latest_by_identity[identity] = row

    listings = [
        _normalize_listing(row, entity_names, relationship_names)
        for _, row in sorted(latest_by_identity.items())
    ]
    listings_path = output / "nawy_listings.jsonl"
    _write_jsonl(listings_path, listings)

    capture_values = [
        row["captured_at"] for row in listings if row.get("captured_at")
    ]
    cutoff = max(capture_values) if capture_values else None
    release_date = str(cutoff or "unknown").split("T", 1)[0]
    release_id = f"nawy_{release_date}"
    entity_counts = {
        "developers": len({row["developer_id"] for row in listings}),
        "projects": len({row["project_id"] for row in listings}),
        "locations": len({row["location_id"] for row in listings}),
        "unit_types": len({row["unit_type_id"] for row in listings}),
    }
    field_coverage = {
        field: sum(row.get(field) not in (None, "", [], {}) for row in listings)
        for field in (
            "total_price_egp",
            "area_sqm",
            "price_per_sqm_egp",
            "bedrooms",
            "bathrooms",
            "delivery",
            "availability_status",
            "payment_plan",
            "down_payment",
            "installment_years",
            "finishing",
            "source_url",
        )
    }
    manifest = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": release_id,
        "status": release_status,
        "source": "nawy",
        "timezone": "Africa/Cairo",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capture_start": min(capture_values) if capture_values else None,
        "capture_cutoff": cutoff,
        "source_collection": source_collection,
        "source_row_count": source_row_count,
        "listing_count": len(listings),
        "superseded_observation_count": max(source_row_count - len(listings), 0),
        "entity_counts": entity_counts,
        "field_coverage": field_coverage,
        "crawl_batch_ids": sorted(
            {str(row["crawl_batch_id"]) for row in listings if row.get("crawl_batch_id")}
        ),
        "files": {
            "nawy_listings.jsonl": {
                "sha256": _sha256(listings_path),
                "bytes": listings_path.stat().st_size,
                "rows": len(listings),
            }
        },
        "limitations": [
            "The release contains validated Nawy asking-price observations, not transaction prices.",
            "A missing field means it was not present in the normalized scrape and must not be inferred.",
            "The release represents the authorized crawl coverage at its recorded cutoff.",
        ],
    }
    manifest_path = output / "nawy_release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_manifest(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "nawy_release_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_listings(directory: str | Path) -> list[dict[str, Any]]:
    path = Path(directory) / "nawy_listings.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_release(directory: str | Path) -> list[str]:
    """Return validation errors; an empty list means the static release is sound."""

    base = Path(directory)
    manifest = load_manifest(base)
    errors: list[str] = []
    if not manifest:
        return ["release manifest is missing"]
    if manifest.get("status") != "complete":
        errors.append("release status is not complete")
    for filename, expected in (manifest.get("files") or {}).items():
        path = base / filename
        if not path.exists():
            errors.append(f"{filename} is missing")
            continue
        if expected.get("sha256") != _sha256(path):
            errors.append(f"{filename} hash does not match the manifest")
    listings = load_listings(base)
    if len(listings) != int(manifest.get("listing_count") or 0):
        errors.append("listing count does not match the manifest")
    identities = [
        f"{row.get('project_id')}:{row.get('unit_id')}" for row in listings
    ]
    if len(identities) != len(set(identities)):
        errors.append("release contains duplicate project/unit identities")
    return errors


def _normalize_listing(
    row: dict[str, Any],
    entity_names: dict[str, dict[str, Any]],
    relationship_names: dict[str, str],
) -> dict[str, Any]:
    raw_fields = row.get("raw_fields") if isinstance(row.get("raw_fields"), dict) else {}
    normalized = {
        field: _json_value(row.get(field))
        for field in LISTING_FIELDS
        if row.get(field) is not None
    }
    for entity_type in ("developer", "project", "location", "unit_type"):
        entity_id = str(row.get(f"{entity_type}_id") or "")
        names = entity_names.get(entity_id, {})
        fallback_name = relationship_names.get(entity_id)
        aliases = [
            str(value)
            for value in [
                names.get("name_en"),
                names.get("name_ar"),
                fallback_name,
                *(names.get("aliases") or []),
            ]
            if str(value or "").strip()
        ]
        normalized[f"{entity_type}_name_en"] = _first_by_script(
            aliases, arabic=False
        )
        normalized[f"{entity_type}_name_ar"] = _first_by_script(
            aliases, arabic=True
        )
        normalized[f"{entity_type}_aliases"] = list(dict.fromkeys(aliases))

    if not normalized.get("unit_type_name_en") and row.get("unit_type_source"):
        normalized["unit_type_name_en"] = row.get("unit_type_source")
    for field, raw_key in (
        ("developer_name_en", "developer_name"),
        ("project_name_en", "project_name"),
        ("location_name_en", "area_name"),
        ("unit_type_name_en", "unit_type"),
    ):
        if not normalized.get(field) and raw_fields.get(raw_key):
            normalized[field] = raw_fields[raw_key]
    normalized["evidence_id"] = (
        f"nawy_listing:{normalized.get('project_id')}:{normalized.get('unit_id')}"
    )
    return {
        key: value
        for key, value in normalized.items()
        if value not in (None, "", [], {})
    }


def _load_entity_names(database: Any) -> dict[str, dict[str, Any]]:
    if "nawy_entities" not in database.list_collection_names():
        return {}
    result: dict[str, dict[str, Any]] = {}
    projection = {"_id": 0, "entity_id": 1, "names": 1, "aliases": 1}
    for row in database["nawy_entities"].find({"source": "nawy"}, projection):
        entity_id = str(row.get("entity_id") or "")
        names = row.get("names") if isinstance(row.get("names"), dict) else {}
        if entity_id:
            result[entity_id] = {
                "name_en": names.get("en"),
                "name_ar": names.get("ar"),
                "aliases": [str(value) for value in row.get("aliases") or []],
            }
    return result


def _load_rollup_names(database: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for collection_name, id_field in (
        ("developers", "developer_id"),
        ("projects", "project_id"),
        ("locations", "location_id"),
    ):
        if collection_name not in database.list_collection_names():
            continue
        projection = {"_id": 0, id_field: 1, "name": 1, "name_en": 1, "name_ar": 1}
        for row in database[collection_name].find({}, projection):
            entity_id = str(row.get(id_field) or "")
            name = row.get("name_en") or row.get("name") or row.get("name_ar")
            if entity_id and name:
                result[entity_id] = str(name)
    return result


def _listing_identity(row: dict[str, Any]) -> str | None:
    project_id = str(row.get("project_id") or "").strip()
    unit_id = str(row.get("unit_id") or "").strip()
    if project_id and unit_id:
        return f"{project_id}:{unit_id}"
    return None


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _first_by_script(values: Iterable[str], *, arabic: bool) -> str | None:
    for value in values:
        if _has_arabic(value) is arabic:
            return value
    return None


def _has_arabic(value: str) -> bool:
    return any("\u0600" <= character <= "\u06ff" for character in str(value))
