"""Structured Nawy retrieval for exact questions over the static scrape release.

Vector search remains useful for descriptive text. Counts, filters, prices,
areas, entity lists, and unit lookups are answered from record-level rows.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re
from statistics import mean, median
from pathlib import Path
from typing import Any, Iterable
import unicodedata

from nawy_release import load_listings, load_manifest, validate_release


APP_DIRECTORY = Path(__file__).resolve().parent
PROCESSED_DIRECTORY = APP_DIRECTORY / "data" / "processed"
TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
GENERIC_ALIASES = {
    "area",
    "city",
    "compound",
    "developer",
    "location",
    "project",
    "property",
    "unit",
    "مطور",
    "مشروع",
    "منطقة",
    "كمبوند",
}
UNSUPPORTED_FIELD_TERMS = {
    "bedrooms": {"bedroom", "bedrooms", "beds", "غرف", "غرفة"},
    "bathrooms": {"bathroom", "bathrooms", "baths", "حمام", "حمامات"},
    "delivery": {"delivery", "handover", "استلام", "التسليم"},
    "availability_status": {"availability", "إتاحة"},
    "payment_plan": {
        "payment",
        "payments",
        "installment",
        "installments",
        "تقسيط",
        "قسط",
    },
    "finishing": {"finishing", "finished", "تشطيب", "التشطيب"},
}
UNSUPPORTED_CONCEPTS = {
    "transaction": {"transaction", "transactions", "sold price", "sale price"},
    "roi": {"roi", "return on investment", "investment return"},
    "rent": {"rent", "rental yield", "إيجار", "عائد"},
}
UNIT_TYPE_ALIASES = {
    "apartment": {"apartment", "apartments", "شقة", "شقق"},
    "villa": {"villa", "villas", "فيلا", "فيلات"},
    "townhouse": {"townhouse", "townhouses", "تاون هاوس"},
    "twin house": {"twin house", "twinhouse", "توين هاوس"},
    "duplex": {"duplex", "دوبلكس"},
    "penthouse": {"penthouse", "بنتهاوس"},
    "chalet": {"chalet", "chalets", "شاليه", "شاليهات"},
    "studio": {"studio", "studios", "استوديو", "ستوديو"},
    "office": {"office", "offices", "مكتب", "مكاتب"},
    "retail": {"retail", "shop", "shops", "محل", "محلات"},
    "clinic": {"clinic", "clinics", "عيادة", "عيادات"},
    "cabin": {"cabin", "cabins", "كابينة"},
}
KNOWN_LOCATION_ALIASES = {
    "هليوبوليس الجديدة": {"New Heliopolis"},
    "المعادي": {"Maadi"},
    "الشيخ زايد": {"Sheikh Zayed", "El Sheikh Zayed"},
    "الساحل الشمالي": {"North Coast", "Sahel"},
    "رأس الحكمة": {"Ras El Hekma", "Ras El Hikma"},
    "سيدي عبد الرحمن": {"Sidi Abdel Rahman", "Sidi Abd El Rahman"},
    "العلمين": {"Al Alamein", "El Alamein"},
    "غزالة باي": {"Ghazala Bay"},
    "التوسعات الشمالية": {"Northern Expansions"},
    "الشيخ زايد الجديدة": {"New Sheikh Zayed"},
    "جنوب القاهرة الجديدة": {"South New Cairo"},
    "التجمع السادس": {"Sixth Settlement", "6th Settlement"},
    "اللوتس": {"Lotus"},
    "الشويفات": {"Choueifat"},
    "المستثمرين الشمالية": {"North Investors", "Northern Investors"},
    "الجونة": {"El Gouna", "Gouna"},
    "جولدن سكوير": {"Golden Square"},
    "المستثمرين الجنوبية": {"South Investors", "Southern Investors"},
    "مدينه الحمام": {"Al Hammam City", "El Hammam City"},
    "مدينة الحمام": {"Al Hammam City", "El Hammam City"},
    "الضبعة": {"Al Dabaa", "El Dabaa"},
    "حدائق اكتوبر": {"October Gardens"},
    "سيدي حنيش": {"Sidi Heneish", "Sidi Hanish"},
    "مدينة المستقبل": {"Mostakbal City"},
}


def load_market_state(
    processed_directory: str | Path = PROCESSED_DIRECTORY,
) -> dict[str, Any]:
    directory = Path(processed_directory)
    manifest = load_manifest(directory)
    listings = load_listings(directory)
    aliases: dict[str, list[tuple[str, str]]] = defaultdict(list)
    entity_names: dict[str, dict[str, str]] = defaultdict(dict)

    for row in listings:
        for entity_type in ("developer", "project", "location"):
            entity_id = str(row.get(f"{entity_type}_id") or "")
            if not entity_id:
                continue
            names = [
                row.get(f"{entity_type}_name_en"),
                row.get(f"{entity_type}_name_ar"),
                *(row.get(f"{entity_type}_aliases") or []),
            ]
            for name in names:
                clean = str(name or "").strip()
                normalized = normalize_text(clean)
                if (
                    not clean
                    or normalized in GENERIC_ALIASES
                    or len(normalized) < 3
                ):
                    continue
                aliases[entity_type].append((normalized, entity_id))
                if entity_id not in entity_names[entity_type]:
                    entity_names[entity_type][entity_id] = clean
                elif _has_latin(clean) and not _has_latin(
                    entity_names[entity_type][entity_id]
                ):
                    entity_names[entity_type][entity_id] = clean

    for entity_type in aliases:
        aliases[entity_type] = sorted(
            set(aliases[entity_type]), key=lambda item: len(item[0]), reverse=True
        )
    _add_known_location_aliases(aliases, entity_names)
    return {
        "manifest": manifest,
        "listings": listings,
        "aliases": dict(aliases),
        "entity_names": dict(entity_names),
        "validation_errors": validate_release(directory),
    }


def query_market(
    question: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Return handled structured evidence, an abstention, or ``unhandled``."""

    question = str(question or "").strip()
    normalized = normalize_text(question)
    manifest = state.get("manifest") or {}
    release_label = _release_label(manifest)
    if not question:
        return {"status": "unhandled"}
    if state.get("validation_errors"):
        return _abstention(
            "The packaged Nawy release failed integrity validation, so I cannot "
            "safely answer from it.",
            manifest,
            reason="invalid_release",
        )

    concept = _requested_unsupported_concept(normalized)
    if concept:
        return _abstention(
            f"The {release_label} Nawy scrape contains asking prices, not "
            f"{concept} evidence. I do not know that answer from this scrape.",
            manifest,
            reason=f"unsupported_{concept}",
        )

    missing_field = _requested_unavailable_field(normalized, manifest)
    if missing_field:
        return _abstention(
            f"The requested {missing_field.replace('_', ' ')} field is not present "
            f"in any normalized listing in the {release_label} scrape, so I do not "
            "know that answer from the available data.",
            manifest,
            reason=f"field_missing_{missing_field}",
        )

    resolved = {
        entity_type: _resolve_entities(
            normalized, state.get("aliases", {}).get(entity_type, [])
        )
        for entity_type in ("developer", "project", "location")
    }
    unit_type = _resolve_unit_type(normalized)
    unit_id = _extract_unit_id(normalized)
    intent = _intent(normalized, unit_id=unit_id)
    unknown_scope = _unknown_requested_scope(normalized, resolved)
    if unknown_scope:
        return _abstention(
            f"I could not resolve `{unknown_scope}` to a location, developer, or "
            f"project in the {release_label} scrape, so I do not know the requested "
            "answer from this release.",
            manifest,
            reason="entity_not_in_release",
        )
    if intent == "unhandled":
        return {"status": "unhandled", "resolved_entities": resolved}

    rows = list(state.get("listings") or [])
    filters: dict[str, Any] = {}
    for entity_type, matches in resolved.items():
        if matches:
            selected = set(matches)
            filters[f"{entity_type}_id"] = (
                matches[0] if len(matches) == 1 else matches
            )
            rows = [
                row
                for row in rows
                if str(row.get(f"{entity_type}_id") or "") in selected
            ]
    if unit_type:
        filters["unit_type"] = unit_type
        rows = [row for row in rows if _listing_matches_unit_type(row, unit_type)]
    if unit_id:
        filters["unit_id"] = unit_id
        rows = [row for row in rows if str(row.get("unit_id") or "") == unit_id]

    price_filter = _price_filter(normalized)
    if price_filter:
        operator, threshold = price_filter
        filters[f"total_price_egp_{operator}"] = threshold
        if operator == "lt":
            rows = [
                row
                for row in rows
                if _number(row.get("total_price_egp")) is not None
                and _number(row.get("total_price_egp")) < threshold
            ]
        else:
            rows = [
                row
                for row in rows
                if _number(row.get("total_price_egp")) is not None
                and _number(row.get("total_price_egp")) > threshold
            ]

    if not rows:
        return _abstention(
            f"No validated listings in the {release_label} scrape match the "
            f"requested filters ({_format_filters(filters, state)}). I do not know "
            "an answer beyond that zero-match result.",
            manifest,
            reason="no_matching_rows",
            filters=filters,
        )

    if intent == "count":
        answer = (
            f"The {release_label} scrape contains **{len(rows):,}** validated latest "
            f"listing observations matching {_format_filters(filters, state)} [S1]."
        )
        return _answered(answer, rows, manifest, filters, "count")

    if intent.startswith("list_"):
        entity_type = intent.removeprefix("list_")
        return _entity_list_answer(
            entity_type, rows, state, manifest, filters
        )

    if intent == "unit_lookup":
        return _unit_answer(rows[0], manifest, filters)

    if intent.startswith("metric_"):
        _, operation, field = intent.split("_", 2)
        values = [
            value
            for row in rows
            if (value := _number(row.get(field))) is not None
        ]
        if not values:
            return _abstention(
                f"The matching listings do not contain {field.replace('_', ' ')}. "
                "I do not know that value from the scrape.",
                manifest,
                reason=f"matching_field_missing_{field}",
                filters=filters,
            )
        value_text = _metric_value(operation, values, field)
        answer = (
            f"For **{len(values):,}** validated listings matching "
            f"{_format_filters(filters, state)}, the {operation} "
            f"{_field_label(field)} is **{value_text}** [S1]. "
            f"These are Nawy asking-price observations from the {release_label} "
            "release, not transaction prices [S1]."
        )
        return _answered(answer, rows, manifest, filters, intent, values=values)

    return {"status": "unhandled", "resolved_entities": resolved}


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
    text = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in text
    )
    text = re.sub(r"[^\w\u0600-\u06ff]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _intent(question: str, *, unit_id: str | None) -> str:
    terms = set(TOKEN_PATTERN.findall(question))
    if unit_id:
        return "unit_lookup"
    list_map = {
        "developer": {"developer", "developers", "المطور", "المطورين"},
        "project": {"project", "projects", "compound", "compounds", "مشروع", "مشاريع"},
        "location": {"location", "locations", "area", "areas", "منطقة", "مناطق"},
        "unit_type": {"type", "types", "نوع", "أنواع"},
    }
    list_signal = bool(
        terms & {"which", "what", "who", "list", "show", "ما", "من", "اعرض"}
    )
    for entity_type, entity_terms in list_map.items():
        if list_signal and terms & entity_terms:
            return f"list_{entity_type}"
    if terms & {"count", "many", "number", "كم", "عدد"}:
        return "count"

    field = "price_per_sqm_egp" if _wants_price_per_sqm(question) else (
        "area_sqm" if terms & {"area", "sqm", "مساحة", "المساحة"} else "total_price_egp"
    )
    if terms & {"average", "avg", "mean", "متوسط"}:
        return f"metric_mean_{field}"
    if terms & {"median", "وسيط"}:
        return f"metric_median_{field}"
    if terms & {"minimum", "min", "cheapest", "lowest", "أقل", "ارخص", "أرخص"}:
        return f"metric_min_{field}"
    if terms & {"maximum", "max", "highest", "expensive", "أعلى", "اغلى", "أغلى"}:
        return f"metric_max_{field}"
    if terms & {"range", "نطاق"}:
        return f"metric_range_{field}"
    return "unhandled"


def _resolve_entities(
    question: str, aliases: list[tuple[str, str]]
) -> list[str]:
    padded = f" {question} "
    matches: list[tuple[int, str]] = []
    for alias, entity_id in aliases:
        if f" {alias} " in padded or question == alias:
            matches.append((len(alias), entity_id))
    if not matches:
        return []
    longest = max(length for length, _ in matches)
    return list(dict.fromkeys(entity_id for length, entity_id in matches if length == longest))


def _add_known_location_aliases(
    aliases: dict[str, list[tuple[str, str]]],
    entity_names: dict[str, dict[str, str]],
) -> None:
    location_aliases = aliases.get("location", [])
    additions: list[tuple[str, str]] = []
    for alias, entity_id in location_aliases:
        for arabic_name, english_aliases in KNOWN_LOCATION_ALIASES.items():
            if alias != normalize_text(arabic_name):
                continue
            for english_alias in english_aliases:
                additions.append((normalize_text(english_alias), entity_id))
            if not _has_latin(entity_names["location"].get(entity_id, "")):
                entity_names["location"][entity_id] = sorted(english_aliases)[0]
    aliases["location"] = sorted(
        set(location_aliases + additions),
        key=lambda item: len(item[0]),
        reverse=True,
    )


def _resolve_unit_type(question: str) -> str | None:
    padded = f" {question} "
    for canonical, aliases in UNIT_TYPE_ALIASES.items():
        if any(f" {normalize_text(alias)} " in padded for alias in aliases):
            return canonical
    return None


def _listing_matches_unit_type(row: dict[str, Any], canonical: str) -> bool:
    values = [
        row.get("unit_type_name_en"),
        row.get("unit_type_name_ar"),
        row.get("unit_type_source"),
        row.get("unit_type_id"),
    ]
    aliases = {normalize_text(value) for value in UNIT_TYPE_ALIASES[canonical]}
    for value in values:
        normalized = normalize_text(str(value or "").replace("unit_type-", ""))
        if normalized in aliases or canonical in normalized:
            return True
    return False


def _extract_unit_id(question: str) -> str | None:
    match = re.search(
        r"\b(?:unit|listing|property|وحدة)\s*(?:id)?\s*#?\s*(\d{3,})\b",
        question,
    )
    return match.group(1) if match else None


def _unknown_requested_scope(
    question: str, resolved: dict[str, list[str]]
) -> str | None:
    """Identify explicit unresolved ``in …``/``by …`` scopes conservatively."""

    if any(resolved.values()):
        return None
    match = re.search(r"\b(?:in|في|about|عن)\s+(.+)$", question)
    if not match:
        return None
    candidate = match.group(1).strip()
    candidate = re.sub(r"\b(?:scrape|release|knowledge base|data|dataset)\b.*$", "", candidate).strip()
    generic = {
        "",
        "egypt",
        "egyptian market",
        "market",
        "nawy",
        "the market",
        "المعرفة",
        "السوق",
        "مصر",
    }
    return None if candidate in generic else candidate


def _price_filter(question: str) -> tuple[str, float] | None:
    match = re.search(
        r"\b(under|below|less than|over|above|more than|اقل من|أقل من|اكثر من|أكثر من)"
        r"\s+([\d,.]+)\s*(million|m|مليون)?\b",
        question,
    )
    if not match:
        return None
    value = float(match.group(2).replace(",", ""))
    if match.group(3):
        value *= 1_000_000
    operator = "lt" if match.group(1) in {
        "under", "below", "less than", "اقل من", "أقل من"
    } else "gt"
    return operator, value


def _requested_unavailable_field(
    question: str, manifest: dict[str, Any]
) -> str | None:
    coverage = manifest.get("field_coverage") or {}
    terms = set(TOKEN_PATTERN.findall(question))
    if not int(coverage.get("availability_status") or 0) and any(
        phrase in question
        for phrase in ("available units", "available listings", "وحدات متاحة")
    ):
        return "availability_status"
    for field, candidates in UNSUPPORTED_FIELD_TERMS.items():
        if not int(coverage.get(field) or 0) and terms & candidates:
            return field
    return None


def _requested_unsupported_concept(question: str) -> str | None:
    for concept, candidates in UNSUPPORTED_CONCEPTS.items():
        if any(candidate in question for candidate in candidates):
            return concept
    return None


def _entity_list_answer(
    entity_type: str,
    rows: list[dict[str, Any]],
    state: dict[str, Any],
    manifest: dict[str, Any],
    filters: dict[str, Any],
) -> dict[str, Any]:
    if entity_type == "unit_type":
        names = sorted(
            {
                str(
                    row.get("unit_type_name_en")
                    or row.get("unit_type_source")
                    or row.get("unit_type_name_ar")
                    or row.get("unit_type_id")
                )
                for row in rows
            }
        )
    else:
        names_by_id = state.get("entity_names", {}).get(entity_type, {})
        names = sorted(
            {
                names_by_id.get(str(row.get(f"{entity_type}_id") or ""))
                or str(row.get(f"{entity_type}_name_en") or row.get(f"{entity_type}_name_ar"))
                for row in rows
                if row.get(f"{entity_type}_id")
            }
        )
    heading = entity_type.replace("_", " ")
    rendered = ", ".join(names)
    answer = (
        f"The {_release_label(manifest)} scrape contains **{len(names):,}** "
        f"{heading}{'' if len(names) == 1 else 's'} matching "
        f"{_format_filters(filters, state)} [S1].\n\n{rendered} [S1]"
    )
    return _answered(answer, rows, manifest, filters, f"list_{entity_type}", names=names)


def _unit_answer(
    row: dict[str, Any],
    manifest: dict[str, Any],
    filters: dict[str, Any],
) -> dict[str, Any]:
    facts = [
        f"unit `{row.get('unit_id')}`",
        f"project {row.get('project_name_en') or row.get('project_name_ar') or row.get('project_id')}",
        f"location {row.get('location_name_en') or row.get('location_name_ar') or row.get('location_id')}",
        f"type {row.get('unit_type_name_en') or row.get('unit_type_source') or row.get('unit_type_id')}",
        f"asking price {_money(row.get('total_price_egp'))}",
    ]
    if row.get("area_sqm"):
        facts.append(f"area {float(row['area_sqm']):,.0f} m²")
    if row.get("price_per_sqm_egp"):
        facts.append(f"price/m² {_money(row['price_per_sqm_egp'])}")
    answer = (
        "The scraped record contains " + ", ".join(facts) + " [S1]. "
        f"Capture time: {row.get('captured_at')} [S1]."
    )
    return _answered(answer, [row], manifest, filters, "unit_lookup")


def _answered(
    answer: str,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    filters: dict[str, Any],
    operation: str,
    **summary: Any,
) -> dict[str, Any]:
    evidence = {
        "release_id": manifest.get("release_id"),
        "capture_cutoff": manifest.get("capture_cutoff"),
        "operation": operation,
        "filters": filters,
        "matching_row_count": len(rows),
        **summary,
        "sample_evidence_ids": [row.get("evidence_id") for row in rows[:20]],
        "sample_source_urls": list(
            dict.fromkeys(
                str(row.get("source_url"))
                for row in rows[:20]
                if row.get("source_url")
            )
        )[:10],
    }
    return {
        "status": "answered",
        "answer": answer,
        "filters": filters,
        "matching_row_count": len(rows),
        "retrieved": [_evidence_item(evidence, answer)],
    }


def _abstention(
    answer: str,
    manifest: dict[str, Any],
    *,
    reason: str,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answer_with_citation = f"{answer} [S1]"
    evidence = {
        "release_id": manifest.get("release_id"),
        "capture_cutoff": manifest.get("capture_cutoff"),
        "status": "insufficient",
        "reason": reason,
        "filters": filters or {},
        "field_coverage": manifest.get("field_coverage") or {},
        "limitations": manifest.get("limitations") or [],
    }
    return {
        "status": "insufficient",
        "answer": answer_with_citation,
        "filters": filters or {},
        "matching_row_count": 0,
        "retrieved": [_evidence_item(evidence, answer_with_citation)],
    }


def _evidence_item(evidence: dict[str, Any], prepared_answer: str) -> dict[str, Any]:
    serialized = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    chunk_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
    return {
        "citation": "S1",
        "chunk_id": chunk_id,
        "document_id": f"structured_{chunk_id}",
        "source_name": "Nawy structured release query",
        "source_url": (evidence.get("sample_source_urls") or [""])[0],
        "document_type": "structured_market_result",
        "entity_type": "structured_query",
        "entity_name": evidence.get("release_id") or "Nawy release",
        "distance": 0.0,
        "text": f"Prepared answer: {prepared_answer}\nEvidence: {serialized}",
        "prepared_answer": prepared_answer,
    }


def _format_filters(filters: dict[str, Any], state: dict[str, Any]) -> str:
    if not filters:
        return "the complete release"
    names = state.get("entity_names") or {}
    parts: list[str] = []
    for key, value in filters.items():
        if key.endswith("_id"):
            entity_type = key.removesuffix("_id")
            values = value if isinstance(value, list) else [value]
            display = list(
                dict.fromkeys(
                    names.get(entity_type, {}).get(str(item), str(item))
                    for item in values
                )
            )
            parts.append(f"{entity_type} {' / '.join(display)}")
        elif key == "unit_type":
            parts.append(f"unit type {value}")
        elif key == "unit_id":
            parts.append(f"unit ID {value}")
        elif key.endswith("_lt"):
            parts.append(f"asking price below {_money(value)}")
        elif key.endswith("_gt"):
            parts.append(f"asking price above {_money(value)}")
    return ", ".join(parts) or "the requested scope"


def _metric_value(operation: str, values: list[float], field: str) -> str:
    if operation == "mean":
        value: float | tuple[float, float] = mean(values)
    elif operation == "median":
        value = median(values)
    elif operation == "min":
        value = min(values)
    elif operation == "max":
        value = max(values)
    else:
        value = (min(values), max(values))
    if isinstance(value, tuple):
        return f"{_format_number(value[0], field)} to {_format_number(value[1], field)}"
    return _format_number(value, field)


def _format_number(value: float, field: str) -> str:
    if field == "area_sqm":
        return f"{value:,.1f} m²"
    return _money(value)


def _field_label(field: str) -> str:
    return {
        "total_price_egp": "asking price",
        "price_per_sqm_egp": "asking price per m²",
        "area_sqm": "area",
    }[field]


def _money(value: Any) -> str:
    number = _number(value)
    return f"{number:,.0f} EGP" if number is not None else "unavailable"


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _release_label(manifest: dict[str, Any]) -> str:
    release_id = str(manifest.get("release_id") or "available")
    cutoff = str(manifest.get("capture_cutoff") or "").split("T", 1)[0]
    return f"{release_id} ({cutoff})" if cutoff and cutoff not in release_id else release_id


def _wants_price_per_sqm(question: str) -> bool:
    return any(
        phrase in question
        for phrase in (
            "price per sqm",
            "price per square meter",
            "price m2",
            "سعر المتر",
        )
    )


def _has_latin(value: str) -> bool:
    return any("a" <= character.casefold() <= "z" for character in str(value))
