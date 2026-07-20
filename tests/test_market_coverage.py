from __future__ import annotations

from collections import Counter
import importlib
import json
from pathlib import Path
import sys


APP_DIRECTORY = Path(__file__).resolve().parents[1]
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))


def _market_rows() -> list[dict]:
    path = APP_DIRECTORY / "data" / "processed" / "market_facts.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_deployed_market_export_covers_all_validated_rollups() -> None:
    rows = _market_rows()
    counts = Counter(row["entity_type"] for row in rows)

    assert counts == {"location": 34, "developer": 125, "project": 429}
    assert sum(row["record_count"] for row in rows if row["entity_type"] == "location") == 12_733
    assert any(
        any("\u0600" <= character <= "\u06ff" for character in str(row.get("name_ar") or ""))
        for row in rows
    )


def test_new_cairo_query_returns_complete_developer_rollup() -> None:
    documents_stage = importlib.import_module("01_documents")
    store_stage = importlib.import_module("05_create_chroma_store")
    retrieval_stage = importlib.import_module("06_retrieve_context")
    prompting_stage = importlib.import_module("07_prompting")

    rows = _market_rows()
    new_cairo = next(
        row
        for row in rows
        if row.get("entity_type") == "location" and row.get("name") == "New Cairo"
    )
    _, collection = store_stage.build_store_from_documents(documents_stage.load_documents())
    retrieved = retrieval_stage.retrieve_context(
        collection,
        "Who are the developers in New Cairo?",
        top_k=4,
    )
    result = prompting_stage.answer_question(
        "Who are the developers in New Cairo?",
        retrieved,
    )

    assert len(new_cairo["developers"]) == 36
    assert new_cairo["record_count"] == 2_330
    assert retrieved[0]["entity_type"] == "location"
    assert retrieved[0]["entity_name"] == "New Cairo"
    assert "36 developer entities" in result["answer"]
    assert "2,330 latest listing snapshots" in result["answer"]
    assert all(name in result["answer"] for name in new_cairo["developers"])
    assert "Coverage limitation:" in result["answer"]
