from __future__ import annotations

from collections import Counter
import importlib
import json
from pathlib import Path
import sys

import pytest


APP_DIRECTORY = Path(__file__).resolve().parents[1]
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))


def _market_rows() -> list[dict]:
    path = APP_DIRECTORY / "data" / "processed" / "market_facts.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_deployed_market_export_covers_all_validated_rollups() -> None:
    rows = _market_rows()
    counts = Counter(row["entity_type"] for row in rows)

    assert counts == {"overview": 1, "location": 42, "developer": 147, "project": 537}
    assert sum(row["record_count"] for row in rows if row["entity_type"] == "location") == 13_079
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

    assert len(new_cairo["developers"]) == 40
    assert new_cairo["record_count"] == 2_348
    assert retrieved[0]["entity_type"] == "location"
    assert retrieved[0]["entity_name"] == "New Cairo"
    assert "40 developer entities" in result["answer"]
    assert "2,348 latest listing snapshots" in result["answer"]
    assert all(name in result["answer"] for name in new_cairo["developers"])
    assert "Coverage limitation:" in result["answer"]


def test_overview_query_returns_all_locations() -> None:
    documents_stage = importlib.import_module("01_documents")
    store_stage = importlib.import_module("05_create_chroma_store")
    retrieval_stage = importlib.import_module("06_retrieve_context")
    prompting_stage = importlib.import_module("07_prompting")

    rows = _market_rows()
    overview = next(row for row in rows if row.get("entity_type") == "overview")
    _, collection = store_stage.build_store_from_documents(documents_stage.load_documents())
    question = "What locations are available in the Nawy knowledge base?"
    retrieved = retrieval_stage.retrieve_context(collection, question, top_k=4)
    result = prompting_stage.answer_question(question, retrieved)

    assert retrieved[0]["entity_type"] == "overview"
    assert "42 locations" in result["answer"]
    assert all(location in result["answer"] for location in overview["locations"])


def test_unknown_market_entity_refuses_wrong_rollup() -> None:
    documents_stage = importlib.import_module("01_documents")
    store_stage = importlib.import_module("05_create_chroma_store")
    retrieval_stage = importlib.import_module("06_retrieve_context")
    prompting_stage = importlib.import_module("07_prompting")

    _, collection = store_stage.build_store_from_documents(documents_stage.load_documents())
    question = "Who are the developers in Banana City?"
    retrieved = retrieval_stage.retrieve_context(collection, question, top_k=4)
    result = prompting_stage.answer_question(question, retrieved)

    assert "do not have that specific market entity" in result["answer"]


def test_chroma_rebuild_removes_stale_chunks() -> None:
    store_stage = importlib.import_module("05_create_chroma_store")
    vector_stage = importlib.import_module("04_vector_representation")

    first_chunks = vector_stage.vectorize_chunks(
        [
            {
                "chunk_id": "stale-chunk",
                "document_id": "stale-document",
                "source_name": "Stale source",
                "source_url": "local://stale",
                "start_word": 0,
                "text": "obsolete evidence",
            },
            {
                "chunk_id": "current-chunk",
                "document_id": "current-document",
                "source_name": "Current source",
                "source_url": "local://current",
                "start_word": 0,
                "text": "current evidence",
            },
        ]
    )
    _, initial = store_stage.create_chroma_store(
        first_chunks,
        collection_name="stale_chunk_regression",
    )
    assert initial.count() == 2

    _, rebuilt = store_stage.create_chroma_store(
        first_chunks[1:],
        collection_name="stale_chunk_regression",
    )
    assert rebuilt.count() == 1
    assert rebuilt.get()["ids"] == ["current-chunk"]


def test_build_store_falls_back_when_chroma_runtime_is_incompatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_stage = importlib.import_module("05_create_chroma_store")
    documents = [
        {
            "document_id": "fallback-document",
            "source_name": "Fallback source",
            "source_url": "local://fallback",
            "text": "New Cairo apartment asking prices",
        }
    ]

    def fail_chroma(*args, **kwargs):
        raise TypeError("simulated Python runtime incompatibility")

    monkeypatch.setattr(store_stage, "create_chroma_store", fail_chroma)
    with pytest.warns(RuntimeWarning, match="in-memory vector fallback"):
        client, collection = store_stage.build_store_from_documents(documents)

    assert client is None
    assert collection.count() == 1
    result = collection.query(
        query_embeddings=[store_stage.vector_stage.embed_query("New Cairo apartment")],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    assert result["ids"][0]
    assert result["documents"][0] == ["New Cairo apartment asking prices"]
