from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest


APP_DIRECTORY = Path(__file__).resolve().parents[1]
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))


market_query = importlib.import_module("08_market_query")


@pytest.fixture(scope="module")
def state() -> dict:
    return market_query.load_market_state()


def test_static_release_is_complete_and_valid(state: dict) -> None:
    assert state["validation_errors"] == []
    assert state["manifest"]["status"] == "complete"
    assert state["manifest"]["release_id"] == "nawy_2026-07-26"
    assert state["manifest"]["listing_count"] == 13_079
    assert len(state["listings"]) == 13_079


def test_exact_average_uses_record_level_rows(state: dict) -> None:
    result = market_query.query_market(
        "What is the average price of apartments in New Cairo?",
        state,
    )

    assert result["status"] == "answered"
    assert result["matching_row_count"] > 0
    assert result["filters"]["unit_type"] == "apartment"
    assert "location_id" in result["filters"]
    assert "asking price" in result["answer"]
    assert "[S1]" in result["answer"]


def test_arabic_entity_and_list_query(state: dict) -> None:
    result = market_query.query_market(
        "من هم المطورين في القاهرة الجديدة؟",
        state,
    )

    assert result["status"] == "answered"
    assert "location_id" in result["filters"]
    assert "developers" in result["answer"]


def test_unknown_entity_abstains(state: dict) -> None:
    result = market_query.query_market(
        "Who are the developers in Banana City?",
        state,
    )

    assert result["status"] == "insufficient"
    assert "could not resolve" in result["answer"]
    assert "do not know" in result["answer"]


def test_open_ended_unknown_entity_abstains_before_vector_search(state: dict) -> None:
    result = market_query.query_market("Tell me about Banana City", state)

    assert result["status"] == "insufficient"
    assert "could not resolve" in result["answer"]


def test_field_absent_from_every_listing_abstains(state: dict) -> None:
    result = market_query.query_market(
        "What is the delivery date for apartments in New Cairo?",
        state,
    )

    assert result["status"] == "insufficient"
    assert "delivery" in result["answer"]
    assert "not present" in result["answer"]


def test_payment_plan_absence_is_explicit(state: dict) -> None:
    result = market_query.query_market(
        "What payment plans are available in New Cairo?",
        state,
    )

    assert result["status"] == "insufficient"
    assert "payment plan" in result["answer"]
    assert "not present" in result["answer"]


def test_transaction_price_is_never_inferred(state: dict) -> None:
    result = market_query.query_market(
        "What is the average transaction price in New Cairo?",
        state,
    )

    assert result["status"] == "insufficient"
    assert "asking prices" in result["answer"]
    assert "not transaction" in result["answer"]


def test_unit_lookup_returns_only_scraped_fields(state: dict) -> None:
    row = state["listings"][0]
    result = market_query.query_market(
        f"What is the price of unit {row['unit_id']}?",
        state,
    )

    assert result["status"] == "answered"
    assert result["matching_row_count"] >= 1
    assert str(row["unit_id"]) in result["answer"]
    assert "asking price" in result["answer"]


def test_price_threshold_filter_is_deterministic(state: dict) -> None:
    result = market_query.query_market(
        "How many apartments in New Cairo are under 10 million?",
        state,
    )

    assert result["status"] == "answered"
    assert result["filters"]["total_price_egp_lt"] == 10_000_000
    assert result["matching_row_count"] > 0


def test_known_english_alias_merges_duplicate_arabic_location_ids(state: dict) -> None:
    result = market_query.query_market(
        "How many apartments are in Al Alamein?",
        state,
    )

    assert result["status"] == "answered"
    assert isinstance(result["filters"]["location_id"], list)
    assert len(result["filters"]["location_id"]) >= 2
    assert result["matching_row_count"] > 0
