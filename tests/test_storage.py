"""
Tests for the DuckDB-backed run persistence layer.

Each test uses an isolated DuckDB path so the suite never touches the
repository's local ``runs.duckdb``.
"""

from __future__ import annotations

import pytest
from sentiment_analyzer.storage.runs import RunStore


@pytest.fixture
def store(tmp_db_path) -> RunStore:
    return RunStore(tmp_db_path)


def _save(store: RunStore, run_id: str, *, engine: str = "rule_based", **summary):
    store.save_run(
        run_id=run_id,
        engine=engine,
        summary=summary or {"total_records": 0},
        rag_enabled=False,
    )


class TestRunStore:
    def test_initializes_schema_idempotently(self, tmp_db_path):
        # Constructing twice against the same path should not error.
        RunStore(tmp_db_path)
        RunStore(tmp_db_path)

    def test_save_and_retrieve_single_run(self, store):
        _save(store, "run-1", engine="llm", total_records=2)
        loaded = store.get_run("run-1")
        assert loaded["run_id"] == "run-1"
        assert loaded["engine"] == "llm"

    def test_get_run_returns_none_when_missing(self, store):
        assert store.get_run("does-not-exist") is None

    def test_load_all_runs_returns_recent_first(self, store):
        _save(store, "first")
        _save(store, "second")
        runs = store._load_all_runs()
        run_ids = [r["run_id"] for r in runs]
        assert set(run_ids) == {"first", "second"}
        # Each entry exposes the documented contract.
        assert all(
            {"run_id", "engine", "summary", "rag_enabled", "created_at"} <= r.keys() for r in runs
        )

    def test_summary_is_deserialized_from_json_string(self, store):
        _save(store, "json-run", total_records=7)
        runs = store._load_all_runs()
        match = next(r for r in runs if r["run_id"] == "json-run")
        assert match["summary"]["total_records"] == 7
