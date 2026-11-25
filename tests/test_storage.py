"""
Tests for the DuckDB-backed run persistence layer, including schema
migrations from the original v1.0.0 layout.
"""

from __future__ import annotations

import duckdb
import pytest

from sentiment_analyzer.storage.runs import MIGRATIONS, RunRecord, RunStore


@pytest.fixture
def store(tmp_db_path) -> RunStore:
    return RunStore(tmp_db_path)


def _record(record_id: int, sentiment: str = "positive", score: float = 0.5, **extra) -> dict:
    return {
        "record_id": record_id,
        "sentiment": sentiment,
        "score": score,
        "confidence": abs(score),
        "contributing_fields": {"text": score},
        "drivers": {"x": 1.0},
        "rag_chunk_ids": ["d:0"],
        "truncated": False,
        "error": None,
        **extra,
    }


class TestRunStore:
    def test_initializes_schema_idempotently(self, tmp_db_path):
        RunStore(tmp_db_path).close()
        store = RunStore(tmp_db_path)
        assert store.schema_version == len(MIGRATIONS)

    def test_creates_parent_directories(self, tmp_path):
        store = RunStore(str(tmp_path / "nested" / "dir" / "runs.duckdb"))
        store.ping()

    def test_save_and_retrieve_run_with_records(self, store):
        store.save_run(
            RunRecord(
                run_id="run-1",
                engine="llm",
                summary={"total_records": 2},
                rag_enabled=True,
                source_type="json",
                input_sha256="abc",
                config={"engine": "llm"},
                records_in=3,
                records_skipped=1,
                duration_ms=12,
            ),
            records=[_record(0), _record(2, "negative", -0.5, error=None)],
        )
        run = store.get_run("run-1")
        assert run["engine"] == "llm"
        assert run["summary"] == {"total_records": 2}
        assert run["config"] == {"engine": "llm"}
        assert run["records_skipped"] == 1
        assert run["created_at"].tzinfo is not None

        records = store.get_records("run-1")
        assert [r["record_id"] for r in records] == [0, 2]
        assert records[1]["sentiment"] == "negative"
        assert records[0]["rag_chunk_ids"] == ["d:0"]
        assert store.count_records("run-1") == 2
        assert store.get_records("run-1", limit=1, offset=1)[0]["record_id"] == 2

    def test_legacy_positional_save(self, store):
        store.save_run("legacy", "rule_based", {"total_records": 0}, False)
        assert store.get_run("legacy")["summary"] == {"total_records": 0}

    def test_get_run_returns_none_when_missing(self, store):
        assert store.get_run("does-not-exist") is None

    def test_list_runs_pagination_and_filters(self, store):
        for i, (engine, rag) in enumerate([("a", False), ("b", True), ("a", True)]):
            store.save_run(RunRecord(run_id=f"r{i}", engine=engine, summary={}, rag_enabled=rag))

        items, total = store.list_runs(limit=2)
        assert total == 3 and len(items) == 2
        assert {r["run_id"] for r in store.list_runs(engine="a")[0]} == {"r0", "r2"}
        assert [r["run_id"] for r in store.list_runs(engine="a", rag_enabled=True)[0]] == ["r2"]

    def test_failed_save_is_rolled_back(self, store):
        store.save_run(RunRecord(run_id="dup", engine="x", summary={}, rag_enabled=False))
        with pytest.raises(duckdb.Error):
            store.save_run(
                RunRecord(run_id="dup", engine="x", summary={}, rag_enabled=False),
                records=[_record(0)],
            )
        assert store.count_records("dup") == 0


class TestMigrations:
    def test_upgrades_a_v1_database_in_place(self, tmp_db_path):
        """Databases created by v1.0.0 have no schema_version table."""
        conn = duckdb.connect(tmp_db_path)
        conn.execute(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, engine TEXT, summary JSON,
                rag_enabled BOOLEAN, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "INSERT INTO runs (run_id, engine, summary, rag_enabled) VALUES (?, ?, ?, ?)",
            ("old", "rule_based", '{"total_records": 4}', False),
        )
        conn.close()

        store = RunStore(tmp_db_path)
        assert store.schema_version == len(MIGRATIONS)
        old = store.get_run("old")
        assert old is not None
        assert old["summary"] == {"total_records": 4}
        assert old["status"] == "completed"
        assert old["input_sha256"] is None
        assert store.get_records("old") == []
