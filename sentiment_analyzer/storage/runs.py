"""
Run persistence layer using DuckDB.

Why DuckDB:
- Zero external service required
- Analytical queries supported
- Perfect for local systems and prototyping

Concurrency: DuckDB allows a single writing *process* per database file.
Run the API as one process (one uvicorn worker); within it, every
operation uses its own cursor from one shared connection, which is safe
across the threadpool.
"""

import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import duckdb


def _decode_json(value: Any) -> Any:
    """DuckDB returns JSON columns as ``str``; decode them to Python objects."""
    return json.loads(value) if isinstance(value, str) else value


def _utcnow() -> datetime:
    # Stored as a naive TIMESTAMP that is always UTC.
    return datetime.now(UTC).replace(tzinfo=None)


def _as_utc(value: datetime | None) -> datetime | None:
    return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


# ----------------------------------------------------------------------
# Schema migrations. Append only; never edit a released migration.
# ----------------------------------------------------------------------
def _m1_initial(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            engine TEXT,
            summary JSON,
            rag_enabled BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _m2_run_details_and_records(conn: duckdb.DuckDBPyConnection) -> None:
    for column, ddl in [
        ("source_type", "TEXT"),
        ("input_sha256", "TEXT"),
        ("config", "JSON"),
        ("records_in", "INTEGER"),
        ("records_skipped", "INTEGER"),
        ("records_failed", "INTEGER"),
        ("duration_ms", "INTEGER"),
        ("status", "TEXT"),
        ("error", "TEXT"),
    ]:
        conn.execute(f"ALTER TABLE runs ADD COLUMN IF NOT EXISTS {column} {ddl}")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_records (
            run_id TEXT NOT NULL,
            record_id INTEGER NOT NULL,
            sentiment TEXT,
            score DOUBLE,
            confidence DOUBLE,
            contributing_fields JSON,
            drivers JSON,
            rag_chunk_ids JSON,
            truncated BOOLEAN,
            error TEXT,
            PRIMARY KEY (run_id, record_id)
        )
        """
    )


MIGRATIONS: list[Callable[[duckdb.DuckDBPyConnection], None]] = [
    _m1_initial,
    _m2_run_details_and_records,
]

_RUN_COLUMNS = (
    "run_id, engine, summary, rag_enabled, created_at, source_type, input_sha256, "
    "config, records_in, records_skipped, records_failed, duration_ms, status, error"
)


@dataclass
class RunRecord:
    """Run-level metadata persisted for every analysis."""

    run_id: str
    engine: str
    summary: dict
    rag_enabled: bool
    source_type: str | None = None
    input_sha256: str | None = None
    config: dict = field(default_factory=dict)
    records_in: int = 0
    records_skipped: int = 0
    records_failed: int = 0
    duration_ms: int = 0
    status: str = "completed"
    error: str | None = None


class RunStore:
    """
    Persistence layer for sentiment analysis runs.

    Stores run metadata, the aggregate summary and every per-record
    result, to support:
    - Run history and detail inspection
    - Export and auditing (input hash + config snapshot)
    - Run comparison over the same input
    """

    def __init__(self, db_path: str = "runs.duckdb"):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent and db_path != ":memory:":
            os.makedirs(parent, exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._write_lock = threading.Lock()
        self._migrate()

    @contextmanager
    def _cursor(self) -> Iterator[duckdb.DuckDBPyConnection]:
        cur = self.conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def _migrate(self) -> None:
        with self._write_lock, self._cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = cur.execute("SELECT max(version) FROM schema_version").fetchone()
            current = row[0] if row and row[0] is not None else 0
            for version, migration in enumerate(MIGRATIONS, start=1):
                if version <= current:
                    continue
                cur.execute("BEGIN TRANSACTION")
                try:
                    migration(cur)
                    cur.execute("INSERT INTO schema_version VALUES (?)", (version,))
                    cur.execute("COMMIT")
                except Exception:
                    cur.execute("ROLLBACK")
                    raise

    @property
    def schema_version(self) -> int:
        with self._cursor() as cur:
            row = cur.execute("SELECT max(version) FROM schema_version").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def ping(self) -> None:
        with self._cursor() as cur:
            cur.execute("SELECT 1").fetchone()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def save_run(
        self,
        run_id: str | RunRecord,
        engine: str | None = None,
        summary: dict | None = None,
        rag_enabled: bool = False,
        records: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Persist a completed analysis run and its per-record results.

        Accepts a :class:`RunRecord`, or the legacy positional
        ``(run_id, engine, summary, rag_enabled)`` form.
        """
        run = (
            run_id
            if isinstance(run_id, RunRecord)
            else RunRecord(
                run_id=run_id, engine=engine or "", summary=summary or {}, rag_enabled=rag_enabled
            )
        )
        with self._write_lock, self._cursor() as cur:
            cur.execute("BEGIN TRANSACTION")
            try:
                cur.execute(
                    f"INSERT INTO runs ({_RUN_COLUMNS}) "  # noqa: S608 - constant columns
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run.run_id,
                        run.engine,
                        json.dumps(run.summary),
                        run.rag_enabled,
                        _utcnow(),
                        run.source_type,
                        run.input_sha256,
                        json.dumps(run.config),
                        run.records_in,
                        run.records_skipped,
                        run.records_failed,
                        run.duration_ms,
                        run.status,
                        run.error,
                    ),
                )
                if records:
                    cur.executemany(
                        "INSERT INTO run_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            (
                                run.run_id,
                                r["record_id"],
                                r.get("sentiment"),
                                r.get("score"),
                                r.get("confidence"),
                                json.dumps(r.get("contributing_fields") or {}),
                                json.dumps(r.get("drivers") or {}),
                                json.dumps(r.get("rag_chunk_ids") or []),
                                bool(r.get("truncated")),
                                r.get("error"),
                            )
                            for r in records
                        ],
                    )
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_run(row: tuple) -> dict[str, Any]:
        return {
            "run_id": row[0],
            "engine": row[1],
            "summary": _decode_json(row[2]) or {},
            "rag_enabled": bool(row[3]),
            "created_at": _as_utc(row[4]),
            "source_type": row[5],
            "input_sha256": row[6],
            "config": _decode_json(row[7]) or {},
            "records_in": row[8],
            "records_skipped": row[9],
            "records_failed": row[10],
            "duration_ms": row[11],
            "status": row[12] or "completed",
            "error": row[13],
        }

    def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        engine: str | None = None,
        rag_enabled: bool | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return one page of runs (newest first) and the total match count."""
        clauses: list[str] = []
        params: list[Any] = []
        if engine is not None:
            clauses.append("engine = ?")
            params.append(engine)
        if rag_enabled is not None:
            clauses.append("rag_enabled = ?")
            params.append(rag_enabled)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._cursor() as cur:
            # `where` holds only fixed "<column> = ?" clauses; values are bound.
            total_row = cur.execute(
                f"SELECT count(*) FROM runs {where}",  # noqa: S608
                params,
            ).fetchone()
            rows = cur.execute(
                f"SELECT {_RUN_COLUMNS} FROM runs {where} "  # noqa: S608
                "ORDER BY created_at DESC, run_id LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        total = int(total_row[0]) if total_row else 0
        return [self._row_to_run(r) for r in rows], total

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Load one run's metadata and summary by ID."""
        with self._cursor() as cur:
            row = cur.execute(
                f"SELECT {_RUN_COLUMNS} FROM runs WHERE run_id = ?",  # noqa: S608
                (run_id,),
            ).fetchone()
        return self._row_to_run(row) if row else None

    def get_records(
        self, run_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Load a run's per-record results ordered by ``record_id``."""
        sql = (
            "SELECT record_id, sentiment, score, confidence, contributing_fields, drivers, "
            "rag_chunk_ids, truncated, error FROM run_records WHERE run_id = ? "
            "ORDER BY record_id"
        )
        params: list[Any] = [run_id]
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params += [limit, offset]
        with self._cursor() as cur:
            rows = cur.execute(sql, params).fetchall()
        return [
            {
                "record_id": r[0],
                "sentiment": r[1],
                "score": r[2],
                "confidence": r[3],
                "contributing_fields": _decode_json(r[4]) or {},
                "drivers": _decode_json(r[5]) or {},
                "rag_chunk_ids": _decode_json(r[6]) or [],
                "truncated": bool(r[7]),
                "error": r[8],
            }
            for r in rows
        ]

    def count_records(self, run_id: str) -> int:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT count(*) FROM run_records WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self.conn.close()
