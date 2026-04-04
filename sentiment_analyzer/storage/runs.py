"""
Run persistence layer using DuckDB.

Why DuckDB:
- Zero external service required
- Analytical queries supported
- Perfect for local systems and prototyping
"""

import json

import duckdb


class RunStore:
    """
    Persistence layer for sentiment analysis runs.

    Stores lightweight run metadata and summary information
    to support:
    - Run history inspection
    - Auditing
    - Future drift and comparison analysis
    """

    def __init__(self, db_path: str = "runs.duckdb"):
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """
        Initialize the runs table if it does not exist.
        """
        self.conn.execute(
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

    def save_run(self, run_id: str, engine: str, summary: dict, rag_enabled: bool) -> None:
        """
        Persist a completed analysis run.

        Parameters
        ----------
        run_id : str
            Unique identifier for the analysis run.
        engine : str
            Sentiment engine used (rule_based, hf_transformer, llm, etc.).
        summary : Dict
            Aggregated sentiment summary for the run.
        rag_enabled : bool
            Enable/Disable RAG.
        """

        self.conn.execute(
            """
            INSERT INTO runs (run_id, engine, summary, rag_enabled)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, engine, summary, rag_enabled),
        )

    def _load_all_runs(self):
        """
        Load all runs from storage and deserialize JSON fields.
        """
        result = self.conn.execute(
            """
            SELECT run_id, engine, summary, rag_enabled, created_at
            FROM runs
            ORDER BY created_at DESC
            """
        ).fetchall()

        return [
            {
                "run_id": row[0],
                "engine": row[1],
                "summary": json.loads(row[2]) if isinstance(row[2], str) else row[2],
                "rag_enabled": row[3],
                "created_at": row[4],
            }
            for row in result
        ]

    def get_run(self, run_id: str):
        """
        Load a full run by ID.
        """
        result = self.conn.execute(
            "SELECT run_id, engine, summary FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        if not result:
            return None

        return {
            "run_id": result[0],
            "engine": result[1],
            "summary": result[2],
        }
