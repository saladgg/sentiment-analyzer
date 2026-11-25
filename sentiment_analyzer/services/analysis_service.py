"""
End-to-end analysis orchestration.

This service wires together:
- Ingestion (adapters + input limits)
- Processing (normalization, text-field detection, validation)
- Optional RAG enrichment (only for engines that read context)
- Batched sentiment inference
- Analytics (summary, anomalies, explanations, optional time trend)
- Persistence (run metadata + per-record results)

It is constructed once per process (see ``main.lifespan``) and is the
single entry point used by the API.
"""

import hashlib
import json
import time
import uuid
from typing import Any

from sentiment_analyzer.analytics.aggregation import aggregate
from sentiment_analyzer.analytics.anomalies import detect_anomalies
from sentiment_analyzer.analytics.explainability import explain_record
from sentiment_analyzer.analytics.time_aggregation import Granularity, aggregate_by_time
from sentiment_analyzer.core.config import Settings
from sentiment_analyzer.core.errors import InputTooLargeError, InvalidInputError
from sentiment_analyzer.core.logging import logger
from sentiment_analyzer.core.version import app_version
from sentiment_analyzer.ingestion.json_adapter import JSONAdapter
from sentiment_analyzer.ingestion.text_adapter import TEXT_FIELD, TextAdapter
from sentiment_analyzer.processing.field_detector import detect_text_fields
from sentiment_analyzer.processing.normalizer import normalize_record
from sentiment_analyzer.processing.validator import validate_record
from sentiment_analyzer.rag.context_enrichment import RAGContextEnricher
from sentiment_analyzer.rag.models import RAGResult
from sentiment_analyzer.sentiment.base import SentimentEngine, SentimentResult
from sentiment_analyzer.sentiment.registry import EngineRegistry
from sentiment_analyzer.storage.runs import RunRecord, RunStore


def fingerprint(records: list[dict[str, Any]]) -> str:
    """Stable SHA-256 of the input records, used to match runs over the same data."""
    canonical = json.dumps(records, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _error_message(exc: Exception) -> str:
    detail = getattr(exc, "detail", None) or str(exc) or type(exc).__name__
    return f"{type(exc).__name__}: {detail}"[:500]


class AnalysisService:
    """Orchestrates end-to-end sentiment analysis from ingestion to persistence."""

    def __init__(
        self,
        settings: Settings,
        run_store: RunStore,
        engines: EngineRegistry,
        enricher: RAGContextEnricher | None = None,
    ):
        self.settings = settings
        self.run_store = run_store
        self.engines = engines
        self.enricher = enricher

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def _load_records(self, data: Any, source_type: str) -> list[dict[str, Any]]:
        if source_type == "text":
            records = TextAdapter().load(data)
        else:
            try:
                records = JSONAdapter().load(data)
            except ValueError as exc:
                raise InvalidInputError(str(exc)) from exc
        if len(records) > self.settings.max_records:
            raise InputTooLargeError(
                f"Input has more than {self.settings.max_records} records "
                "(ASA_MAX_RECORDS); split it into smaller batches"
            )
        return records

    def _resolve_text_fields(
        self, records: list[dict[str, Any]], source_type: str, text_fields: list[str] | None
    ) -> list[str]:
        if source_type == "text":
            return [TEXT_FIELD]
        if text_fields:
            present = {str(k) for r in records for k in r}
            missing = [f for f in text_fields if f not in present]
            if missing:
                raise InvalidInputError(
                    f"text_fields not found in the input: {missing}. "
                    f"Available fields: {sorted(present)}"
                )
            return list(text_fields)
        detected = detect_text_fields(records)
        if records and not detected:
            fields = sorted({str(k) for r in records for k in r})
            raise InvalidInputError(
                "No free-text fields were detected. Pass text_fields explicitly to choose "
                f"which columns to analyze. Available fields: {fields}"
            )
        return detected

    def _truncate(self, text: str) -> tuple[str, bool]:
        limit = self.settings.max_text_chars
        return (text[:limit], True) if len(text) > limit else (text, False)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        data: Any,
        source_type: str,
        sentiment_engine: str | None = None,
        *,
        text_fields: list[str] | None = None,
        use_rag: bool | None = None,
        timestamp_field: str | None = None,
        granularity: Granularity = "day",
    ) -> dict[str, Any]:
        """Execute a full analysis pipeline, persist it, and return the results."""
        started = time.perf_counter()
        engine_name = str(sentiment_engine or self.settings.default_sentiment_engine)
        engine = self.engines.get(engine_name)

        raw_records = self._load_records(data, source_type)
        if engine.is_paid and len(raw_records) > self.settings.max_llm_records:
            raise InputTooLargeError(
                f"The {engine_name} engine accepts at most {self.settings.max_llm_records} "
                "records per run (ASA_MAX_LLM_RECORDS)"
            )

        fields = self._resolve_text_fields(raw_records, source_type, text_fields)
        if timestamp_field and raw_records and all(timestamp_field not in r for r in raw_records):
            raise InvalidInputError(f"timestamp_field {timestamp_field!r} not found in the input")

        run_id = str(uuid.uuid4())
        logger.info("analysis_started", run_id=run_id, engine=engine_name, records=len(raw_records))

        # Normalize and keep only records with text to analyze.
        kept: list[tuple[int, dict[str, str], str, bool]] = []
        for idx, record in enumerate(raw_records):
            normalized = normalize_record(record)
            if not validate_record(normalized, fields):
                continue
            text, truncated = self._truncate(
                " ".join(normalized[f] for f in fields if normalized.get(f))
            )
            kept.append((idx, normalized, text, truncated))
        texts = [text for _, _, text, _ in kept]

        # Optional RAG enrichment: one batched query, only for engines that use it.
        rag_requested = self.settings.rag_enabled if use_rag is None else use_rag
        if use_rag and self.enricher is None:
            raise InvalidInputError("RAG was requested but is disabled (ASA_RAG_ENABLED=false)")
        rag_active = bool(rag_requested and self.enricher and engine.uses_context)
        rag_results: list[RAGResult | None] = (
            list(self.enricher.enrich_batch(texts))
            if rag_active and self.enricher and texts
            else [None] * len(texts)
        )
        contexts = [r.context if r and r.chunks else None for r in rag_results]

        outputs = engine.analyze_batch(texts, contexts) if texts else []
        field_scores = self._field_contributions(engine, kept, fields)

        records: list[dict[str, Any]] = []
        for (idx, _, _, truncated), output, rag in zip(kept, outputs, rag_results, strict=True):
            record = self._build_record(idx, output, rag, rag_active, truncated, field_scores)
            if len(fields) == 1 and not record["error"]:
                record["contributing_fields"] = {fields[0]: record["score"]}
            records.append(record)

        summary = aggregate(records)
        anomalies = detect_anomalies(records)
        explanations = [explain_record(r) for r in records]

        trend = None
        if timestamp_field:
            timestamps = [raw_records[idx].get(timestamp_field) for idx, *_ in kept]
            trend = aggregate_by_time(records, timestamps, granularity)

        rag_used = any(r is not None and r.chunks for r in rag_results)
        rag_sources = {s for r in rag_results if r for s in r.sources}
        input_sha256 = fingerprint(raw_records)
        duration_ms = int((time.perf_counter() - started) * 1000)
        failed = summary["failed_records"]

        config = {
            **engine.config_snapshot(),
            "app_version": app_version(),
            "text_fields": fields,
            "rag": {
                "requested": bool(rag_requested),
                "active": rag_active,
                "top_k": self.settings.rag_top_k,
                "namespace": self.settings.rag_namespace,
            },
            "max_text_chars": self.settings.max_text_chars,
        }
        self.run_store.save_run(
            RunRecord(
                run_id=run_id,
                engine=engine_name,
                summary=summary,
                rag_enabled=rag_active,
                source_type=source_type,
                input_sha256=input_sha256,
                config=config,
                records_in=len(raw_records),
                records_skipped=len(raw_records) - len(kept),
                records_failed=failed,
                duration_ms=duration_ms,
                status="completed_with_errors" if failed else "completed",
            ),
            records=records,
        )

        logger.info(
            "analysis_completed",
            run_id=run_id,
            engine=engine_name,
            records=len(records),
            failed=failed,
            duration_ms=duration_ms,
        )

        rag_metadata = None
        if self.settings.rag_enabled:
            rag_metadata = {
                "enabled": rag_active,
                "used": rag_used,
                "source_count": len(rag_sources),
            }

        return {
            "run_id": run_id,
            "engine": engine_name,
            "input_sha256": input_sha256,
            "text_fields": fields,
            "summary": summary,
            "records": records,
            "anomalies": anomalies,
            "explanations": explanations,
            "rag": rag_metadata,
            "trend": trend,
        }

    def _field_contributions(
        self,
        engine: SentimentEngine,
        kept: list[tuple[int, dict[str, str], str, bool]],
        fields: list[str],
    ) -> dict[int, dict[str, float]]:
        """
        Score each text field of each record separately.

        Only for engines that are free to call (doubling paid calls is not
        worth it) and when there is more than one field; with a single
        field its contribution is the record score itself.
        """
        if engine.is_paid or len(fields) < 2:
            return {}
        jobs = [
            (idx, field, self._truncate(normalized[field])[0])
            for idx, normalized, _, _ in kept
            for field in fields
            if normalized.get(field)
        ]
        outputs = engine.analyze_batch([text for _, _, text in jobs]) if jobs else []
        scores: dict[int, dict[str, float]] = {}
        for (idx, field, _), output in zip(jobs, outputs, strict=True):
            if isinstance(output, SentimentResult):
                scores.setdefault(idx, {})[field] = round(output.polarity, 4)
        return scores

    @staticmethod
    def _build_record(
        record_id: int,
        output: SentimentResult | Exception,
        rag: RAGResult | None,
        rag_active: bool,
        truncated: bool,
        field_scores: dict[int, dict[str, float]],
    ) -> dict[str, Any]:
        rag_block = {
            "enabled": rag_active,
            "sources": rag.sources if rag else [],
            "chunks_used": len(rag.chunks) if rag else 0,
        }
        base = {
            "record_id": record_id,
            "rag": rag_block,
            "rag_chunk_ids": rag.chunk_ids if rag else [],
            "truncated": truncated,
        }
        if isinstance(output, Exception):
            logger.warning("record_failed", record_id=record_id, error=_error_message(output))
            return {
                **base,
                "sentiment": None,
                "score": None,
                "confidence": None,
                "contributing_fields": {},
                "drivers": {},
                "error": _error_message(output),
            }
        return {
            **base,
            "sentiment": output.label,
            "score": round(output.polarity, 4),
            "confidence": round(output.confidence, 4),
            "contributing_fields": field_scores.get(record_id, {}),
            "drivers": output.drivers,
            "error": None,
        }
