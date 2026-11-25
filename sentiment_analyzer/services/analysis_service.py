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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from sentiment_analyzer.analytics.aggregation import aggregate
from sentiment_analyzer.analytics.anomalies import detect_anomalies
from sentiment_analyzer.analytics.explainability import explain_record
from sentiment_analyzer.analytics.time_aggregation import Granularity, aggregate_by_time
from sentiment_analyzer.core.config import Settings
from sentiment_analyzer.core.errors import InputTooLargeError, InvalidInputError
from sentiment_analyzer.core.logging import logger
from sentiment_analyzer.core.metrics import Metrics
from sentiment_analyzer.core.version import app_version
from sentiment_analyzer.ingestion.json_adapter import JSONAdapter
from sentiment_analyzer.ingestion.text_adapter import TEXT_FIELD, TextAdapter
from sentiment_analyzer.processing.field_detector import detect_text_fields
from sentiment_analyzer.processing.normalizer import normalize_record
from sentiment_analyzer.processing.validator import validate_record
from sentiment_analyzer.rag.context_enrichment import RAGContextEnricher
from sentiment_analyzer.rag.models import RAGResult
from sentiment_analyzer.sentiment.base import BatchItem, SentimentEngine, SentimentResult
from sentiment_analyzer.sentiment.registry import EngineRegistry
from sentiment_analyzer.storage.runs import RunRecord, RunStore


@dataclass
class PreparedInput:
    """A validated request: parsed records, chosen engine and text fields."""

    engine_name: str
    engine: SentimentEngine
    source_type: str
    raw_records: list[dict[str, Any]]
    fields: list[str]
    timestamp_field: str | None = None


# Records per engine/RAG call: bounds memory per call and sets progress granularity.
CHUNK_SIZE = 256


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
        metrics: Metrics | None = None,
    ):
        self.settings = settings
        self.run_store = run_store
        self.engines = engines
        self.enricher = enricher
        self.metrics = metrics

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def _load_records(self, data: Any, source_type: str, max_records: int) -> list[dict[str, Any]]:
        if source_type == "text":
            records = TextAdapter().load(data)
        else:
            try:
                records = JSONAdapter().load(data)
            except ValueError as exc:
                raise InvalidInputError(str(exc)) from exc
        if len(records) > max_records:
            hint = (
                "submit it as a background job (POST /api/v1/jobs/analyze)"
                if max_records == self.settings.max_records
                else "split it into smaller batches"
            )
            raise InputTooLargeError(f"Input has more than {max_records} records; {hint}")
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

    def check_record_limits(self, count: int, engine_name: str, *, background: bool) -> None:
        """Raise 413 if ``count`` records exceed the limits for this engine and mode."""
        engine_cls = self.engines.engine_class(engine_name)
        if background:
            total, paid = self.settings.max_job_records, self.settings.max_llm_job_records
            total_var, paid_var = "ASA_MAX_JOB_RECORDS", "ASA_MAX_LLM_JOB_RECORDS"
        else:
            total, paid = self.settings.max_records, self.settings.max_llm_records
            total_var, paid_var = "ASA_MAX_RECORDS", "ASA_MAX_LLM_RECORDS"
        if count > total:
            raise InputTooLargeError(f"Input has more than {total} records ({total_var})")
        if engine_cls.is_paid and count > paid:
            hint = "" if background else "; larger runs can be submitted as background jobs"
            raise InputTooLargeError(
                f"The {engine_name} engine accepts at most {paid} records per run "
                f"({paid_var}){hint}"
            )

    @contextmanager
    def _timer(self, kind: str, engine_name: str = "") -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics:
                elapsed = time.perf_counter() - started
                if kind == "rag":
                    self.metrics.rag_latency.observe(elapsed)
                else:
                    self.metrics.inference_latency.labels(engine_name).observe(elapsed)

    def _usage_totals(self, outputs: list[BatchItem]) -> dict[str, Any] | None:
        """Sum per-record usage (paid engines); also feeds the LLM metrics."""
        usages = [
            o.usage for o in outputs if isinstance(o, SentimentResult) and o.usage is not None
        ]
        if not usages:
            return None
        billed = [u for u in usages if not u.cached]
        costs = [u.cost_usd for u in billed]
        totals: dict[str, Any] = {
            "requests": len(billed),
            "cached": len(usages) - len(billed),
            "prompt_tokens": sum(u.prompt_tokens for u in billed),
            "completion_tokens": sum(u.completion_tokens for u in billed),
            "cost_usd": round(sum(c for c in costs if c is not None), 6),
            # False when the provider's pricing is unknown for some requests.
            "cost_complete": all(c is not None for c in costs),
        }
        if self.metrics:
            self.metrics.llm_calls.labels("api").inc(totals["requests"])
            self.metrics.llm_calls.labels("cache").inc(totals["cached"])
            self.metrics.llm_tokens.labels("prompt").inc(totals["prompt_tokens"])
            self.metrics.llm_tokens.labels("completion").inc(totals["completion_tokens"])
            self.metrics.llm_cost.inc(totals["cost_usd"])
        return totals

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
        background: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a full analysis pipeline, persist it, and return the results.

        ``background=True`` applies the larger job limits
        (``ASA_MAX_JOB_RECORDS`` / ``ASA_MAX_LLM_JOB_RECORDS``).
        ``progress(done, total)`` is called after each chunk of records.
        """
        started = time.perf_counter()
        prepared = self.prepare(
            data,
            source_type,
            sentiment_engine,
            text_fields=text_fields,
            timestamp_field=timestamp_field,
            background=background,
        )
        return self.run_prepared(
            prepared,
            use_rag=use_rag,
            granularity=granularity,
            progress=progress,
            started=started,
        )

    def prepare(
        self,
        data: Any,
        source_type: str,
        sentiment_engine: str | None = None,
        *,
        text_fields: list[str] | None = None,
        timestamp_field: str | None = None,
        background: bool = False,
    ) -> PreparedInput:
        """
        Parse and validate a request without running it.

        Raises the same 4xx errors ``run`` would (unknown engine, limits,
        missing fields), so background jobs can be rejected up front.
        """
        engine_name = str(sentiment_engine or self.settings.default_sentiment_engine)
        engine = self.engines.get(engine_name)
        max_records = self.settings.max_job_records if background else self.settings.max_records
        raw_records = self._load_records(data, source_type, max_records)
        self.check_record_limits(len(raw_records), engine_name, background=background)

        fields = self._resolve_text_fields(raw_records, source_type, text_fields)
        if timestamp_field and raw_records and all(timestamp_field not in r for r in raw_records):
            raise InvalidInputError(f"timestamp_field {timestamp_field!r} not found in the input")
        return PreparedInput(
            engine_name=engine_name,
            engine=engine,
            source_type=source_type,
            raw_records=raw_records,
            fields=fields,
            timestamp_field=timestamp_field,
        )

    def run_prepared(
        self,
        prepared: PreparedInput,
        *,
        use_rag: bool | None = None,
        granularity: Granularity = "day",
        progress: Callable[[int, int], None] | None = None,
        started: float | None = None,
    ) -> dict[str, Any]:
        """Run the pipeline on input already validated by :meth:`prepare`."""
        started = time.perf_counter() if started is None else started
        engine_name, engine = prepared.engine_name, prepared.engine
        source_type, raw_records = prepared.source_type, prepared.raw_records
        fields, timestamp_field = prepared.fields, prepared.timestamp_field

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

        # Optional RAG enrichment, only for engines that use it.
        rag_requested = self.settings.rag_enabled if use_rag is None else use_rag
        if use_rag and self.enricher is None:
            raise InvalidInputError("RAG was requested but is disabled (ASA_RAG_ENABLED=false)")
        rag_active = bool(rag_requested and self.enricher and engine.uses_context)

        # Inference in chunks: bounded memory per call and progress reporting.
        rag_results: list[RAGResult | None] = []
        outputs: list[BatchItem] = []
        if progress:
            progress(0, len(texts))
        for start in range(0, len(texts), CHUNK_SIZE):
            chunk = texts[start : start + CHUNK_SIZE]
            chunk_rag: list[RAGResult | None] = [None] * len(chunk)
            if rag_active and self.enricher:
                with self._timer("rag"):
                    chunk_rag = list(self.enricher.enrich_batch(chunk))
            contexts = [r.context if r and r.chunks else None for r in chunk_rag]
            with self._timer("engine", engine_name):
                outputs.extend(engine.analyze_batch(chunk, contexts))
            rag_results.extend(chunk_rag)
            if progress:
                progress(len(outputs), len(texts))

        field_scores = self._field_contributions(engine, kept, fields)

        records: list[dict[str, Any]] = []
        for (idx, _, _, truncated), output, rag in zip(kept, outputs, rag_results, strict=True):
            record = self._build_record(idx, output, rag, rag_active, truncated, field_scores)
            if len(fields) == 1 and not record["error"]:
                record["contributing_fields"] = {fields[0]: record["score"]}
            records.append(record)

        summary = aggregate(records)
        usage = self._usage_totals(outputs)
        if usage is not None:
            summary["llm_usage"] = usage
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

        status = "completed_with_errors" if failed else "completed"
        if self.metrics:
            self.metrics.runs.labels(engine_name, status).inc()
            self.metrics.records.labels(engine_name, "ok").inc(len(records) - failed)
            self.metrics.records.labels(engine_name, "failed").inc(failed)

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
