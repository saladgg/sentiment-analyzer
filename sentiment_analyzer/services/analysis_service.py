"""
End-to-end analysis orchestration.

This service wires together:
- Ingestion
- Processing
- Optional RAG enrichment
- Sentiment inference
- Analytics
- Persistence

This is the single entry point used by the API.
"""

import uuid

from sentiment_analyzer.analytics.aggregation import aggregate
from sentiment_analyzer.analytics.anomalies import detect_anomalies
from sentiment_analyzer.analytics.explainability import explain_record
from sentiment_analyzer.analytics.per_record import build_per_record_result
from sentiment_analyzer.api.schemas import RecordSentiment
from sentiment_analyzer.core.config import settings
from sentiment_analyzer.core.logging import logger
from sentiment_analyzer.ingestion.json_adapter import JSONAdapter
from sentiment_analyzer.processing.field_detector import detect_text_fields
from sentiment_analyzer.processing.normalizer import normalize_record
from sentiment_analyzer.processing.validator import validate_record
from sentiment_analyzer.rag.context_enrichment import RAGContextEnricher
from sentiment_analyzer.sentiment.hf_transformer import HuggingFaceSentimentEngine
from sentiment_analyzer.sentiment.llm_stub import LLMSentimentEngine
from sentiment_analyzer.sentiment.rule_based import RuleBasedSentimentEngine
from sentiment_analyzer.storage.runs import RunStore


class AnalysisService:
    """Orchestrates end-to-end sentiment analysis from ingestion to persistence."""

    def __init__(self):
        self.run_store = RunStore(settings.rag_db_path)

        self.sentiment_engines = {
            "rule_based": RuleBasedSentimentEngine(),
            "hf_transformer": HuggingFaceSentimentEngine(settings.hf_sentiment_model),
            "llm": LLMSentimentEngine(),
        }

        self.enricher = RAGContextEnricher() if settings.rag_enabled else None

    def run(self, data, source_type: str, sentiment_engine: str) -> dict:
        """Execute a full analysis pipeline and return structured results."""
        run_id = str(uuid.uuid4())
        rag_used = False
        rag_source_count = 0
        logger.info("analysis_started", run_id=run_id)

        adapter = JSONAdapter()
        raw_records = adapter.load(data)

        text_fields = detect_text_fields(raw_records)
        engine = self.sentiment_engines[sentiment_engine]

        per_record_results: list[dict] = []

        for idx, record in enumerate(raw_records):
            normalized = normalize_record(record)

            if not validate_record(normalized):
                continue

            text = " ".join(normalized[f] for f in text_fields)

            # Optional RAG enrichment (single, explicit call)
            if self.enricher:
                rag_result = self.enricher.enrich(text)
                if rag_result and rag_result.context:
                    rag_used = True
                    rag_source_count += len(rag_result.sources)
            else:
                rag_result = None

            sentiment_output = engine.analyze(
                text=text,
                context=rag_result.context if rag_result else None,
            )

            result = build_per_record_result(
                idx,
                sentiment_output,
                rag_result=rag_result,
            )

            validated = RecordSentiment(**result)
            per_record_results.append(validated.model_dump())

        summary = aggregate(per_record_results)
        anomalies = detect_anomalies(per_record_results)
        explanations = [explain_record(r) for r in per_record_results]

        self.run_store.save_run(
            run_id=run_id,
            engine=sentiment_engine,
            summary=summary,
            rag_enabled=rag_used,
        )

        logger.info("analysis_completed", run_id=run_id)

        rag_metadata = None

        if settings.rag_enabled:
            rag_metadata = {
                "enabled": True,
                "used": rag_used,
                "source_count": rag_source_count,
            }

        return {
            "run_id": run_id,
            "summary": summary,
            "records": per_record_results,
            "anomalies": anomalies,
            "explanations": explanations,
            "rag": rag_metadata,
        }
