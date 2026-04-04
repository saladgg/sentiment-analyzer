"""
Workflow abstraction for batch and streaming sentiment analysis.

This layer allows the same analysis logic to be reused for:
- One-off batch uploads
- Streaming ingestion (Kafka, Pub/Sub, WebSockets)
"""

from collections.abc import Iterable

from sentiment_analyzer.services.analysis_service import AnalysisService


class SentimentWorkflow:
    """High-level workflow for batch and streaming sentiment analysis."""

    def __init__(self):
        self.service = AnalysisService()

    def run_batch(self, records: list[dict], **kwargs):
        """
        Run batch sentiment analysis.
        """
        return self.service.run(data=records, **kwargs)

    def run_stream(self, record_stream: Iterable[dict], **kwargs):
        """
        Process records incrementally (streaming-ready).
        """
        results = []
        for record in record_stream:
            result = self.service.run(data=[record], **kwargs)
            results.append(result)
        return results
