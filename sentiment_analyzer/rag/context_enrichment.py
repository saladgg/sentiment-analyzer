"""
RAG-based context enrichment for sentiment analysis.

This module provides OPTIONAL, read-only contextual enrichment
using a vector store. It does NOT perform sentiment inference.

This component is:
- Optional (only used for engines that read context)
- Deterministic for a given store
- Transparent (chunk ids are recorded per record)
"""

from sentiment_analyzer.rag.models import RAGResult
from sentiment_analyzer.rag.vector_store import VectorStore


class RAGContextEnricher:
    """Retrieves relevant context from the vector store for sentiment enrichment."""

    def __init__(self, store: VectorStore, *, top_k: int = 5, namespace: str | None = None):
        self.store = store
        self.top_k = top_k
        self.namespace = namespace

    def enrich(self, text: str) -> RAGResult:
        """Retrieve context for a single text."""
        return self.enrich_batch([text])[0]

    def enrich_batch(self, texts: list[str]) -> list[RAGResult]:
        """Retrieve context for many texts with a single vector-store query."""
        hits = self.store.query_many(texts, top_k=self.top_k, namespace=self.namespace)
        return [RAGResult(chunks=chunks) for chunks in hits]
