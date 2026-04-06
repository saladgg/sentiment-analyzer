"""
RAG-based context enrichment for sentiment analysis.
This module provides OPTIONAL, read-only contextual enrichment
using a vector store. It does NOT perform sentiment inference.
Retrieves relevant context for a given input text.

This component is:
- Optional
- Deterministic
- Transparent
"""

from sentiment_analyzer.core.config import settings

from .models import RAGChunk, RAGResult
from .vector_store import VectorStore


class RAGContextEnricher:
    """Retrieves relevant context from the vector store for sentiment enrichment."""

    def __init__(self):
        self.enabled = settings.rag_enabled
        self.store = VectorStore(settings.rag_db_path)

    def enrich(self, text: str) -> RAGResult | None:
        """Retrieve and assemble RAG context for the given text."""
        if not self.enabled:
            return None

        # results = self.store.query(text)
        results = self.store.query(
            text,
            top_k=settings.rag_top_k,
            namespace=settings.rag_namespace,
        )

        chunks = []
        context_parts = []
        sources = set()

        for doc, meta, score in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        ):
            chunks.append(
                RAGChunk(
                    content=doc,
                    source=meta["source"],
                    score=1 - score,
                )
            )
            context_parts.append(doc)
            sources.add(meta["source"])

        return RAGResult(
            enabled=True,
            context="\n".join(context_parts),
            sources=list(sources),
            chunks=chunks,
        )
