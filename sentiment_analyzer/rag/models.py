"""
Core data structures for Retrieval-Augmented Generation (RAG).

These models define the contract between:
- Vector storage
- Retrieval
- Analysis orchestration
- API responses
"""

from dataclasses import dataclass


@dataclass
class RAGChunk:
    """A single chunk of retrieved context with its source and relevance score."""

    content: str
    source: str
    score: float


@dataclass
class RAGResult:
    """Aggregated RAG retrieval output including all chunks and assembled context."""

    enabled: bool
    context: str
    sources: list[str]
    chunks: list[RAGChunk]
