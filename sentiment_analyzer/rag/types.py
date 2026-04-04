"""
Shared type definitions for RAG retrieval results.
"""

from dataclasses import dataclass


@dataclass
class RAGResult:
    """Container for RAG retrieval results including context and sources."""

    context: str | None
    sources: list[dict]

    @staticmethod
    def empty():
        """Return an empty RAGResult with no context or sources."""
        return RAGResult(context=None, sources=[])

    @staticmethod
    def from_documents(docs):
        """Build a RAGResult from a list of LangChain-style documents."""
        return RAGResult(
            context=" ".join(d.page_content for d in docs),
            sources=[d.metadata for d in docs],
        )
