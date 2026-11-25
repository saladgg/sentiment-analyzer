"""
Core data structures for Retrieval-Augmented Generation (RAG).

These models define the contract between:
- Vector storage
- Retrieval
- Analysis orchestration
- API responses
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RAGChunk:
    """A single retrieved chunk with its provenance and relevance score."""

    content: str
    source: str
    score: float
    document_id: str = ""
    namespace: str = ""
    chunk_id: str = ""


@dataclass
class RAGResult:
    """Context retrieved for one text: the chunks and their joined content."""

    chunks: list[RAGChunk] = field(default_factory=list)

    @property
    def context(self) -> str:
        return "\n".join(c.content for c in self.chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks if c.source})

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]
