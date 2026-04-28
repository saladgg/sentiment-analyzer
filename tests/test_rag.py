"""
Tests for RAG result dataclasses and the standalone ``types.RAGResult``.

Note: the project currently has two ``RAGResult`` definitions —
``rag.models.RAGResult`` (used by the production pipeline) and
``rag.types.RAGResult`` (used by helper utilities). Both are exercised
here.
"""

from __future__ import annotations

from sentiment_analyzer.rag.models import RAGChunk
from sentiment_analyzer.rag.models import RAGResult as ModelsRAGResult
from sentiment_analyzer.rag.types import RAGResult as TypesRAGResult


class TestModelsRAGResult:
    def test_constructs_with_chunks(self):
        chunk = RAGChunk(content="hi", source="doc-a", score=0.42)
        result = ModelsRAGResult(
            enabled=True,
            context="hi",
            sources=["doc-a"],
            chunks=[chunk],
        )
        assert result.enabled is True
        assert result.chunks[0].source == "doc-a"


class TestTypesRAGResult:
    def test_empty_factory_returns_no_context(self):
        result = TypesRAGResult.empty()
        assert result.context is None
        assert result.sources == []

    def test_from_documents_concatenates_content(self):
        class _Doc:
            def __init__(self, content, meta):
                self.page_content = content
                self.metadata = meta

        docs = [
            _Doc("alpha", {"source": "a"}),
            _Doc("beta", {"source": "b"}),
        ]
        result = TypesRAGResult.from_documents(docs)
        assert result.context == "alpha beta"
        assert result.sources == [{"source": "a"}, {"source": "b"}]

    def test_from_documents_with_empty_list(self):
        result = TypesRAGResult.from_documents([])
        assert result.context == ""
        assert result.sources == []
