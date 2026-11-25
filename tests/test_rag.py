"""
Tests for RAG models, chunking, enrichment, the bootstrap script, and the
RAG HTTP endpoints.
"""

from __future__ import annotations

import io
import itertools

import pytest
from fastapi.testclient import TestClient

from sentiment_analyzer.rag.chunking import chunk_text
from sentiment_analyzer.rag.context_enrichment import RAGContextEnricher
from sentiment_analyzer.rag.models import RAGChunk, RAGResult
from sentiment_analyzer.rag.vector_store import VectorStore
from sentiment_analyzer.scripts import bootstrap_rag_store

API = "/api/v1"


class TestRAGResult:
    def test_derived_fields(self):
        result = RAGResult(
            chunks=[
                RAGChunk(content="a", source="s2", score=0.4, chunk_id="d:0"),
                RAGChunk(content="b", source="s1", score=0.3, chunk_id="d:1"),
                RAGChunk(content="c", source="s1", score=0.2, chunk_id="e:0"),
            ]
        )
        assert result.context == "a\nb\nc"
        assert result.sources == ["s1", "s2"]
        assert result.chunk_ids == ["d:0", "d:1", "e:0"]
        assert RAGResult().context == ""


class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert chunk_text("Hello world.", 100, 10) == ["Hello world."]

    def test_respects_size_and_never_splits_words(self):
        text = " ".join(f"word{i}" for i in range(200))
        chunks = chunk_text(text, 50, 10)
        assert all(len(c) <= 50 for c in chunks)
        words = set(text.split())
        assert all(w in words for c in chunks for w in c.split())

    def test_consecutive_chunks_overlap(self):
        text = ". ".join(f"Sentence number {i} is here" for i in range(20)) + "."
        chunks = chunk_text(text, 80, 30)
        assert len(chunks) > 1
        for prev, nxt in itertools.pairwise(chunks):
            assert nxt.split(" ")[0] in prev.split(" ")

    def test_paragraphs_and_long_words(self):
        text = "First para.\n\nSecond para here.\n\n" + "x" * 25
        chunks = chunk_text(text, 20, 0)
        assert chunks[0] == "First para."
        assert all(len(c) <= 20 for c in chunks)
        assert "".join(chunks).count("x") == 25

    def test_empty_text(self):
        assert chunk_text("   \n\n ", 50, 5) == []

    def test_overlap_must_be_smaller_than_size(self):
        with pytest.raises(ValueError):
            chunk_text("x", 10, 10)


class TestEnricher:
    def test_enrich_batch_uses_one_query(self, tmp_vector_path, fake_embedding, monkeypatch):
        store = VectorStore(tmp_vector_path, embedding_function=fake_embedding)
        store.add(["refunds take thirty days"], document_id="d", namespace="ns", source="p")
        calls = []
        original = store.query_many

        def _spy(queries, **kwargs):
            calls.append(queries)
            return original(queries, **kwargs)

        monkeypatch.setattr(store, "query_many", _spy)
        enricher = RAGContextEnricher(store, top_k=1, namespace="ns")
        results = enricher.enrich_batch(["refunds please", "thirty days?"])

        assert len(calls) == 1
        assert [r.sources for r in results] == [["p"], ["p"]]
        assert enricher.enrich("refunds").chunks[0].document_id == "d"

    def test_empty_store_returns_no_context(self, tmp_vector_path, fake_embedding):
        store = VectorStore(tmp_vector_path, embedding_function=fake_embedding)
        assert RAGContextEnricher(store).enrich("x").chunks == []
        assert store.query_many([]) == []


class TestBootstrap:
    def test_is_idempotent(self, tmp_vector_path, fake_embedding):
        store = VectorStore(tmp_vector_path, embedding_function=fake_embedding)
        bootstrap_rag_store.bootstrap(store, "default")
        bootstrap_rag_store.bootstrap(store, "default")
        assert store.heartbeat() == len(bootstrap_rag_store.STARTER_DOCUMENTS)

    def test_main_writes_to_given_path(self, tmp_vector_path, monkeypatch, capsys, fake_embedding):
        real = bootstrap_rag_store.VectorStore
        monkeypatch.setattr(
            bootstrap_rag_store,
            "VectorStore",
            lambda path: real(path, embedding_function=fake_embedding),
        )
        bootstrap_rag_store.main(["--path", tmp_vector_path, "--namespace", "seed"])
        assert "Loaded 2 chunks" in capsys.readouterr().out
        docs = real(tmp_vector_path, embedding_function=fake_embedding).list_documents()
        assert {d["namespace"] for d in docs} == {"seed"}


class TestRAGEndpoints:
    @pytest.fixture
    def rag_client(self, make_app):
        with TestClient(make_app(rag_chunk_size=60, rag_chunk_overlap=10)) as c:
            yield c

    def _upload(self, client, content: bytes, name="doc.txt", namespace="default"):
        return client.post(
            f"{API}/rag/documents",
            files={"file": (name, io.BytesIO(content), "text/plain")},
            params={"namespace": namespace},
        )

    def test_upload_list_retrieve_delete(self, rag_client):
        text = b"Refunds are processed within thirty days. Shipping delays upset customers a lot."
        uploaded = self._upload(rag_client, text, namespace="ops").json()
        assert uploaded["chunks_added"] == 2
        assert uploaded["namespace"] == "ops"

        docs = rag_client.get(f"{API}/rag/documents").json()
        assert docs == [
            {"document_id": uploaded["document_id"], "namespace": "ops", "source": "doc.txt"}
        ]
        assert rag_client.get(f"{API}/rag/documents", params={"namespace": "x"}).json() == []

        hits = rag_client.post(
            f"{API}/rag/retrieve",
            json={"query": "refunds processed", "top_k": 1, "namespace": "ops"},
        ).json()
        assert len(hits) == 1
        assert hits[0]["content"].startswith("Refunds")
        assert hits[0]["document_id"] == uploaded["document_id"]

        deleted = rag_client.delete(f"{API}/rag/documents/{uploaded['document_id']}").json()
        assert deleted["status"] == "deleted"
        assert rag_client.get(f"{API}/rag/documents").json() == []

    def test_rejects_non_utf8(self, rag_client):
        assert self._upload(rag_client, "héllo".encode("utf-16")).status_code == 415

    def test_rejects_empty_document(self, rag_client):
        assert self._upload(rag_client, b"   \n ").status_code == 422

    def test_rejects_bad_namespace(self, rag_client):
        assert self._upload(rag_client, b"text", namespace="../etc").status_code == 422

    def test_upload_size_limit(self, make_app):
        with TestClient(make_app(max_upload_bytes=10)) as c:
            assert self._upload(c, b"x" * 50).status_code == 413

    @pytest.mark.parametrize(
        "payload",
        [{}, {"query": ""}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 1000}],
    )
    def test_retrieve_validation(self, rag_client, payload):
        assert rag_client.post(f"{API}/rag/retrieve", json=payload).status_code == 422
