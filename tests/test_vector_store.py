"""
Tests for the persistent Chroma-backed ``VectorStore``.

A deterministic fake embedding function keeps these offline.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
from sentiment_analyzer.rag.vector_store import COLLECTION_NAME, VectorStore


@pytest.fixture
def store(tmp_vector_path, fake_embedding) -> VectorStore:
    return VectorStore(tmp_vector_path, embedding_function=fake_embedding)


class TestVectorStore:
    def test_add_and_query_returns_most_similar_chunk(self, store):
        store.add(
            ["refunds are processed within thirty days", "shipping delays upset customers"],
            document_id="doc-1",
            namespace="ops",
            source="policy.txt",
        )
        result = store.query("how long do refunds take", top_k=1, namespace="ops")
        assert result["documents"][0] == ["refunds are processed within thirty days"]
        assert result["metadatas"][0][0]["source"] == "policy.txt"

    def test_add_empty_list_is_noop(self, store):
        assert store.add([]) == 0

    def test_namespace_filter(self, store):
        store.add(["alpha text here"], document_id="a", namespace="one")
        store.add(["alpha text here"], document_id="b", namespace="two")
        result = store.query("alpha", top_k=5, namespace="two")
        assert {m["document_id"] for m in result["metadatas"][0]} == {"b"}

    def test_list_search_and_delete(self, store):
        store.add(["one", "two"], document_id="doc-a", namespace="n1", source="a.txt")
        store.add(["three"], document_id="doc-b", namespace="n2", source="b.md")

        listed = {d["document_id"] for d in store.list_documents()}
        assert listed == {"doc-a", "doc-b"}
        assert [d["document_id"] for d in store.search_documents(namespace="n2")] == ["doc-b"]
        assert [d["document_id"] for d in store.search_documents(source_contains="a.t")] == [
            "doc-a"
        ]

        store.delete_document("doc-a")
        assert {d["document_id"] for d in store.list_documents()} == {"doc-b"}

    def test_data_survives_a_new_process(self, store, tmp_vector_path):
        """Regression for BUG-1: the old ephemeral client wrote nothing to disk."""
        store.add(["persist me please"], document_id="durable", source="d.txt")

        script = textwrap.dedent(
            f"""
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path={tmp_vector_path!r}, settings=Settings(anonymized_telemetry=False)
            )
            print(client.get_collection({COLLECTION_NAME!r}).count())
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip().splitlines()[-1] == "1"
