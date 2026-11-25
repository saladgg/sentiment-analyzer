"""
Persistent vector store abstraction.

Responsibilities:
- Store embedded chunks on disk (survives restarts)
- Support similarity search
- Apply namespace filtering
- Enable document-level deletion & listing
"""

import uuid
from typing import Any

import chromadb
from chromadb.config import Settings

COLLECTION_NAME = "rag_context"

# Every client for a given path must be created with identical settings,
# otherwise Chroma refuses to open a second client in the same process.
_CLIENT_SETTINGS = Settings(anonymized_telemetry=False)


class VectorStore:
    def __init__(self, path: str, embedding_function: Any | None = None):
        """
        Open (or create) a persistent Chroma store at ``path``.

        ``path`` is a directory. It must not point at a file, e.g. the
        DuckDB runs database.

        ``embedding_function`` defaults to Chroma's built-in ONNX
        all-MiniLM-L6-v2 embedder; tests inject a deterministic stub.
        """
        self.path = path
        self.client = chromadb.PersistentClient(path=path, settings=_CLIENT_SETTINGS)

        kwargs: dict[str, Any] = {"name": COLLECTION_NAME}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self.collection = self.client.get_or_create_collection(**kwargs)

    # ------------------------------------------------------------------
    # WRITE PATH
    # ------------------------------------------------------------------
    def add(
        self,
        texts: list[str],
        *,
        document_id: str | None = None,
        namespace: str | None = None,
        source: str | None = None,
    ) -> int:
        """
        Add text chunks to the vector store.

        Args:
            texts: List of text chunks to embed & store
            document_id: Stable document identifier
            namespace: Logical grouping label (not an access boundary)
            source: Human-readable source (filename, URL)

        Returns:
            Number of chunks added
        """
        if not texts:
            return 0

        document_id = document_id or str(uuid.uuid4())

        ids = [f"{document_id}:{i}" for i in range(len(texts))]

        metadatas: list[dict[str, str | int]] = [
            {
                "document_id": document_id,
                "namespace": namespace or "",
                "source": source or "",
                "chunk_index": i,
            }
            for i in range(len(texts))
        ]

        self.collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

        return len(texts)

    # ------------------------------------------------------------------
    # READ PATH
    # ------------------------------------------------------------------
    def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        namespace: str | None = None,
    ):
        """
        Retrieve similar documents from the vector store.
        """
        where: dict[str, str] | None = {"namespace": namespace} if namespace else None

        return self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # MAINTENANCE
    # ------------------------------------------------------------------
    def delete_document(self, document_id: str):
        """
        Delete all chunks belonging to a document.
        """
        self.collection.delete(where={"document_id": document_id})  # type: ignore[arg-type]

    def list_documents(self):
        """
        List unique documents stored in the vector DB.
        """
        return self.search_documents()

    def search_documents(
        self,
        *,
        namespace: str | None = None,
        source_contains: str | None = None,
    ):
        """
        Search documents by metadata (non-embedding search).

        This is intended for administrative & UI purposes,
        not semantic retrieval.
        """
        where = {"namespace": namespace} if namespace else None

        results = self.collection.get(
            where=where,  # type: ignore[arg-type]
            include=["metadatas"],
        )

        seen: dict[str, dict] = {}
        metadatas = results.get("metadatas") or []
        for meta in metadatas:
            doc_id = meta.get("document_id")
            # Chroma metadata filters have no substring operator; filter here.
            if source_contains and source_contains not in str(meta.get("source") or ""):
                continue
            if doc_id and str(doc_id) not in seen:
                seen[str(doc_id)] = {
                    "document_id": doc_id,
                    "namespace": meta.get("namespace"),
                    "source": meta.get("source"),
                }

        return list(seen.values())
