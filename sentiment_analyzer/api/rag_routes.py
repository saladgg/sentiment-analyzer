"""
RAG document ingestion, inspection, and retrieval endpoints.

These endpoints support:
- Uploading context documents
- Previewing semantic retrieval results
- Listing and deleting stored documents
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Body, File, UploadFile

from sentiment_analyzer.core.config import settings
from sentiment_analyzer.rag.vector_store import VectorStore

router = APIRouter(prefix="/rag", tags=["RAG"])

store = VectorStore(settings.rag_db_path)


# ---------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------
@router.post("/documents")
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    namespace: str = "default",
):
    """
    Upload a document to be used as RAG context.

    The document is:
    - Read as text
    - Chunked
    - Embedded
    - Persisted in the vector store
    """
    raw = await file.read()
    content = raw.decode("utf-8")

    # Simple fixed-size chunking (can be replaced later)
    chunks = [
        content[i : i + 500] for i in range(0, len(content), 500) if content[i : i + 500].strip()
    ]

    document_id = str(uuid.uuid4())

    chunks_added = store.add(
        texts=chunks,
        document_id=document_id,
        namespace=namespace,
        source=file.filename,
    )

    return {
        "document_id": document_id,
        "filename": file.filename,
        "namespace": namespace,
        "chunks_added": chunks_added,
    }


# ---------------------------------------------------------------------
# Retrieval preview (semantic search)
# ---------------------------------------------------------------------
@router.post("/retrieve")
async def retrieve_context(payload: Annotated[dict, Body(...)]):
    """
    Preview what context would be retrieved for a given query.

    This endpoint does NOT affect analysis runs.
    It is intended for inspection and debugging.
    """
    query: str = payload["query"]
    top_k: int = payload.get("top_k", settings.rag_top_k)
    namespace: str | None = payload.get("namespace", settings.rag_namespace)

    results = store.query(
        query=query,
        top_k=top_k,
        namespace=namespace,
    )

    return [
        {
            "content": doc,
            "source": meta.get("source"),
            "document_id": meta.get("document_id"),
            "namespace": meta.get("namespace"),
            "score": 1 - distance,
        }
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        )
    ]


# ---------------------------------------------------------------------
# Document inspection
# ---------------------------------------------------------------------
@router.get("/documents")
def list_documents():
    """
    List all uploaded RAG documents.
    """
    return store.list_documents()


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """
    Delete a document and all its chunks from the vector store.
    """
    store.delete_document(document_id)
    return {"status": "deleted", "document_id": document_id}
