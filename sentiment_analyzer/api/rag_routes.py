"""
RAG document ingestion, inspection, and retrieval endpoints.

These endpoints support:
- Uploading context documents
- Previewing semantic retrieval results
- Listing and deleting stored documents

Uploaded documents are trusted context for the LLM engine, so these
routes sit behind the same API-key check as everything else. A
namespace is a grouping label, not an access boundary.
"""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Query, UploadFile

from sentiment_analyzer.api.deps import get_settings, get_vector_store, read_upload
from sentiment_analyzer.api.schemas import (
    NAMESPACE_PATTERN,
    RagDeleteResponse,
    RagDocument,
    RagUploadResponse,
    RetrieveHit,
    RetrieveRequest,
)
from sentiment_analyzer.core.config import Settings
from sentiment_analyzer.core.errors import InvalidInputError, UnsupportedInputError
from sentiment_analyzer.rag.chunking import chunk_text
from sentiment_analyzer.rag.vector_store import VectorStore

router = APIRouter(prefix="/rag", tags=["RAG"])

StoreDep = Annotated[VectorStore, Depends(get_vector_store)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# ---------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------
@router.post("/documents", response_model=RagUploadResponse)
def upload_document(
    file: Annotated[UploadFile, File(...)],
    store: StoreDep,
    settings: SettingsDep,
    namespace: Annotated[str, Query(pattern=NAMESPACE_PATTERN)] = "default",
) -> dict[str, Any]:
    """
    Upload a UTF-8 text document to be used as RAG context.

    The document is chunked on paragraph/sentence boundaries, embedded,
    and persisted in the vector store.
    """
    raw = read_upload(file, settings.max_upload_bytes)
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedInputError("RAG documents must be UTF-8 text") from exc

    chunks = chunk_text(content, settings.rag_chunk_size, settings.rag_chunk_overlap)
    if not chunks:
        raise InvalidInputError("Document contains no text")

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
@router.post("/retrieve", response_model=list[RetrieveHit])
def retrieve_context(
    payload: RetrieveRequest, store: StoreDep, settings: SettingsDep
) -> list[dict[str, Any]]:
    """
    Preview what context would be retrieved for a given query.

    This endpoint does NOT affect analysis runs.
    It is intended for inspection and debugging.
    """
    hits = store.query(
        payload.query,
        top_k=payload.top_k or settings.rag_top_k,
        namespace=payload.namespace or settings.rag_namespace,
    )
    return [
        {
            "content": hit.content,
            "source": hit.source,
            "document_id": hit.document_id,
            "namespace": hit.namespace,
            "score": hit.score,
        }
        for hit in hits
    ]


# ---------------------------------------------------------------------
# Document inspection
# ---------------------------------------------------------------------
@router.get("/documents", response_model=list[RagDocument])
def list_documents(
    store: StoreDep,
    namespace: Annotated[str | None, Query(pattern=NAMESPACE_PATTERN)] = None,
) -> list[dict[str, Any]]:
    """List uploaded RAG documents, optionally within one namespace."""
    return store.search_documents(namespace=namespace)


@router.delete("/documents/{document_id}", response_model=RagDeleteResponse)
def delete_document(document_id: str, store: StoreDep) -> dict[str, Any]:
    """Delete a document and all its chunks from the vector store."""
    store.delete_document(document_id)
    return {"status": "deleted", "document_id": document_id}
