"""
RAG Management page — upload context documents, browse the vector store,
and preview semantic retrieval results.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from api_client import (
    delete_rag_document,
    list_rag_documents,
    rag_retrieve,
    upload_rag_document,
)

st.title("RAG Management")
st.markdown(
    "Upload context documents to enrich sentiment analysis with domain knowledge. "
    "The platform chunks, embeds, and stores them in a local Chroma vector store."
)

# ── upload ──────────────────────────────────────────────────────────
st.subheader("Upload Document")

col_file, col_ns = st.columns([3, 1])
with col_file:
    doc_file = st.file_uploader(
        "Choose a text file",
        type=["txt", "md", "csv", "json"],
        help="Plain-text documents work best. The file is chunked into 500-char segments.",
    )
with col_ns:
    namespace = st.text_input("Namespace", value="default")

if st.button("Upload", type="primary", disabled=doc_file is None):
    with st.spinner("Uploading and embedding..."):
        try:
            res = upload_rag_document(doc_file, namespace=namespace)
            st.success(
                f"Uploaded **{res['filename']}** — "
                f"{res['chunks_added']} chunks stored (doc `{res['document_id'][:8]}...`)"
            )
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

# ── document list ───────────────────────────────────────────────────
st.divider()
st.subheader("Stored Documents")

if st.button("Refresh list"):
    st.rerun()

try:
    docs = list_rag_documents()
except Exception as exc:
    st.error(f"Could not load documents: {exc}")
    docs = []

if not docs:
    st.info("No documents in the vector store yet.")
else:
    df = pd.DataFrame(docs)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # delete action
    doc_ids = [d.get("document_id", d.get("id", "")) for d in docs]
    to_delete = st.selectbox("Select document to delete", doc_ids)
    if st.button("Delete selected", type="secondary"):
        with st.spinner("Deleting..."):
            try:
                delete_rag_document(to_delete)
                st.success(f"Deleted `{to_delete[:8]}...`")
                st.rerun()
            except Exception as exc:
                st.error(f"Delete failed: {exc}")

# ── semantic search preview ─────────────────────────────────────────
st.divider()
st.subheader("Semantic Search Preview")
st.markdown("Test what context the RAG engine would retrieve for a given query.")

query = st.text_input("Query", placeholder="e.g. customer satisfaction with delivery")
col_k, col_ns2 = st.columns(2)
with col_k:
    top_k = st.slider("Top K results", min_value=1, max_value=20, value=5)
with col_ns2:
    search_ns = st.text_input("Namespace", value="default", key="search_ns")

if st.button("Search", disabled=not query.strip()):
    with st.spinner("Retrieving..."):
        try:
            results = rag_retrieve(query, top_k=top_k, namespace=search_ns)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            results = []

    if not results:
        st.info("No matching chunks found.")
    else:
        for i, chunk in enumerate(results, 1):
            score = chunk.get("score", 0)
            source = chunk.get("source", "unknown")
            content = chunk.get("content", "")

            with st.container(border=True):
                st.markdown(f"**#{i}** — relevance **{score:.3f}** — source: `{source}`")
                st.text(content[:500])
