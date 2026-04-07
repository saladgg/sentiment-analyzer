"""
Run History page — browse past runs and compare two runs side-by-side.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from api_client import compare_runs, list_runs
from theme import SENTIMENT_COLORS

st.title("Run History")

# ── fetch runs ──────────────────────────────────────────────────────
try:
    runs = list_runs()
except Exception as exc:
    st.error(f"Could not load runs: {exc}")
    st.stop()

if not runs:
    st.info("No analysis runs yet. Head to **Analyze** to create one.")
    st.stop()

# ── runs table ──────────────────────────────────────────────────────
df = pd.DataFrame(runs)

# DuckDB returns created_at as string — parse it for display
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.sort_values("created_at", ascending=False)

display_cols = [c for c in ["run_id", "engine", "rag_enabled", "created_at"] if c in df.columns]
st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

# ── run detail expander ─────────────────────────────────────────────
st.subheader("Run Detail")

run_ids = df["run_id"].tolist()
selected_run_id = st.selectbox("Select a run", run_ids)

if selected_run_id:
    row = df[df["run_id"] == selected_run_id].iloc[0]
    summary = row.get("summary")

    if isinstance(summary, dict):
        dist = summary.get("distribution", {})
        total = summary.get("total_records", 0)
        dominant = summary.get("dominant_sentiment", "—")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total records", total)
        m2.metric("Dominant", dominant.capitalize())
        m3.metric("Engine", row.get("engine", "—"))

        if dist:
            labels = list(dist.keys())
            values = list(dist.values())
            colors = [SENTIMENT_COLORS.get(lbl, "#94a3b8") for lbl in labels]

            fig = go.Figure(
                go.Pie(
                    labels=[lbl.capitalize() for lbl in labels],
                    values=values,
                    hole=0.5,
                    marker={"colors": colors},
                    textinfo="label+percent",
                )
            )
            fig.update_layout(
                margin={"t": 20, "b": 0, "l": 0, "r": 0},
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.json(row.to_dict())

# ── run comparison ──────────────────────────────────────────────────
st.divider()
st.subheader("Compare Runs")
st.markdown(
    "Select a **base** run and a **RAG-enriched** run to see how RAG context affects sentiment."
)

col_base, col_rag = st.columns(2)
with col_base:
    base_id = st.selectbox("Base run", run_ids, key="cmp_base")
with col_rag:
    rag_id = st.selectbox("RAG run", run_ids, key="cmp_rag", index=min(1, len(run_ids) - 1))

if st.button("Compare", type="primary"):
    if base_id == rag_id:
        st.warning("Please select two different runs.")
    else:
        with st.spinner("Comparing..."):
            try:
                cmp = compare_runs(base_id, rag_id)
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")
                st.stop()

        comparison = cmp.get("comparison", {})
        st.success("Comparison complete.")

        if isinstance(comparison, dict):
            cols = st.columns(len(comparison))
            for col, (key, val) in zip(cols, comparison.items(), strict=False):
                col.metric(key.replace("_", " ").title(), f"{val}")
        else:
            st.json(cmp)
