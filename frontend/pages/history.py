"""
Run History page — browse past runs, inspect one, export it, and compare two.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import APIError, compare_runs, export_run, get_run, list_runs
from theme import SENTIMENT_COLORS

PAGE_SIZE = 25

st.title("Run History")

# ── fetch one page of runs ──────────────────────────────────────────
page_number = st.session_state.get("history_page", 0)
try:
    page = list_runs(limit=PAGE_SIZE, offset=page_number * PAGE_SIZE)
except APIError as exc:
    st.error(f"Could not load runs: {exc}")
    st.stop()

runs = page["items"]
total = page["total"]

if not total:
    st.info("No analysis runs yet. Head to **Analyze** to create one.")
    st.stop()

df = pd.DataFrame(runs)
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
display_cols = [
    c
    for c in ["run_id", "engine", "status", "records_in", "rag_enabled", "created_at"]
    if c in df.columns
]
st.dataframe(df[display_cols], width="stretch", hide_index=True)

last_page = max(0, (total - 1) // PAGE_SIZE)
nav_prev, nav_info, nav_next = st.columns([1, 2, 1])
if nav_prev.button("← Newer", disabled=page_number == 0):
    st.session_state["history_page"] = page_number - 1
    st.rerun()
nav_info.caption(f"Page {page_number + 1} of {last_page + 1} · {total} runs")
if nav_next.button("Older →", disabled=page_number >= last_page):
    st.session_state["history_page"] = page_number + 1
    st.rerun()

# ── run detail ──────────────────────────────────────────────────────
st.subheader("Run Detail")
run_ids = df["run_id"].tolist()
selected_run_id = st.selectbox("Select a run", run_ids)

if selected_run_id:
    try:
        detail = get_run(selected_run_id, records_limit=200)
    except APIError as exc:
        st.error(f"Could not load run: {exc}")
        st.stop()

    summary = detail.get("summary") or {}
    dist = summary.get("distribution", {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Analyzed", summary.get("total_records", 0))
    m2.metric("Dominant", (summary.get("dominant_sentiment") or "—").capitalize())
    m3.metric("Engine", detail.get("engine", "—"))
    m4.metric("Failed", detail.get("records_failed") or 0)

    if dist:
        labels = list(dist.keys())
        fig = go.Figure(
            go.Pie(
                labels=[lbl.capitalize() for lbl in labels],
                values=list(dist.values()),
                hole=0.5,
                marker={"colors": [SENTIMENT_COLORS.get(lbl, "#94a3b8") for lbl in labels]},
                textinfo="label+percent",
            )
        )
        fig.update_layout(margin={"t": 20, "b": 0, "l": 0, "r": 0}, height=300, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    if detail["records"]:
        st.caption(f"Showing {len(detail['records'])} of {detail['records_total']} records")
        st.dataframe(
            pd.DataFrame(detail["records"])[
                ["record_id", "sentiment", "score", "confidence", "truncated", "error"]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("This run predates per-record storage; only its summary is available.")

    with st.expander("Configuration snapshot"):
        st.json(
            {
                "config": detail.get("config"),
                "source_type": detail.get("source_type"),
                "input_sha256": detail.get("input_sha256"),
                "duration_ms": detail.get("duration_ms"),
            }
        )

    col_csv, col_json, _ = st.columns([1, 1, 4])
    for col, fmt in ((col_csv, "csv"), (col_json, "json")):
        with col:
            try:
                st.download_button(
                    f"Download {fmt.upper()}",
                    data=export_run(selected_run_id, fmt),
                    file_name=f"run-{selected_run_id}.{fmt}",
                    key=f"dl-{fmt}",
                )
            except APIError as exc:
                st.caption(f"Export unavailable: {exc}")

# ── run comparison ──────────────────────────────────────────────────
st.divider()
st.subheader("Compare Runs")
st.markdown(
    "Compare two runs **of the same input** — e.g. without and with RAG, or two engines — "
    "to see which records changed label."
)

col_base, col_other = st.columns(2)
with col_base:
    base_id = st.selectbox("Base run", run_ids, key="cmp_base")
with col_other:
    other_id = st.selectbox(
        "Comparison run", run_ids, key="cmp_other", index=min(1, len(run_ids) - 1)
    )

if st.button("Compare", type="primary"):
    if base_id == other_id:
        st.warning("Please select two different runs.")
    else:
        with st.spinner("Comparing..."):
            try:
                cmp = compare_runs(base_id, other_id)
            except APIError as exc:
                st.error(f"Comparison failed: {exc}")
                st.stop()

        stats = cmp["comparison"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records compared", stats["records_compared"])
        c2.metric("Label changes", stats["label_changes"], f"{stats['label_change_rate']:.0%}")
        c3.metric(
            "Mean score Δ",
            "—" if stats["mean_score_delta"] is None else f"{stats['mean_score_delta']:+.3f}",
        )
        c4.metric(
            "Mean confidence Δ",
            "—"
            if stats["mean_confidence_delta"] is None
            else f"{stats['mean_confidence_delta']:+.3f}",
        )
        if cmp["changed_records"]:
            st.dataframe(pd.DataFrame(cmp["changed_records"]), width="stretch", hide_index=True)
