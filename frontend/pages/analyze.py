"""
Analyze page — upload data, pick an engine, visualize results.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from api_client import analyze_csv, analyze_excel, analyze_json
from theme import ENGINE_DESCRIPTIONS, ENGINE_LABELS, SENTIMENT_COLORS

# ── header ──────────────────────────────────────────────────────────
st.title("Sentiment Analysis")
st.markdown("Upload your data, choose an engine, and get explainable sentiment scores.")

# ── engine selector ─────────────────────────────────────────────────
col_eng, col_desc = st.columns([1, 2])
with col_eng:
    engine = st.selectbox(
        "Sentiment engine",
        options=list(ENGINE_LABELS.keys()),
        format_func=lambda k: ENGINE_LABELS[k],
    )
with col_desc:
    st.info(ENGINE_DESCRIPTIONS[engine], icon=":material/info:")

st.divider()

# ── data input ──────────────────────────────────────────────────────
input_method = st.radio(
    "Input method",
    ["File upload (CSV / Excel)", "Paste JSON"],
    horizontal=True,
)

uploaded_file = None
json_text = ""

if input_method == "File upload (CSV / Excel)":
    uploaded_file = st.file_uploader(
        "Drop a CSV or Excel file",
        type=["csv", "xlsx"],
        help="The platform auto-detects text fields and runs sentiment on each record.",
    )
else:
    json_text = st.text_area(
        "Paste a JSON array of objects",
        height=200,
        placeholder='[{"comment": "I love this product"}, {"comment": "Terrible experience"}]',
    )

# ── run analysis ────────────────────────────────────────────────────
run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

if run_btn:
    # Validate input
    if input_method == "File upload (CSV / Excel)" and uploaded_file is None:
        st.warning("Please upload a file first.")
        st.stop()
    if input_method == "Paste JSON" and not json_text.strip():
        st.warning("Please paste some JSON data first.")
        st.stop()

    with st.spinner("Running sentiment analysis..."):
        try:
            if input_method == "Paste JSON":
                data = json.loads(json_text)
                result = analyze_json(data, engine=engine)
            elif uploaded_file.name.endswith(".csv"):
                result = analyze_csv(uploaded_file, engine=engine)
            else:
                result = analyze_excel(uploaded_file, engine=engine)
        except json.JSONDecodeError:
            st.error("Invalid JSON. Please check your input.")
            st.stop()
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    # store in session so the dashboard survives widget interactions
    st.session_state["last_result"] = result

# ── results dashboard ───────────────────────────────────────────────
if "last_result" not in st.session_state:
    st.stop()

result = st.session_state["last_result"]
summary = result["summary"]
records = result["records"]
anomalies = result.get("anomalies", [])
explanations = result.get("explanations", [])
rag_meta = result.get("rag")

# ── KPI row ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Results")

distribution = summary.get("distribution", {})
total = summary.get("total_records", len(records))
dominant = summary.get("dominant_sentiment", "—")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total records", total)
k2.metric("Dominant sentiment", dominant.capitalize())
k3.metric("Positive", distribution.get("positive", 0))
k4.metric("Neutral", distribution.get("neutral", 0))
k5.metric("Negative", distribution.get("negative", 0))

if rag_meta:
    rag_cols = st.columns(3)
    rag_cols[0].metric("RAG enabled", "Yes" if rag_meta["enabled"] else "No")
    rag_cols[1].metric("RAG used", "Yes" if rag_meta["used"] else "No")
    rag_cols[2].metric("RAG sources", rag_meta.get("source_count", 0))

st.caption(f"Run ID: `{result['run_id']}`")

# ── charts ──────────────────────────────────────────────────────────
chart_left, chart_right = st.columns(2)

with chart_left:
    # donut chart — sentiment distribution
    labels = list(distribution.keys())
    values = list(distribution.values())
    colors = [SENTIMENT_COLORS.get(lbl, "#94a3b8") for lbl in labels]

    fig_donut = go.Figure(
        go.Pie(
            labels=[lbl.capitalize() for lbl in labels],
            values=values,
            hole=0.5,
            marker={"colors": colors},
            textinfo="label+percent",
            hoverinfo="label+value",
        )
    )
    fig_donut.update_layout(
        title="Sentiment Distribution",
        margin={"t": 40, "b": 0, "l": 0, "r": 0},
        height=350,
        showlegend=False,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with chart_right:
    # score histogram
    df_records = pd.DataFrame(records)
    if not df_records.empty and "score" in df_records.columns:
        fig_hist = px.histogram(
            df_records,
            x="score",
            color="sentiment",
            nbins=20,
            color_discrete_map=dict(SENTIMENT_COLORS),
            title="Score Distribution",
        )
        fig_hist.update_layout(
            margin={"t": 40, "b": 0, "l": 0, "r": 0},
            height=350,
            bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ── per-record table ────────────────────────────────────────────────
st.subheader("Per-Record Results")

if not df_records.empty:
    # colour-code sentiment column
    def _color_sentiment(val: str) -> str:
        color = SENTIMENT_COLORS.get(val, "#94a3b8")
        return f"color: {color}; font-weight: 600"

    styled = (
        df_records[["record_id", "sentiment", "score"]]
        .style.map(_color_sentiment, subset=["sentiment"])
        .format({"score": "{:.3f}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ── contributing fields detail ──────────────────────────────────────
with st.expander("Contributing field scores per record"):
    for rec in records:
        fields = rec.get("contributing_fields", {})
        if fields:
            st.markdown(f"**Record {rec['record_id']}** — {rec['sentiment']} ({rec['score']:.3f})")
            field_df = pd.DataFrame([{"field": k, "score": v} for k, v in fields.items()])
            fig_bar = px.bar(
                field_df,
                x="field",
                y="score",
                color_discrete_sequence=[SENTIMENT_COLORS.get(rec["sentiment"], "#6366f1")],
                height=220,
            )
            fig_bar.update_layout(
                margin={"t": 10, "b": 0, "l": 0, "r": 0},
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ── anomalies ───────────────────────────────────────────────────────
if anomalies:
    st.subheader("Anomalies Detected")
    st.warning(f"{len(anomalies)} anomalous record(s) found (z-score outliers).")
    st.dataframe(pd.DataFrame(anomalies), use_container_width=True, hide_index=True)

# ── explanations ────────────────────────────────────────────────────
if explanations:
    with st.expander("Explanations"):
        for exp in explanations:
            st.markdown(f"- {exp}")
