"""
Analyze page — upload data, pick an engine, visualize results.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api_client import (
    APIError,
    analyze_csv,
    analyze_excel,
    analyze_json,
    export_run,
    get_job,
    get_run,
    list_engines,
)
from theme import ENGINE_DESCRIPTIONS, ENGINE_LABELS, SENTIMENT_COLORS

RUN_NOW = "Run now"
BACKGROUND = "Background job"
AB_TEST = "RAG A/B test"
FILE_INPUT = "File upload (CSV / Excel)"
JSON_INPUT = "Paste JSON"
TEXT_INPUT = "Plain text (one per line)"


@st.cache_data(ttl=60, show_spinner=False)
def _engines() -> tuple[list[dict], dict]:
    """Engines from the API; falls back to the built-in list if it's unreachable."""
    try:
        payload = list_engines()
        return payload["engines"], payload["limits"]
    except APIError:
        engines = [
            {
                "name": k,
                "display_name": v,
                "description": ENGINE_DESCRIPTIONS[k],
                "configured": True,
            }
            for k, v in ENGINE_LABELS.items()
        ]
        return engines, {}


# ── header ──────────────────────────────────────────────────────────
st.title("Sentiment Analysis")
st.markdown("Upload your data, choose an engine, and get explainable sentiment scores.")

engines, limits = _engines()
by_name = {e["name"]: e for e in engines}
default_index = next((i for i, e in enumerate(engines) if e.get("default")), 0)

# ── engine selector ─────────────────────────────────────────────────
col_eng, col_desc = st.columns([1, 2])
with col_eng:
    engine = st.selectbox(
        "Sentiment engine",
        options=list(by_name),
        index=default_index,
        format_func=lambda k: (
            by_name[k]["display_name"]
            + ("" if by_name[k].get("configured", True) else " (not configured)")
        ),
    )
with col_desc:
    st.info(by_name[engine]["description"], icon=":material/info:")

if limits:
    st.caption(
        f"Limits: {limits['max_records']:,} records per run "
        f"({limits['max_llm_records']:,} for paid engines), "
        f"{limits['max_upload_bytes'] // (1024 * 1024)} MB per file, "
        f"{limits['max_text_chars']:,} characters per record."
    )

st.divider()

# ── data input ──────────────────────────────────────────────────────
input_method = st.radio("Input method", [FILE_INPUT, JSON_INPUT, TEXT_INPUT], horizontal=True)

uploaded_file = None
json_text = ""
plain_text = ""

if input_method == FILE_INPUT:
    uploaded_file = st.file_uploader(
        "Drop a CSV or Excel file",
        type=["csv", "xlsx"],
        help="Text columns are auto-detected unless you list them under Options.",
    )
elif input_method == JSON_INPUT:
    json_text = st.text_area(
        "Paste a JSON array of objects",
        height=200,
        placeholder='[{"comment": "I love this product"}, {"comment": "Terrible experience"}]',
    )
else:
    plain_text = st.text_area(
        "Paste text — each non-empty line is analyzed separately",
        height=200,
        placeholder="Great service!\nThe delivery was late.",
    )

with st.expander("Options"):
    fields_raw = st.text_input(
        "Text fields (comma-separated)",
        help="Columns to analyze. Leave empty to auto-detect. Ignored for plain text.",
        disabled=input_method == TEXT_INPUT,
    )
    col_ts, col_gran = st.columns(2)
    with col_ts:
        timestamp_field = st.text_input(
            "Timestamp field (optional)",
            help="Column with dates, to chart sentiment over time.",
            disabled=input_method == TEXT_INPUT,
        )
    with col_gran:
        granularity = st.selectbox("Trend granularity", ["day", "week", "month", "year"])
    rag_choice = st.radio(
        "RAG context",
        ["Server default", "On", "Off"],
        horizontal=True,
        help="Only engines that read context (llm) use RAG.",
        disabled=not by_name[engine].get("uses_context", False),
    )

mode = st.radio(
    "Mode",
    [RUN_NOW, BACKGROUND, AB_TEST],
    horizontal=True,
    help=(
        "Background jobs accept larger inputs and report progress. The RAG A/B test "
        "analyzes the input twice (without and with RAG) and compares the runs."
    ),
)
if mode == AB_TEST and (input_method == FILE_INPUT or not by_name[engine].get("uses_context")):
    st.info(
        "The RAG A/B test needs pasted JSON or text and an engine that uses context (llm).",
        icon=":material/info:",
    )

text_fields = [f.strip() for f in fields_raw.split(",") if f.strip()] or None
use_rag = {"Server default": None, "On": True, "Off": False}[rag_choice]
options = {
    "text_fields": text_fields,
    "timestamp_field": timestamp_field.strip() or None,
    "granularity": granularity,
    "use_rag": use_rag if mode != AB_TEST else None,
}


def _result_from_run(detail: dict) -> dict:
    """Shape a stored run like an analyze response (no anomalies/explanations)."""
    return {
        "run_id": detail["run_id"],
        "engine": detail["engine"],
        "text_fields": (detail.get("config") or {}).get("text_fields", []),
        "summary": detail["summary"],
        "records": detail["records"],
        "rag": None,
        "from_job": detail["records_total"],
    }


def _wait_for_job(job: dict) -> dict:
    """Poll a background job, showing progress, until it finishes."""
    bar = st.progress(0.0, text="Queued...")
    cancel_hint = st.caption(f"Job `{job['job_id']}` — also listed under Run History.")
    while job["status"] in ("queued", "running"):
        label = (
            "Queued..."
            if job["status"] == "queued"
            else (f"Analyzing {job['records_done'] or 0:,} / {job['records_total'] or 0:,} records")
        )
        bar.progress(job["progress"], text=label)
        time.sleep(1)
        job = get_job(job["job_id"])
    bar.progress(1.0 if job["status"] == "completed" else job["progress"], text=job["status"])
    cancel_hint.empty()
    return job


# ── run analysis ────────────────────────────────────────────────────
run_btn = st.button("Run Analysis", type="primary", width="stretch")

if run_btn:
    if input_method == FILE_INPUT and uploaded_file is None:
        st.warning("Please upload a file first.")
        st.stop()
    if input_method == JSON_INPUT and not json_text.strip():
        st.warning("Please paste some JSON data first.")
        st.stop()
    if input_method == TEXT_INPUT and not plain_text.strip():
        st.warning("Please paste some text first.")
        st.stop()

    if mode == AB_TEST and (input_method == FILE_INPUT or not by_name[engine].get("uses_context")):
        st.warning("Pick pasted JSON or text and the llm engine for a RAG A/B test.")
        st.stop()

    background = mode == BACKGROUND
    path = {RUN_NOW: "/analyze", BACKGROUND: "/jobs/analyze", AB_TEST: "/runs/ab"}[mode]
    st.session_state.pop("ab_comparison", None)
    try:
        with st.spinner("Submitting..." if background else "Running sentiment analysis..."):
            if input_method == JSON_INPUT:
                response = analyze_json(json.loads(json_text), engine=engine, path=path, **options)
            elif input_method == TEXT_INPUT:
                response = analyze_json(
                    plain_text,
                    engine=engine,
                    source_type="text",
                    granularity=granularity,
                    use_rag=options["use_rag"],
                    path=path,
                )
            elif uploaded_file is not None and uploaded_file.name.lower().endswith(".csv"):
                response = analyze_csv(
                    uploaded_file, engine=engine, background=background, **options
                )
            else:
                response = analyze_excel(
                    uploaded_file, engine=engine, background=background, **options
                )

        if background:
            job = _wait_for_job(response)
            if job["status"] != "completed":
                st.error(f"Job {job['status']}: {job.get('error') or 'no result'}")
                st.stop()
            result = _result_from_run(get_run(job["run_id"], records_limit=1000))
        elif mode == AB_TEST:
            result = response["rag"]
            st.session_state["ab_comparison"] = response["comparison"]
        else:
            result = response
    except json.JSONDecodeError:
        st.error("Invalid JSON. Please check your input.")
        st.stop()
    except APIError as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()

    st.session_state["last_result"] = result

# ── results dashboard ───────────────────────────────────────────────
if "last_result" not in st.session_state:
    st.stop()

result = st.session_state["last_result"]
summary = result["summary"]
records = result["records"]
ok_records = [r for r in records if not r.get("error")]
failed_records = [r for r in records if r.get("error")]
anomalies = result.get("anomalies", [])
explanations = result.get("explanations", [])
rag_meta = result.get("rag")

st.divider()
st.subheader("Results")

distribution = summary.get("distribution", {})
dominant = summary.get("dominant_sentiment") or "—"

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Analyzed", summary.get("total_records", len(ok_records)))
k2.metric("Dominant", dominant.capitalize())
k3.metric("Positive", distribution.get("positive", 0))
k4.metric("Neutral", distribution.get("neutral", 0))
k5.metric("Negative", distribution.get("negative", 0))
k6.metric("Failed", summary.get("failed_records", 0))

ab = st.session_state.get("ab_comparison")
if ab:
    stats = ab["comparison"]
    st.markdown("**RAG A/B test** — the results below are the run *with* RAG.")
    a1, a2, a3 = st.columns(3)
    a1.metric("Label changes", stats["label_changes"], f"{stats['label_change_rate']:.0%}")
    a2.metric(
        "Mean score Δ",
        "—" if stats["mean_score_delta"] is None else f"{stats['mean_score_delta']:+.3f}",
    )
    a3.metric(
        "Mean confidence Δ",
        "—" if stats["mean_confidence_delta"] is None else f"{stats['mean_confidence_delta']:+.3f}",
    )
    if ab["changed_records"]:
        st.dataframe(pd.DataFrame(ab["changed_records"]), width="stretch", hide_index=True)

if result.get("from_job"):
    st.caption(
        f"Background job result: showing up to 1,000 of {result['from_job']:,} records. "
        "Download the export for all of them."
    )

usage = summary.get("llm_usage")
if usage:
    cost = f"${usage['cost_usd']:.4f}" + ("" if usage["cost_complete"] else " (partial)")
    st.caption(
        f"LLM usage: {usage['requests']} requests ({usage['cached']} cached), "
        f"{usage['prompt_tokens']:,} + {usage['completion_tokens']:,} tokens, est. {cost}"
    )

if summary.get("avg_confidence") is not None:
    st.caption(
        f"Average score {summary['avg_score']:+.3f} · "
        f"average confidence {summary['avg_confidence']:.2f}"
    )

if rag_meta:
    rag_cols = st.columns(3)
    rag_cols[0].metric("RAG active", "Yes" if rag_meta["enabled"] else "No")
    rag_cols[1].metric("Context found", "Yes" if rag_meta["used"] else "No")
    rag_cols[2].metric("RAG sources", rag_meta.get("source_count", 0))

st.caption(
    f"Run ID: `{result['run_id']}` · engine `{result['engine']}` · "
    f"text fields: {', '.join(result.get('text_fields', [])) or '—'}"
)

col_csv, col_json, _ = st.columns([1, 1, 4])
for col, fmt in ((col_csv, "csv"), (col_json, "json")):
    with col:
        try:
            st.download_button(
                f"Download {fmt.upper()}",
                data=export_run(result["run_id"], fmt),
                file_name=f"run-{result['run_id']}.{fmt}",
                mime="text/csv" if fmt == "csv" else "application/json",
            )
        except APIError as exc:
            st.caption(f"Export unavailable: {exc}")

if failed_records:
    st.warning(f"{len(failed_records)} record(s) could not be analyzed.")
    with st.expander("Failed records"):
        st.dataframe(
            pd.DataFrame(failed_records)[["record_id", "error"]],
            width="stretch",
            hide_index=True,
        )

# ── charts ──────────────────────────────────────────────────────────
chart_left, chart_right = st.columns(2)

with chart_left:
    labels = list(distribution.keys())
    fig_donut = go.Figure(
        go.Pie(
            labels=[lbl.capitalize() for lbl in labels],
            values=list(distribution.values()),
            hole=0.5,
            marker={"colors": [SENTIMENT_COLORS.get(lbl, "#94a3b8") for lbl in labels]},
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
    st.plotly_chart(fig_donut, width="stretch")

df_records = pd.DataFrame(ok_records)
with chart_right:
    if not df_records.empty:
        fig_hist = px.histogram(
            df_records,
            x="score",
            color="sentiment",
            nbins=20,
            color_discrete_map=dict(SENTIMENT_COLORS),
            title="Score Distribution",
        )
        fig_hist.update_layout(margin={"t": 40, "b": 0, "l": 0, "r": 0}, height=350, bargap=0.05)
        st.plotly_chart(fig_hist, width="stretch")

# ── trend ───────────────────────────────────────────────────────────
trend = result.get("trend")
if trend and trend["buckets"]:
    st.subheader("Sentiment Over Time")
    df_trend = pd.DataFrame(trend["buckets"])
    fig_trend = px.line(df_trend, x="bucket", y="avg_score", markers=True, hover_data=["count"])
    fig_trend.update_layout(margin={"t": 10, "b": 0, "l": 0, "r": 0}, height=300)
    fig_trend.update_yaxes(range=[-1, 1], title="Average score")
    st.plotly_chart(fig_trend, width="stretch")
    if trend.get("unparseable"):
        st.caption(f"{trend['unparseable']} record(s) had no parseable timestamp.")

# ── per-record table ────────────────────────────────────────────────
st.subheader("Per-Record Results")

if not df_records.empty:

    def _color_sentiment(val: str) -> str:
        return f"color: {SENTIMENT_COLORS.get(val, '#94a3b8')}; font-weight: 600"

    columns = ["record_id", "sentiment", "score", "confidence", "truncated"]
    styled = (
        df_records[[c for c in columns if c in df_records.columns]]
        .style.map(_color_sentiment, subset=["sentiment"])
        .format({"score": "{:+.3f}", "confidence": "{:.2f}"})
    )
    st.dataframe(styled, width="stretch", hide_index=True)

# ── contributing fields detail ──────────────────────────────────────
if len(result.get("text_fields", [])) > 1:
    with st.expander("Per-field sentiment"):
        rows = [
            {"record_id": r["record_id"], "field": f, "score": s}
            for r in ok_records
            for f, s in r.get("contributing_fields", {}).items()
        ]
        if rows:
            fig_fields = px.bar(
                pd.DataFrame(rows),
                x="record_id",
                y="score",
                color="field",
                barmode="group",
                height=320,
            )
            fig_fields.update_yaxes(range=[-1, 1])
            st.plotly_chart(fig_fields, width="stretch")
        else:
            st.caption("Per-field scores are not computed for paid engines.")

# ── anomalies ───────────────────────────────────────────────────────
if anomalies:
    st.subheader("Anomalies Detected")
    st.warning(f"{len(anomalies)} anomalous record(s) found (z-score outliers).")
    st.dataframe(
        pd.DataFrame(anomalies)[["record_id", "sentiment", "score", "z_score"]],
        width="stretch",
        hide_index=True,
    )

# ── explanations ────────────────────────────────────────────────────
if explanations:
    with st.expander("Explanations"):
        for exp in explanations:
            st.markdown(f"- {exp}")
