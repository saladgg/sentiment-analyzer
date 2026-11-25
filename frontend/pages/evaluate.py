"""
Evaluate page — run an engine over labeled texts and see how it scores.
"""

from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import APIError, evaluate_engine, list_engines

LABELS = {"positive", "neutral", "negative"}

st.title("Evaluate an Engine")
st.markdown(
    "Upload labeled examples (a CSV with `text` and `label` columns, where label is "
    "`positive`, `neutral` or `negative`). The engine predicts each text and the page "
    "reports accuracy, per-class precision/recall/F1 and the confusion matrix. "
    "Evaluations are not saved as runs."
)

try:
    engines = list_engines()["engines"]
except APIError as exc:
    st.error(f"Could not load engines: {exc}")
    st.stop()

by_name = {e["name"]: e for e in engines}
engine = st.selectbox(
    "Engine",
    list(by_name),
    format_func=lambda k: (
        by_name[k]["display_name"] + ("" if by_name[k]["configured"] else " (not configured)")
    ),
)
if by_name[engine]["is_paid"]:
    st.warning("This engine is billed per text.", icon=":material/payments:")

uploaded = st.file_uploader("Labeled CSV", type=["csv"])
sample = st.toggle("Use a small built-in sample instead")

items: list[dict[str, str]] = []
if sample:
    items = [
        {"text": "Absolutely love it, works perfectly", "label": "positive"},
        {"text": "Terrible support, never again", "label": "negative"},
        {"text": "It arrived on Tuesday", "label": "neutral"},
        {"text": "Not bad at all for the price", "label": "positive"},
        {"text": "The screen is not good", "label": "negative"},
        {"text": "Package contains two cables", "label": "neutral"},
    ]
elif uploaded is not None:
    try:
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    except Exception as exc:  # noqa: BLE001 - any parse failure is shown to the user
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
    if not {"text", "label"} <= set(df.columns):
        st.error("The CSV needs `text` and `label` columns.")
        st.stop()
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    unknown = sorted(set(df["label"]) - LABELS)
    if unknown:
        st.error(f"Unknown labels: {unknown}. Use positive, neutral or negative.")
        st.stop()
    items = [{"text": str(t), "label": lbl} for t, lbl in zip(df["text"], df["label"], strict=True)]

if items:
    st.caption(f"{len(items)} labeled texts ready.")

if st.button("Evaluate", type="primary", disabled=not items):
    with st.spinner("Evaluating..."):
        try:
            result = evaluate_engine(items, engine=engine)
        except APIError as exc:
            st.error(f"Evaluation failed: {exc}")
            st.stop()

    report = result["report"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{report['accuracy']:.1%}")
    m2.metric("Macro F1", f"{report['macro_f1']:.3f}")
    m3.metric("Evaluated", report["total_samples"])
    m4.metric("Failed", result["failed"])

    if report["per_class"]:
        st.subheader("Per class")
        st.dataframe(
            pd.DataFrame(report["per_class"]).T.reset_index(names="label"),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Confusion matrix")
        cm = report["confusion_matrix"]
        fig = px.imshow(
            cm["matrix"],
            x=cm["labels"],
            y=cm["labels"],
            text_auto=True,
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            color_continuous_scale="Blues",
        )
        fig.update_layout(margin={"t": 10, "b": 0, "l": 0, "r": 0}, height=360)
        st.plotly_chart(fig, width="stretch")
