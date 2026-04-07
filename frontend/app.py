"""
SentimentAnalyzer — Streamlit frontend entry-point.

Launch:
    streamlit run frontend/app.py
"""

import streamlit as st

# --------------- page config (must be first st call) ---------------
st.set_page_config(
    page_title="SentimentAnalyzer",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------- navigation ----------------------------------------
analyze_page = st.Page("pages/analyze.py", title="Analyze", icon=":material/search:", default=True)
history_page = st.Page("pages/history.py", title="Run History", icon=":material/history:")
rag_page = st.Page("pages/rag.py", title="RAG Management", icon=":material/library_books:")

pg = st.navigation(
    {
        "Analysis": [analyze_page],
        "Platform": [history_page, rag_page],
    }
)

# --------------- sidebar branding ----------------------------------
with st.sidebar:
    st.markdown("### SentimentAnalyzer")
    st.caption("Input-agnostic, explainable sentiment analysis with optional RAG enrichment.")
    st.divider()

pg.run()
