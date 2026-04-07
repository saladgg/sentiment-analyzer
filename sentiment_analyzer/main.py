"""
FastAPI entrypoint for SentimentAnalyzer.

This module initializes the API application, mounts routes,
and exposes the HTTP interface consumed by the React frontend.
"""

from importlib.metadata import version

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sentiment_analyzer.api.evaluation_routes import router as evaluation_router
from sentiment_analyzer.api.rag_routes import router as rag_router
from sentiment_analyzer.api.routes import router as analysis_router
from sentiment_analyzer.api.run_routes import router as run_router
from sentiment_analyzer.core.logging import configure_logging

# init logging
configure_logging()

app = FastAPI(
    title="SentimentAnalyzer",
    description="Input-agnostic, explainable sentiment analysis platform",
    version=version("sentiment-analyzer"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8501"],  # React & Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")
app.include_router(rag_router, prefix="/api")
app.include_router(run_router, prefix="/api")
