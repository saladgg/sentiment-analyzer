"""
Centralized configuration for SentimentAnalyzer.

This module defines all environment-driven configuration
to ensure:
- Deterministic behavior
- Clear separation between code and environment
- Easy deployment across dev / staging / prod

All configuration values should be read from here,
never hard-coded elsewhere.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings.

    Values can be overridden via environment variables.
    """

    # AllSentimentAnalyzer(ASA)
    model_config = SettingsConfigDict(
        env_prefix="ASA_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Application
    app_name: str = "SentimentAnalyzer"
    environment: str = Field(default="dev", description="dev | staging | prod")

    # -------------------------
    # Models
    # -------------------------
    default_sentiment_engine: str = "rule_based"
    hf_sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"

    # -------------------------
    # LLM Engine (any litellm-supported provider)
    # -------------------------
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"

    # -------------------------
    # RAG configuration
    # -------------------------
    # Visit scripts/bootstrap_rag_store.py for better context.
    rag_top_k: int = 5
    rag_enabled: bool = True
    rag_namespace: str = "default"
    vector_store_path: str = "./vector_store"

    # -------------------------
    # Evaluation
    # -------------------------
    enable_evaluation: bool = True
    evaluation_sample_rate: float = 1.0

    # -------------------------
    # Storage
    # -------------------------
    rag_db_path: str = "runs.duckdb"


settings = Settings()
