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

import os
import warnings
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

EngineSetting = Literal["rule_based", "hf_transformer", "llm"]


def _split_csv(value: Any) -> Any:
    """Accept comma-separated strings for list settings (``a,b,c``)."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """
    Application-wide settings.

    Values can be overridden via environment variables prefixed ``ASA_``
    (AllSentimentAnalyzer) or a ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="ASA_",
        env_file=".env",
        env_file_encoding="utf-8",
        # Unknown ASA_* keys in .env (e.g. removed settings) must not
        # prevent the app from booting.
        extra="ignore",
        validate_by_name=True,
    )

    # -------------------------
    # Application
    # -------------------------
    app_name: str = "SentimentAnalyzer"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Load the default engine at startup so the first request is fast.
    warmup_default_engine: bool = True

    # -------------------------
    # HTTP / security
    # -------------------------
    # Comma-separated in the environment: ASA_API_KEYS=key-one,key-two
    # Empty means "no auth", which is only allowed when environment=dev.
    api_keys: Annotated[list[SecretStr], NoDecode] = Field(default_factory=list)
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:8501"]
    )

    # -------------------------
    # Input limits
    # -------------------------
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_records: int = Field(default=10_000, gt=0)
    max_text_chars: int = Field(default=5_000, gt=0)
    max_llm_records: int = Field(default=500, gt=0)

    # -------------------------
    # Engines
    # -------------------------
    default_sentiment_engine: EngineSetting = "rule_based"

    hf_sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    hf_model_revision: str | None = None
    hf_batch_size: int = Field(default=16, gt=0)
    # Binary models never predict "neutral". Predictions whose positive-class
    # probability lies within 0.5 +/- hf_neutral_band are reported as neutral.
    hf_neutral_band: float = Field(default=0.0, ge=0.0, lt=0.5)

    # LLM engine (any litellm-supported provider)
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_timeout: float = Field(default=60.0, gt=0)
    llm_num_retries: int = Field(default=2, ge=0)
    llm_max_tokens: int = Field(default=512, gt=0)
    llm_concurrency: int = Field(default=8, gt=0)

    # -------------------------
    # RAG configuration
    # -------------------------
    rag_top_k: int = Field(default=5, ge=1, le=50)
    rag_enabled: bool = True
    rag_namespace: str = "default"
    # Directory holding the persistent Chroma store.
    vector_store_path: str = "./vector_store"
    rag_chunk_size: int = Field(default=500, gt=0)
    rag_chunk_overlap: int = Field(default=50, ge=0)

    # -------------------------
    # Storage
    # -------------------------
    # DuckDB file holding run history. ASA_RAG_DB_PATH is accepted as a
    # deprecated alias.
    runs_db_path: str = Field(
        default="runs.duckdb",
        validation_alias=AliasChoices("ASA_RUNS_DB_PATH", "ASA_RAG_DB_PATH", "runs_db_path"),
    )

    @field_validator("api_keys", "cors_origins", mode="before")
    @classmethod
    def _split_lists(cls, value: Any) -> Any:
        return _split_csv(value)

    @field_validator("hf_model_revision", mode="before")
    @classmethod
    def _blank_is_none(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("rag_chunk_overlap")
    @classmethod
    def _overlap_smaller_than_chunk(cls, value: int, info: ValidationInfo) -> int:
        size = info.data.get("rag_chunk_size", 500)
        if value >= size:
            raise ValueError("rag_chunk_overlap must be smaller than rag_chunk_size")
        return value

    def model_post_init(self, __context: Any) -> None:
        if "ASA_RAG_DB_PATH" in os.environ and "ASA_RUNS_DB_PATH" not in os.environ:
            warnings.warn(
                "ASA_RAG_DB_PATH is deprecated; use ASA_RUNS_DB_PATH",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)

    def validate_for_startup(self) -> None:
        """Refuse insecure configurations outside local development."""
        if self.environment != "dev" and not self.api_keys:
            raise RuntimeError(
                f"ASA_API_KEYS must be set when ASA_ENVIRONMENT={self.environment!r}"
            )


settings = Settings()


def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return settings
