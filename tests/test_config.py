"""Tests for settings parsing and startup validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentiment_analyzer.core.config import Settings


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


class TestSettings:
    def test_comma_separated_lists_from_env(self, monkeypatch):
        monkeypatch.setenv("ASA_API_KEYS", "one, two,,")
        monkeypatch.setenv("ASA_CORS_ORIGINS", "https://a.example,https://b.example")
        settings = _settings()
        assert [k.get_secret_value() for k in settings.api_keys] == ["one", "two"]
        assert settings.cors_origins == ["https://a.example", "https://b.example"]
        assert settings.auth_enabled

    def test_blank_model_revision_means_latest(self, monkeypatch):
        monkeypatch.setenv("ASA_HF_MODEL_REVISION", "")
        assert _settings().hf_model_revision is None

    def test_env_example_is_valid(self):
        """Every value in .env.example must parse."""
        settings = Settings(_env_file=".env.example")  # type: ignore[call-arg]
        assert settings.environment == "dev"
        assert settings.api_keys == []

    def test_secrets_are_not_in_repr(self, monkeypatch):
        monkeypatch.setenv("ASA_LLM_API_KEY", "sk-very-secret")
        assert "sk-very-secret" not in repr(_settings())

    def test_runs_db_path_and_deprecated_alias(self, monkeypatch):
        monkeypatch.delenv("ASA_RUNS_DB_PATH", raising=False)
        monkeypatch.setenv("ASA_RAG_DB_PATH", "legacy.duckdb")
        with pytest.warns(DeprecationWarning, match="ASA_RUNS_DB_PATH"):
            settings = _settings()
        assert settings.runs_db_path == "legacy.duckdb"

        monkeypatch.setenv("ASA_RUNS_DB_PATH", "new.duckdb")
        assert _settings().runs_db_path == "new.duckdb"

    def test_unknown_keys_are_ignored(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("ASA_ENABLE_EVALUATION=true\nASA_RAG_TOP_K=3\n")
        settings = Settings(_env_file=str(env))  # type: ignore[call-arg]
        assert settings.rag_top_k == 3

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"environment": "production"},
            {"default_sentiment_engine": "magic"},
            {"rag_chunk_size": 100, "rag_chunk_overlap": 100},
            {"max_records": 0},
            {"hf_neutral_band": 0.5},
        ],
    )
    def test_invalid_values(self, kwargs):
        with pytest.raises(ValidationError):
            _settings(**kwargs)

    def test_startup_validation(self):
        _settings(environment="dev").validate_for_startup()
        _settings(environment="prod", api_keys=["k"]).validate_for_startup()
        with pytest.raises(RuntimeError):
            _settings(environment="staging").validate_for_startup()
