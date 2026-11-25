"""
Engine registry with lazy, cached construction.

Engines are expensive (the HF engine loads model weights), so each is
built at most once per process, on first use. A ``rule_based`` request
therefore never touches torch.
"""

import threading
from collections.abc import Callable
from typing import Any

from sentiment_analyzer.core.config import Settings
from sentiment_analyzer.core.errors import AppError, EngineUnavailableError
from sentiment_analyzer.core.logging import logger
from sentiment_analyzer.sentiment.base import SentimentEngine
from sentiment_analyzer.sentiment.hf_transformer import HuggingFaceSentimentEngine
from sentiment_analyzer.sentiment.llm import LLMSentimentEngine
from sentiment_analyzer.sentiment.rule_based import RuleBasedSentimentEngine

ENGINE_CLASSES: dict[str, type[SentimentEngine]] = {
    cls.name: cls
    for cls in (RuleBasedSentimentEngine, HuggingFaceSentimentEngine, LLMSentimentEngine)
}


class EngineRegistry:
    """Builds engines on demand and caches them for the process lifetime."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._factories: dict[str, Callable[[], SentimentEngine]] = {
            RuleBasedSentimentEngine.name: RuleBasedSentimentEngine,
            HuggingFaceSentimentEngine.name: lambda: HuggingFaceSentimentEngine(
                settings.hf_sentiment_model,
                revision=settings.hf_model_revision,
                batch_size=settings.hf_batch_size,
                neutral_band=settings.hf_neutral_band,
            ),
            LLMSentimentEngine.name: lambda: LLMSentimentEngine(
                settings.llm_model,
                api_key=settings.llm_api_key.get_secret_value(),
                timeout=settings.llm_timeout,
                num_retries=settings.llm_num_retries,
                max_tokens=settings.llm_max_tokens,
                concurrency=settings.llm_concurrency,
            ),
        }
        self._engines: dict[str, SentimentEngine] = {}
        self._lock = threading.Lock()

    @property
    def names(self) -> list[str]:
        return list(self._factories)

    def engine_class(self, name: str) -> type[SentimentEngine]:
        if name not in ENGINE_CLASSES:
            raise EngineUnavailableError(f"Unknown sentiment engine: {name!r}")
        return ENGINE_CLASSES[name]

    def get(self, name: str) -> SentimentEngine:
        """Return the engine, building it on first use."""
        if name not in self._factories:
            raise EngineUnavailableError(f"Unknown sentiment engine: {name!r}")
        if name in self._engines:
            return self._engines[name]
        with self._lock:
            if name not in self._engines:
                logger.info("engine_loading", engine=name)
                try:
                    self._engines[name] = self._factories[name]()
                except AppError:
                    raise
                except Exception as exc:
                    logger.exception("engine_load_failed", engine=name)
                    raise EngineUnavailableError(
                        f"Engine {name!r} could not be loaded: {exc}"
                    ) from exc
                logger.info("engine_loaded", engine=name)
        return self._engines[name]

    def is_loaded(self, name: str) -> bool:
        return name in self._engines

    def is_configured(self, name: str) -> bool:
        if name == LLMSentimentEngine.name:
            return bool(self._settings.llm_model)
        return name in self._factories

    def describe(self) -> list[dict[str, Any]]:
        """Static metadata for every engine, without loading any."""
        return [
            {
                "name": name,
                "display_name": ENGINE_CLASSES[name].display_name,
                "description": ENGINE_CLASSES[name].description,
                "uses_context": ENGINE_CLASSES[name].uses_context,
                "is_paid": ENGINE_CLASSES[name].is_paid,
                "configured": self.is_configured(name),
                "loaded": self.is_loaded(name),
                "default": name == self._settings.default_sentiment_engine,
            }
            for name in self._factories
        ]
