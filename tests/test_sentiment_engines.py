"""
Unit tests for the sentiment engine implementations.

The HuggingFace and LLM engines avoid network/model loads by using the
``stub_hf_transformer`` autouse fixture and a monkeypatched
``litellm.completion`` respectively.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from sentiment_analyzer.sentiment.base import SentimentEngine
from sentiment_analyzer.sentiment.hf_transformer import HuggingFaceSentimentEngine
from sentiment_analyzer.sentiment.llm_stub import LLMSentimentEngine
from sentiment_analyzer.sentiment.rule_based import RuleBasedSentimentEngine


class TestRuleBasedEngine:
    @pytest.fixture
    def engine(self) -> RuleBasedSentimentEngine:
        return RuleBasedSentimentEngine()

    def test_positive_text(self, engine):
        result = engine.analyze("I love this great product")
        assert result["label"] == "positive"
        assert result["score"] > 0
        assert result["drivers"]["positive_hits"] >= 2
        assert result["drivers"]["negative_hits"] == 0

    def test_negative_text(self, engine):
        result = engine.analyze("a terrible and awful experience")
        assert result["label"] == "negative"
        assert result["score"] < 0

    def test_neutral_when_no_lexical_hits(self, engine):
        result = engine.analyze("the package arrived on tuesday")
        assert result["label"] == "neutral"
        assert result["score"] == 0.0
        assert result["drivers"] == {"lexical_rules": 0.0}

    def test_balanced_hits_yield_neutral(self, engine):
        result = engine.analyze("good but bad")
        assert result["label"] == "neutral"
        assert result["score"] == 0.0

    def test_context_argument_is_ignored(self, engine):
        without_ctx = engine.analyze("good product")
        with_ctx = engine.analyze("good product", context="extra context")
        assert without_ctx == with_ctx


class TestHuggingFaceEngine:
    """The pipeline is replaced by ``stub_hf_transformer`` (autouse)."""

    def test_positive_label_keeps_score_positive(self):
        engine = HuggingFaceSentimentEngine("dummy-model")
        result = engine.analyze("good thing")
        assert result["label"] == "positive"
        assert result["score"] > 0
        assert "model_confidence" in result["drivers"]

    def test_negative_label_flips_score_sign(self):
        engine = HuggingFaceSentimentEngine("dummy-model")
        result = engine.analyze("nope")
        assert result["label"] == "negative"
        assert result["score"] < 0


class TestLLMEngine:
    def test_parses_structured_json_response(self, monkeypatch):
        from sentiment_analyzer.sentiment import llm_stub

        captured: dict = {}

        def _fake_completion(*, model, max_tokens, messages):
            captured["model"] = model
            captured["messages"] = messages
            payload = json.dumps({"label": "positive", "score": 0.75, "drivers": {"tone": 0.9}})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
            )

        monkeypatch.setattr(llm_stub.litellm, "completion", _fake_completion)

        result = LLMSentimentEngine().analyze("amazing service", context="VIP customer")

        assert result == {
            "label": "positive",
            "score": 0.75,
            "drivers": {"tone": 0.9},
        }
        # Context must be appended to the user prompt when provided.
        assert "CONTEXT:" in captured["messages"][1]["content"]
        assert "VIP customer" in captured["messages"][1]["content"]

    def test_omits_context_block_when_no_context(self, monkeypatch):
        from sentiment_analyzer.sentiment import llm_stub

        captured: dict = {}

        def _fake_completion(*, model, max_tokens, messages):
            captured["messages"] = messages
            payload = json.dumps({"label": "neutral", "score": 0.0, "drivers": {}})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
            )

        monkeypatch.setattr(llm_stub.litellm, "completion", _fake_completion)

        LLMSentimentEngine().analyze("just text")

        assert "CONTEXT:" not in captured["messages"][1]["content"]


class TestBaseEngineContract:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            SentimentEngine()  # type: ignore[abstract]
