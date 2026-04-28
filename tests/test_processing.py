"""
Tests for the input-processing utilities.

Covers field detection heuristics, record normalization, and validation.
"""

from __future__ import annotations

from sentiment_analyzer.processing.field_detector import detect_text_fields
from sentiment_analyzer.processing.normalizer import normalize_record
from sentiment_analyzer.processing.validator import validate_record


class TestDetectTextFields:
    def test_returns_empty_when_no_records(self):
        assert detect_text_fields([]) == []

    def test_finds_long_string_fields(self):
        records = [{"text": "this is long enough", "tag": "ok"}]
        assert detect_text_fields(records) == ["text"]

    def test_ignores_short_strings(self):
        assert detect_text_fields([{"text": "short"}]) == []

    def test_ignores_non_string_values(self):
        records = [{"score": 0.99, "flag": True, "items": [1, 2]}]
        assert detect_text_fields(records) == []

    def test_uses_first_record_as_sample(self):
        # Only the first record is inspected.
        records = [{"a": "short"}, {"a": "this is much longer"}]
        assert detect_text_fields(records) == []


class TestNormalizeRecord:
    def test_strips_whitespace(self):
        assert normalize_record({"a": "  hello  "}) == {"a": "hello"}

    def test_converts_non_strings_to_strings(self):
        assert normalize_record({"n": 1, "f": 2.5, "b": True}) == {
            "n": "1",
            "f": "2.5",
            "b": "True",
        }

    def test_replaces_none_with_empty_string(self):
        assert normalize_record({"a": None, "b": "x"}) == {"a": "", "b": "x"}

    def test_empty_record_is_returned_empty(self):
        assert normalize_record({}) == {}


class TestValidateRecord:
    def test_record_with_content_is_valid(self):
        assert validate_record({"text": "hello"}) is True

    def test_record_with_only_empty_strings_is_invalid(self):
        assert validate_record({"a": "", "b": ""}) is False

    def test_empty_record_is_invalid(self):
        assert validate_record({}) is False

    def test_zero_value_is_treated_as_empty(self):
        # ``bool(0)`` is False — the validator treats this as empty.
        assert validate_record({"n": 0, "s": ""}) is False

    def test_truthy_non_string_values_are_valid(self):
        assert validate_record({"n": 1, "s": ""}) is True
