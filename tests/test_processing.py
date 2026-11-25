"""
Unit tests for the processing layer: text-field detection,
normalization, and validation.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from sentiment_analyzer.processing.field_detector import detect_text_fields
from sentiment_analyzer.processing.normalizer import is_missing, normalize_record
from sentiment_analyzer.processing.validator import validate_record


class TestDetectTextFields:
    def test_returns_empty_when_no_records(self):
        assert detect_text_fields([]) == []

    def test_finds_free_text_fields(self):
        records = [{"comment": "this is a longer comment", "id": 1}]
        assert detect_text_fields(records) == ["comment"]

    def test_ignores_short_strings(self):
        assert detect_text_fields([{"code": "ABC"}, {"code": "XYZ"}]) == []

    def test_ignores_numeric_columns(self):
        assert detect_text_fields([{"n": 123456789012, "f": 1.5}]) == []

    def test_looks_past_a_blank_first_record(self):
        """BUG-3: the first record used to be the only sample."""
        records: list[dict[str, Any]] = [
            {"id": 1, "comment": None},
            {"id": 2, "comment": "terrible service today"},
            {"id": 3, "comment": "really great support team"},
        ]
        assert detect_text_fields(records) == ["comment"]

    def test_ignores_categorical_columns(self):
        records = [
            {"category": "Electronics", "review": f"review number {i} text"} for i in range(10)
        ]
        assert detect_text_fields(records) == ["review"]

    def test_ignores_code_like_values(self):
        records = [
            {"sku": f"{i:010d}-{i:05d}", "note": f"fine product number {i}"} for i in range(6)
        ]
        assert detect_text_fields(records) == ["note"]

    def test_keeps_column_order(self):
        records = [{"title": "short title text", "body": "a much longer body of text"}]
        assert detect_text_fields(records) == ["title", "body"]


class TestNormalizeRecord:
    def test_strips_whitespace(self):
        assert normalize_record({"a": "  hi  "}) == {"a": "hi"}

    def test_converts_non_strings_to_strings(self):
        assert normalize_record({"n": 3, "b": True}) == {"n": "3", "b": "True"}

    @pytest.mark.parametrize("missing", [None, float("nan"), pd.NaT, pd.NA])
    def test_missing_values_become_empty(self, missing):
        """BUG-4: NaN used to become the string 'nan'."""
        assert normalize_record({"a": missing}) == {"a": ""}

    def test_empty_record_is_returned_empty(self):
        assert normalize_record({}) == {}


class TestIsMissing:
    def test_scalars(self):
        assert is_missing(math.nan)
        assert not is_missing("nan")
        assert not is_missing(0)

    def test_containers_are_not_missing(self):
        assert not is_missing([None])
        assert not is_missing({})


class TestValidateRecord:
    def test_record_with_content_is_valid(self):
        assert validate_record({"a": "x"})

    def test_record_with_only_empty_strings_is_invalid(self):
        assert not validate_record({"a": "", "b": ""})

    def test_empty_record_is_invalid(self):
        assert not validate_record({})

    def test_text_fields_must_have_content(self):
        assert not validate_record({"id": "7", "comment": ""}, ["comment"])
        assert validate_record({"id": "7", "comment": "ok"}, ["comment"])
