"""
Tests for ingestion adapters.

Covers JSON, CSV and Excel adapters, plus the abstract base contract.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from sentiment_analyzer.ingestion.base import InputAdapter
from sentiment_analyzer.ingestion.csv_adapter import CSVAdapter
from sentiment_analyzer.ingestion.excel_adapter import ExcelAdapter
from sentiment_analyzer.ingestion.json_adapter import JSONAdapter


class TestJSONAdapter:
    def test_passes_through_list_of_dicts(self):
        adapter = JSONAdapter()
        records = [{"text": "hi"}, {"text": "bye"}]
        assert adapter.load(records) == records

    def test_empty_list_is_valid(self):
        assert JSONAdapter().load([]) == []

    @pytest.mark.parametrize("bad_input", ["not a list", {"x": 1}, 42, None])
    def test_rejects_non_list_input(self, bad_input):
        with pytest.raises(ValueError, match="list of objects"):
            JSONAdapter().load(bad_input)


class TestCSVAdapter:
    def test_loads_from_string_io(self):
        csv = io.StringIO("text,score\nhello,0.5\nworld,-0.5\n")
        records = CSVAdapter().load(csv)
        assert records == [
            {"text": "hello", "score": 0.5},
            {"text": "world", "score": -0.5},
        ]

    def test_loads_from_file_path(self, tmp_path):
        path = tmp_path / "input.csv"
        path.write_text("comment\ngreat\nawful\n")
        records = CSVAdapter().load(str(path))
        assert records == [{"comment": "great"}, {"comment": "awful"}]


class TestExcelAdapter:
    def test_loads_first_sheet(self, tmp_path):
        path = tmp_path / "input.xlsx"
        pd.DataFrame({"comment": ["good", "bad"]}).to_excel(path, index=False)
        records = ExcelAdapter().load(str(path))
        assert records == [{"comment": "good"}, {"comment": "bad"}]


class TestInputAdapterBase:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            InputAdapter()  # type: ignore[abstract]

    def test_subclass_must_implement_load(self):
        class Incomplete(InputAdapter):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]
