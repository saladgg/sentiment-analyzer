"""
Unit tests for the ingestion adapters.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from sentiment_analyzer.core.errors import InvalidInputError, UnsupportedInputError
from sentiment_analyzer.ingestion.base import InputAdapter
from sentiment_analyzer.ingestion.csv_adapter import CSVAdapter
from sentiment_analyzer.ingestion.excel_adapter import ExcelAdapter
from sentiment_analyzer.ingestion.json_adapter import JSONAdapter
from sentiment_analyzer.ingestion.text_adapter import TextAdapter


class TestJSONAdapter:
    def test_passes_through_list_of_dicts(self):
        data = [{"a": 1}, {"b": 2}]
        assert JSONAdapter().load(data) == data

    def test_empty_list_is_valid(self):
        assert JSONAdapter().load([]) == []

    @pytest.mark.parametrize("bad_input", ["text", {"a": 1}, None, 42, [1, 2]])
    def test_rejects_non_list_of_objects(self, bad_input):
        with pytest.raises(ValueError):
            JSONAdapter().load(bad_input)


class TestTextAdapter:
    def test_one_record_per_non_empty_line(self):
        assert TextAdapter().load("first\n\n  second  \n") == [
            {"text": "first"},
            {"text": "second"},
        ]

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            TextAdapter().load(["x"])


class TestCSVAdapter:
    def test_loads_from_string_io(self):
        records = CSVAdapter().load(io.StringIO("name,comment\nalice,hi\nbob,bye\n"))
        assert records == [{"name": "alice", "comment": "hi"}, {"name": "bob", "comment": "bye"}]

    def test_missing_cells_become_none(self):
        records = CSVAdapter().load(io.StringIO("a,b\n1,\n,x\n"))
        assert records == [{"a": 1.0, "b": None}, {"a": None, "b": "x"}]

    def test_loads_from_file_path(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("x\n1\n2\n")
        assert CSVAdapter().load(str(path)) == [{"x": 1}, {"x": 2}]

    def test_reads_at_most_max_records_plus_one(self):
        records = CSVAdapter(max_records=2).load(io.StringIO("x\n1\n2\n3\n4\n5\n"))
        assert len(records) == 3

    def test_encoding_and_parse_errors(self):
        with pytest.raises(UnsupportedInputError):
            CSVAdapter().load(io.BytesIO("a\nü\n".encode("utf-16")))
        with pytest.raises(InvalidInputError):
            CSVAdapter().load(io.StringIO(""))


class TestExcelAdapter:
    def test_loads_first_sheet(self, tmp_path):
        path = tmp_path / "data.xlsx"
        pd.DataFrame({"comment": ["good", None, "bad"]}).to_excel(path, index=False)
        assert ExcelAdapter().load(str(path)) == [
            {"comment": "good"},
            {"comment": None},
            {"comment": "bad"},
        ]

    def test_corrupt_file(self):
        with pytest.raises(InvalidInputError):
            ExcelAdapter().load(io.BytesIO(b"definitely not xlsx"))


class TestInputAdapterBase:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            InputAdapter()  # type: ignore[abstract]

    def test_subclass_must_implement_load(self):
        class Incomplete(InputAdapter):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]
