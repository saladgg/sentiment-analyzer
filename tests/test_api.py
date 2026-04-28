"""
Integration tests for the FastAPI surface.

The ``client`` fixture (defined in ``conftest.py``) builds a FastAPI
``TestClient`` against an isolated DuckDB and with RAG disabled, so
these tests never touch the developer's local state.
"""

from __future__ import annotations

import io


class TestAnalyzeJSON:
    def test_returns_record_per_input(self, client, analyze_payload):
        response = client.post("/api/analyze", json=analyze_payload)
        assert response.status_code == 200

        payload = response.json()
        assert payload["run_id"]
        assert len(payload["records"]) == len(analyze_payload["data"])

        record = payload["records"][0]
        assert "contributing_fields" in record
        assert record["sentiment"] in {"positive", "negative", "neutral"}

    def test_summary_counts_match_records(self, client, analyze_payload):
        response = client.post("/api/analyze", json=analyze_payload)
        body = response.json()
        assert body["summary"]["total_records"] == len(body["records"])

    def test_invalid_payload_returns_422(self, client):
        # ``source_type`` is required.
        response = client.post("/api/analyze", json={"data": []})
        assert response.status_code == 422

    def test_unknown_engine_propagates_keyerror(self, client):
        # ``TestClient`` re-raises server-side exceptions by default — assert
        # that an unknown engine surfaces as a ``KeyError`` rather than being
        # silently swallowed.
        import pytest

        with pytest.raises(KeyError):
            client.post(
                "/api/analyze",
                json={
                    "source_type": "json",
                    "sentiment_engine": "does-not-exist",
                    "data": [{"text": "hello"}],
                },
            )


class TestAnalyzeCSV:
    def test_uploads_and_analyzes_csv(self, client):
        csv = io.BytesIO(b"comment\ngreat product\nawful service\n")
        response = client.post(
            "/api/analyze/csv",
            files={"file": ("input.csv", csv, "text/csv")},
            data={"sentiment_engine": "rule_based"},
        )
        assert response.status_code == 200
        assert len(response.json()["records"]) == 2

    def test_rejects_non_csv_filename(self, client):
        response = client.post(
            "/api/analyze/csv",
            files={"file": ("not_a_csv.txt", io.BytesIO(b"hi"), "text/plain")},
        )
        assert response.status_code == 400


class TestAnalyzeExcel:
    def test_rejects_non_excel_filename(self, client):
        response = client.post(
            "/api/analyze/excel",
            files={"file": ("not_excel.csv", io.BytesIO(b"hi"), "text/csv")},
        )
        assert response.status_code == 400


class TestRunListing:
    def test_runs_endpoint_returns_array(self, client, analyze_payload):
        # Ensure at least one run exists.
        client.post("/api/analyze", json=analyze_payload)

        response = client.get("/api/runs")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestEvaluationEndpoint:
    def test_returns_accuracy_metric(self, client):
        response = client.post(
            "/api/evaluation/sentiment",
            json={
                "predictions": ["positive", "negative", "positive"],
                "labels": ["positive", "negative", "negative"],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["metric"] == "accuracy"
        assert body["total_samples"] == 3
        assert 0.0 <= body["value"] <= 1.0


class TestRunCompareEndpoint:
    def test_404_when_runs_missing(self, client):
        response = client.get(
            "/api/runs/compare",
            params={"base_run_id": "missing-a", "rag_run_id": "missing-b"},
        )
        assert response.status_code == 404
