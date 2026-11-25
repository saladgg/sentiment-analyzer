"""
Integration tests for the FastAPI surface.

Each test gets a fresh app built from explicit test settings (isolated
DuckDB + Chroma, RAG disabled, no auth) unless it builds its own.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

API = "/api/v1"


def _csv(text: str) -> dict:
    return {"file": ("input.csv", io.BytesIO(text.encode()), "text/csv")}


def _xlsx(df: pd.DataFrame) -> dict:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return {
        "file": (
            "input.xlsx",
            buffer,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }


class TestAnalyzeJSON:
    def test_returns_record_per_input(self, client, analyze_payload):
        response = client.post(f"{API}/analyze", json=analyze_payload)
        assert response.status_code == 200

        payload = response.json()
        assert payload["run_id"]
        assert payload["engine"] == "rule_based"
        assert payload["text_fields"] == ["text"]
        assert len(payload["input_sha256"]) == 64
        assert [r["sentiment"] for r in payload["records"]] == ["positive", "negative", "neutral"]
        record = payload["records"][0]
        assert record["contributing_fields"] == {"text": 1.0}
        assert record["drivers"]["positive_hits"] == 1.0
        assert record["confidence"] == 1.0

    def test_summary_counts_match_records(self, client, analyze_payload):
        body = client.post(f"{API}/analyze", json=analyze_payload).json()
        assert body["summary"]["total_records"] == len(body["records"])
        assert body["summary"]["failed_records"] == 0
        assert body["rag"] is None

    def test_invalid_payload_returns_422(self, client):
        response = client.post(f"{API}/analyze", json={"data": []})
        assert response.status_code == 422

    def test_unknown_engine_returns_422(self, client):
        response = client.post(
            f"{API}/analyze",
            json={"source_type": "json", "sentiment_engine": "nope", "data": [{"t": "hi"}]},
        )
        assert response.status_code == 422

    def test_engine_defaults_to_configured_default(self, make_app, analyze_payload):
        payload = {k: v for k, v in analyze_payload.items() if k != "sentiment_engine"}
        with TestClient(make_app(default_sentiment_engine="hf_transformer")) as c:
            body = c.post(f"{API}/analyze", json=payload).json()
        assert body["engine"] == "hf_transformer"
        assert "model_confidence" in body["records"][0]["drivers"]

    def test_unknown_source_type_returns_422(self, client):
        response = client.post(f"{API}/analyze", json={"source_type": "pdf", "data": [{}]})
        assert response.status_code == 422

    def test_explicit_text_fields(self, client):
        response = client.post(
            f"{API}/analyze",
            json={
                "source_type": "json",
                "text_fields": ["title", "body"],
                "data": [{"title": "Great!", "body": "but the box was bad", "id": "x1"}],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["text_fields"] == ["title", "body"]
        assert body["records"][0]["contributing_fields"] == {"title": 1.0, "body": -1.0}

    def test_unknown_text_field_returns_422(self, client):
        response = client.post(
            f"{API}/analyze",
            json={"source_type": "json", "text_fields": ["nope"], "data": [{"a": "hello"}]},
        )
        assert response.status_code == 422
        assert "Available fields" in response.json()["detail"]

    def test_no_detectable_text_field_returns_422(self, client):
        response = client.post(
            f"{API}/analyze", json={"source_type": "json", "data": [{"id": 1, "n": 2}]}
        )
        assert response.status_code == 422
        assert "text_fields" in response.json()["detail"]

    def test_trend_by_timestamp_field(self, client):
        response = client.post(
            f"{API}/analyze",
            json={
                "source_type": "json",
                "timestamp_field": "at",
                "granularity": "month",
                "data": [
                    {"text": "good product overall", "at": "2026-01-05"},
                    {"text": "bad product overall", "at": "2026-01-20"},
                    {"text": "good service today", "at": "2026-02-01"},
                    {"text": "a plain sentence", "at": "not a date"},
                ],
            },
        )
        assert response.status_code == 200
        trend = response.json()["trend"]
        assert [b["bucket"] for b in trend["buckets"]] == ["2026-01", "2026-02"]
        assert trend["buckets"][0]["avg_score"] == 0.0
        assert trend["unparseable"] == 1

    def test_unknown_timestamp_field_returns_422(self, client, analyze_payload):
        response = client.post(f"{API}/analyze", json={**analyze_payload, "timestamp_field": "x"})
        assert response.status_code == 422


class TestAnalyzeText:
    def test_plain_text_one_record_per_line(self, client):
        response = client.post(
            f"{API}/analyze", json={"source_type": "text", "data": "I love it\n\nterrible!\nok"}
        )
        assert response.status_code == 200
        labels = [r["sentiment"] for r in response.json()["records"]]
        # Short lines are still analyzed: text input skips field detection.
        assert labels == ["positive", "negative", "neutral"]

    def test_text_source_requires_string(self, client):
        response = client.post(f"{API}/analyze", json={"source_type": "text", "data": [{}]})
        assert response.status_code == 422

    def test_json_source_rejects_string(self, client):
        response = client.post(f"{API}/analyze", json={"source_type": "json", "data": "hi"})
        assert response.status_code == 422


class TestAnalyzeCSV:
    def test_uploads_and_analyzes_csv(self, client):
        response = client.post(
            f"{API}/analyze/csv", files=_csv("comment\ngreat product\nawful service\n")
        )
        assert response.status_code == 200
        assert len(response.json()["records"]) == 2

    def test_engine_query_param_is_honoured(self, client):
        response = client.post(
            f"{API}/analyze/csv",
            files=_csv("comment\ngood product overall\n"),
            params={"sentiment_engine": "hf_transformer"},
        )
        assert response.status_code == 200
        assert response.json()["engine"] == "hf_transformer"

    def test_unknown_engine_query_param_returns_422(self, client):
        response = client.post(
            f"{API}/analyze/csv", files=_csv("c\nhello\n"), params={"sentiment_engine": "nope"}
        )
        assert response.status_code == 422

    def test_blank_first_row_and_missing_cells(self, client):
        """BUG-3/BUG-4: a blank first comment must not hide the column; NaN is not 'nan'."""
        response = client.post(
            f"{API}/analyze/csv",
            files=_csv("id,comment\n1,\n2,this is terrible service\n3,really great support\n"),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["text_fields"] == ["comment"]
        assert [r["record_id"] for r in body["records"]] == [1, 2]
        assert [r["sentiment"] for r in body["records"]] == ["negative", "positive"]

    def test_text_fields_query_param(self, client):
        response = client.post(
            f"{API}/analyze/csv",
            files=_csv("a,b\nok,good\nfine,bad\n"),
            params={"text_fields": "b"},
        )
        assert response.status_code == 200
        assert response.json()["text_fields"] == ["b"]

    def test_rejects_non_csv_filename(self, client):
        response = client.post(
            f"{API}/analyze/csv",
            files={"file": ("not_a_csv.txt", io.BytesIO(b"hi"), "text/plain")},
        )
        assert response.status_code == 415

    def test_rejects_non_utf8(self, client):
        response = client.post(
            f"{API}/analyze/csv",
            files={"file": ("x.csv", io.BytesIO("comment\ncafé\n".encode("utf-16")), "text/csv")},
        )
        assert response.status_code == 415


class TestAnalyzeExcel:
    def test_uploads_and_analyzes_excel(self, client):
        df = pd.DataFrame({"comment": ["great product here", None, "awful service again"]})
        response = client.post(f"{API}/analyze/excel", files=_xlsx(df))
        assert response.status_code == 200
        assert [r["sentiment"] for r in response.json()["records"]] == ["positive", "negative"]

    def test_rejects_non_excel_filename(self, client):
        response = client.post(
            f"{API}/analyze/excel",
            files={"file": ("not_excel.csv", io.BytesIO(b"hi"), "text/csv")},
        )
        assert response.status_code == 415

    def test_rejects_corrupt_excel(self, client):
        response = client.post(
            f"{API}/analyze/excel",
            files={"file": ("bad.xlsx", io.BytesIO(b"not a zip"), "application/octet-stream")},
        )
        assert response.status_code == 422


class TestInputLimits:
    def test_too_many_json_records(self, make_app):
        with TestClient(make_app(max_records=2)) as c:
            response = c.post(
                f"{API}/analyze",
                json={"source_type": "json", "data": [{"t": "good one"}] * 3},
            )
        assert response.status_code == 413
        assert response.json()["code"] == "input_too_large"

    def test_too_many_csv_rows(self, make_app):
        with TestClient(make_app(max_records=2)) as c:
            response = c.post(f"{API}/analyze/csv", files=_csv("c\na b c\nd e f\ng h i\n"))
        assert response.status_code == 413

    def test_upload_too_large(self, make_app):
        with TestClient(make_app(max_upload_bytes=64)) as c:
            response = c.post(f"{API}/analyze/csv", files=_csv("comment\n" + "word " * 100))
        assert response.status_code == 413

    def test_json_body_too_large(self, make_app):
        with TestClient(make_app(max_upload_bytes=200)) as c:
            response = c.post(
                f"{API}/analyze",
                json={"source_type": "text", "data": "good " * 200},
            )
        assert response.status_code == 413

    def test_llm_record_cap(self, make_app, fake_llm):
        with TestClient(make_app(max_llm_records=1)) as c:
            response = c.post(
                f"{API}/analyze",
                json={"source_type": "text", "sentiment_engine": "llm", "data": "a\nb"},
            )
        assert response.status_code == 413
        assert fake_llm["calls"] == []

    def test_long_text_is_truncated_and_flagged(self, make_app):
        with TestClient(make_app(max_text_chars=10)) as c:
            body = c.post(
                f"{API}/analyze",
                json={"source_type": "text", "data": "terrible " + "good " * 20},
            ).json()
        record = body["records"][0]
        assert record["truncated"] is True
        assert record["sentiment"] == "negative"  # only "terrible " fits in 10 chars


class TestLLMEngineViaAPI:
    def test_llm_unconfigured_returns_503(self, make_app):
        with TestClient(make_app(llm_model="")) as c:
            response = c.post(
                f"{API}/analyze",
                json={"source_type": "text", "sentiment_engine": "llm", "data": "good"},
            )
        assert response.status_code == 503
        assert response.json()["code"] == "engine_unavailable"

    def test_per_record_failure_does_not_fail_the_run(self, client, monkeypatch):
        from sentiment_analyzer.sentiment import llm

        from tests.conftest import llm_response

        def _completion(**kwargs):
            if "boom" in kwargs["messages"][1]["content"]:
                return llm_response("not json at all")
            return llm_response({"label": "positive", "score": 0.5})

        monkeypatch.setattr(llm.litellm, "completion", _completion)
        body = client.post(
            f"{API}/analyze",
            json={"source_type": "text", "sentiment_engine": "llm", "data": "fine\nboom"},
        ).json()
        assert body["summary"]["failed_records"] == 1
        assert body["records"][1]["error"].startswith("EngineOutputError")
        assert body["records"][1]["sentiment"] is None
        assert body["records"][0]["sentiment"] == "positive"
        assert "could not be analyzed" in body["explanations"][1]


class TestRuns:
    def test_list_is_paginated(self, client, analyze_payload):
        for _ in range(3):
            client.post(f"{API}/analyze", json=analyze_payload)

        page = client.get(f"{API}/runs", params={"limit": 2}).json()
        assert page["total"] == 3
        assert len(page["items"]) == 2
        assert page["items"][0]["status"] == "completed"
        assert page["items"][0]["created_at"].endswith(("Z", "+00:00"))

        rest = client.get(f"{API}/runs", params={"limit": 2, "offset": 2}).json()
        assert len(rest["items"]) == 1

    def test_list_filters_by_engine(self, client, analyze_payload):
        client.post(f"{API}/analyze", json=analyze_payload)
        client.post(
            f"{API}/analyze", json={**analyze_payload, "sentiment_engine": "hf_transformer"}
        )
        page = client.get(f"{API}/runs", params={"engine": "hf_transformer"}).json()
        assert page["total"] == 1
        assert page["items"][0]["engine"] == "hf_transformer"

    def test_run_detail_includes_records_and_config(self, client, analyze_payload):
        run_id = client.post(f"{API}/analyze", json=analyze_payload).json()["run_id"]
        detail = client.get(f"{API}/runs/{run_id}", params={"records_limit": 2}).json()
        assert detail["records_total"] == 3
        assert len(detail["records"]) == 2
        assert detail["config"]["text_fields"] == ["text"]
        assert detail["config"]["engine"] == "rule_based"
        assert detail["records_in"] == 3
        assert detail["source_type"] == "json"

    def test_run_detail_404(self, client):
        response = client.get(f"{API}/runs/missing")
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_export_csv_and_json(self, client, analyze_payload):
        run_id = client.post(f"{API}/analyze", json=analyze_payload).json()["run_id"]

        csv_response = client.get(f"{API}/runs/{run_id}/export")
        assert csv_response.status_code == 200
        assert csv_response.headers["content-type"].startswith("text/csv")
        assert "attachment" in csv_response.headers["content-disposition"]
        lines = csv_response.text.strip().splitlines()
        assert lines[0].startswith("record_id,sentiment,score")
        assert len(lines) == 4

        json_response = client.get(f"{API}/runs/{run_id}/export", params={"format": "json"})
        assert json_response.json()["run_id"] == run_id
        assert len(json_response.json()["records"]) == 3

    def test_export_404(self, client):
        assert client.get(f"{API}/runs/nope/export").status_code == 404


class TestRunCompareEndpoint:
    def test_compares_two_runs_of_the_same_input(self, client, analyze_payload):
        first = client.post(f"{API}/analyze", json=analyze_payload).json()["run_id"]
        second = client.post(
            f"{API}/analyze", json={**analyze_payload, "sentiment_engine": "hf_transformer"}
        ).json()["run_id"]

        response = client.get(
            f"{API}/runs/compare", params={"base_run_id": first, "rag_run_id": second}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["comparison"]["records_compared"] == 3
        # rule_based: positive/negative/neutral; HF stub: positive/negative/negative
        assert body["comparison"]["label_changes"] == 1
        assert body["changed_records"][0]["record_id"] == 2

    def test_different_inputs_conflict(self, client, analyze_payload):
        first = client.post(f"{API}/analyze", json=analyze_payload).json()["run_id"]
        second = client.post(
            f"{API}/analyze", json={"source_type": "text", "data": "something else entirely"}
        ).json()["run_id"]
        response = client.get(
            f"{API}/runs/compare", params={"base_run_id": first, "rag_run_id": second}
        )
        assert response.status_code == 409

    def test_404_when_runs_missing(self, client):
        response = client.get(
            f"{API}/runs/compare", params={"base_run_id": "missing-a", "rag_run_id": "missing-b"}
        )
        assert response.status_code == 404


class TestRagABTest:
    def test_ab_test_runs_both_and_compares(self, make_app, fake_llm):
        with TestClient(make_app(rag_enabled=True)) as c:
            c.post(
                f"{API}/rag/documents",
                files={
                    "file": (
                        "p.txt",
                        io.BytesIO(b"Our refund policy takes thirty days."),
                        "text/plain",
                    )
                },
            )
            response = c.post(
                f"{API}/runs/ab",
                json={
                    "source_type": "text",
                    "sentiment_engine": "llm",
                    "data": "the refund was fine\na plain day",
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["base"]["rag"]["enabled"] is False
        assert body["rag"]["rag"]["enabled"] is True
        assert body["rag"]["rag"]["used"] is True
        assert body["comparison"]["comparison"]["records_compared"] == 2
        assert body["comparison"]["comparison"]["label_changes"] >= 1

    def test_ab_test_rejects_engines_that_ignore_context(self, client, analyze_payload):
        response = client.post(f"{API}/runs/ab", json=analyze_payload)
        assert response.status_code == 422

    def test_ab_test_requires_rag_enabled(self, client, fake_llm):
        response = client.post(
            f"{API}/runs/ab",
            json={"source_type": "text", "sentiment_engine": "llm", "data": "good"},
        )
        assert response.status_code == 422


class TestEvaluationEndpoints:
    def test_accuracy_with_report(self, client):
        response = client.post(
            f"{API}/evaluation/sentiment",
            json={
                "predictions": ["positive", "negative", "positive"],
                "labels": ["positive", "negative", "negative"],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["metric"] == "accuracy"
        assert body["total_samples"] == 3
        assert body["value"] == pytest.approx(2 / 3, abs=1e-4)
        assert body["report"]["confusion_matrix"]["labels"] == ["negative", "positive"]

    def test_mismatched_lengths_return_422(self, client):
        response = client.post(
            f"{API}/evaluation/sentiment",
            json={"predictions": ["positive"], "labels": ["positive", "negative"]},
        )
        assert response.status_code == 422

    def test_engine_evaluation(self, client):
        response = client.post(
            f"{API}/evaluation/engine",
            json={
                "sentiment_engine": "rule_based",
                "items": [
                    {"text": "great stuff", "label": "positive"},
                    {"text": "awful stuff", "label": "negative"},
                    {"text": "great but awful", "label": "negative"},
                ],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["engine"] == "rule_based"
        assert body["failed"] == 0
        assert body["report"]["accuracy"] == pytest.approx(2 / 3, abs=1e-4)

    def test_engine_evaluation_all_failed(self, client):
        response = client.post(
            f"{API}/evaluation/engine",
            json={
                "sentiment_engine": "hf_transformer",
                "items": [{"text": "explode", "label": "positive"}],
            },
        )
        assert response.status_code == 200
        assert response.json()["failed"] == 1
        assert response.json()["report"]["total_samples"] == 0

    def test_engine_evaluation_limit(self, make_app):
        with TestClient(make_app(max_records=1)) as c:
            response = c.post(
                f"{API}/evaluation/engine",
                json={"items": [{"text": "a", "label": "neutral"}] * 2},
            )
        assert response.status_code == 413


class TestEnginesEndpoint:
    def test_lists_engines_and_limits(self, client):
        body = client.get(f"{API}/engines").json()
        names = [e["name"] for e in body["engines"]]
        assert names == ["rule_based", "hf_transformer", "llm"]
        default = next(e for e in body["engines"] if e["default"])
        assert default["name"] == "rule_based"
        llm = next(e for e in body["engines"] if e["name"] == "llm")
        assert llm["uses_context"] is True and llm["is_paid"] is True
        assert body["limits"]["max_records"] == 10_000
        assert body["rag_enabled"] is False


class TestAuth:
    def test_requires_key_when_configured(self, make_app, analyze_payload):
        with TestClient(make_app(api_keys=["s3cret", "other"])) as c:
            assert c.post(f"{API}/analyze", json=analyze_payload).status_code == 401
            bad = c.post(f"{API}/analyze", json=analyze_payload, headers={"X-API-Key": "nope"})
            assert bad.status_code == 401
            ok = c.post(f"{API}/analyze", json=analyze_payload, headers={"X-API-Key": "other"})
            assert ok.status_code == 200
            # Health probes stay open for orchestrators.
            assert c.get("/healthz").status_code == 200

    def test_prod_without_keys_refuses_to_start(self, make_settings):
        from sentiment_analyzer.main import create_app

        with pytest.raises(RuntimeError, match="ASA_API_KEYS"):
            create_app(make_settings(environment="prod"))

    def test_prod_hides_docs(self, make_app):
        with TestClient(make_app(environment="prod", api_keys=["k"])) as c:
            assert c.get("/docs").status_code == 404
            assert c.get("/openapi.json").status_code == 404

    def test_dev_serves_docs(self, client):
        assert client.get("/docs").status_code == 200


class TestPlatform:
    def test_healthz(self, client):
        body = client.get("/healthz").json()
        assert body["status"] == "ok"
        assert body["version"]

    def test_readyz(self, client):
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["checks"] == {
            "runs_db": "ok",
            "vector_store": "ok",
            "default_engine": "ok",
        }

    def test_readyz_reports_broken_engine(self, make_app):
        with TestClient(make_app(default_sentiment_engine="llm", llm_model="")) as c:
            response = c.get("/readyz")
        assert response.status_code == 503
        assert response.json()["checks"]["default_engine"].startswith("error")

    def test_legacy_api_prefix_still_works(self, client, analyze_payload):
        assert client.post("/api/analyze", json=analyze_payload).status_code == 200
        assert "/api/analyze" not in client.get("/openapi.json").json()["paths"]
        assert f"{API}/analyze" in client.get("/openapi.json").json()["paths"]

    def test_request_id_is_returned_and_honoured(self, client):
        generated = client.get("/healthz").headers["x-request-id"]
        assert len(generated) == 32
        echoed = client.get("/healthz", headers={"X-Request-ID": "abc-123"})
        assert echoed.headers["x-request-id"] == "abc-123"

    def test_unexpected_errors_return_generic_500(self, make_app):
        from sentiment_analyzer.api.deps import get_service

        app = make_app()

        def _broken():
            raise RuntimeError("secret internals")

        app.dependency_overrides[get_service] = _broken
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post(f"{API}/analyze", json={"source_type": "text", "data": "x"})
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error"
        assert "secret" not in response.text
        assert response.json()["request_id"]

    def test_cors_preflight(self, client):
        response = client.options(
            f"{API}/analyze",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
        assert "access-control-allow-credentials" not in response.headers
