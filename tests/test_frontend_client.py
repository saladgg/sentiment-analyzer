"""Tests for the Streamlit frontend's HTTP client (``frontend/api_client.py``)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


@pytest.fixture
def client_module(monkeypatch):
    monkeypatch.syspath_prepend(str(FRONTEND))
    monkeypatch.setenv("ASA_API_URL", "http://api.test/api/v1/")
    monkeypatch.setenv("ASA_API_KEY", "k1")
    sys.modules.pop("api_client", None)
    module = importlib.import_module("api_client")
    yield module
    sys.modules.pop("api_client", None)


class _Resp:
    def __init__(self, status=200, body=None, text="", headers=None, content=b""):
        self.status_code = status
        self.ok = status < 400
        self._body = body
        self.text = text
        self.reason = "Error"
        self.headers = headers or {}
        self.content = content

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _capture(monkeypatch, module, response):
    calls = []

    def _request(method, url, **kwargs):
        calls.append(SimpleNamespace(method=method, url=url, **kwargs))
        return response

    monkeypatch.setattr(module.requests, "request", _request)
    return calls


def test_config_from_environment(client_module):
    assert client_module.BASE_URL == "http://api.test/api/v1"
    assert client_module._headers() == {"X-API-Key": "k1"}


def test_analyze_json_sends_only_set_options(client_module, monkeypatch):
    calls = _capture(monkeypatch, client_module, _Resp(body={"run_id": "r"}))
    assert client_module.analyze_json([{"a": "b"}], engine="llm", text_fields=["a"]) == {
        "run_id": "r"
    }
    call = calls[0]
    assert call.url == "http://api.test/api/v1/analyze"
    assert call.json == {
        "data": [{"a": "b"}],
        "source_type": "json",
        "sentiment_engine": "llm",
        "text_fields": ["a"],
        "granularity": "day",
    }
    assert call.headers == {"X-API-Key": "k1"}


def test_upload_params(client_module, monkeypatch):
    calls = _capture(monkeypatch, client_module, _Resp(body={}))
    file = SimpleNamespace(name="x.csv")
    client_module.analyze_csv(file, engine="rule_based", text_fields=["a", "b"])
    assert calls[0].params == {
        "sentiment_engine": "rule_based",
        "text_fields": "a,b",
        "granularity": "day",
    }
    client_module.analyze_excel(file)
    assert calls[1].url.endswith("/analyze/excel")


def test_error_detail_is_surfaced(client_module, monkeypatch):
    _capture(
        monkeypatch,
        client_module,
        _Resp(413, {"detail": "too big"}, headers={"X-Request-ID": "rid"}),
    )
    with pytest.raises(client_module.APIError) as info:
        client_module.list_runs()
    assert info.value.status_code == 413
    assert str(info.value) == "too big (request rid)"


def test_validation_errors_are_flattened(client_module, monkeypatch):
    body = {"detail": [{"loc": ["body", "data"], "msg": "field required"}]}
    _capture(monkeypatch, client_module, _Resp(422, body))
    with pytest.raises(client_module.APIError, match="data: field required"):
        client_module.analyze_json([])


def test_non_json_error_and_raw_download(client_module, monkeypatch):
    _capture(monkeypatch, client_module, _Resp(502, None, text="bad gateway"))
    with pytest.raises(client_module.APIError, match="bad gateway"):
        client_module.list_engines()

    _capture(monkeypatch, client_module, _Resp(200, None, content=b"a,b\n"))
    assert client_module.export_run("r1") == b"a,b\n"


def test_connection_error(client_module, monkeypatch):
    def _down(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(client_module.requests, "request", _down)
    with pytest.raises(client_module.APIError, match="Cannot reach the API"):
        client_module.get_run("r")


def test_rag_helpers(client_module, monkeypatch):
    calls = _capture(monkeypatch, client_module, _Resp(body=[]))
    client_module.upload_rag_document(SimpleNamespace(name="d.txt"), namespace="ns")
    client_module.list_rag_documents()
    client_module.delete_rag_document("d1")
    client_module.rag_retrieve("q", top_k=2, namespace="ns")
    client_module.compare_runs("a", "b")
    assert [c.method for c in calls] == ["POST", "GET", "DELETE", "POST", "GET"]
    assert calls[3].json == {"query": "q", "top_k": 2, "namespace": "ns"}


def test_job_and_ab_paths(client_module, monkeypatch):
    calls = _capture(monkeypatch, client_module, _Resp(body={"job_id": "j"}))
    client_module.analyze_json("a\nb", source_type="text", path="/jobs/analyze", use_rag=False)
    assert calls[0].url.endswith("/jobs/analyze")
    assert calls[0].json["use_rag"] is False
    client_module.analyze_json([{"t": "x"}], path="/runs/ab")
    assert calls[1].url.endswith("/runs/ab")
    assert "use_rag" not in calls[1].json

    file = SimpleNamespace(name="x.csv")
    client_module.analyze_csv(file, background=True, use_rag=True)
    assert calls[2].url.endswith("/jobs/analyze/csv")
    assert calls[2].params["use_rag"] == "true"
    client_module.analyze_excel(file, background=True)
    assert calls[3].url.endswith("/jobs/analyze/excel")


def test_job_helpers(client_module, monkeypatch):
    calls = _capture(monkeypatch, client_module, _Resp(body={}))
    client_module.get_job("j1")
    client_module.list_jobs(limit=5, status="queued")
    client_module.cancel_job("j1")
    assert [(c.method, c.url.rsplit("/api/v1", 1)[1]) for c in calls] == [
        ("GET", "/jobs/j1"),
        ("GET", "/jobs"),
        ("DELETE", "/jobs/j1"),
    ]
    assert calls[1].params == {"limit": 5, "status": "queued"}


def test_evaluate_and_namespace_filter(client_module, monkeypatch):
    calls = _capture(monkeypatch, client_module, _Resp(body={}))
    client_module.evaluate_engine([{"text": "x", "label": "neutral"}], engine="rule_based")
    assert calls[0].json == {
        "items": [{"text": "x", "label": "neutral"}],
        "sentiment_engine": "rule_based",
    }
    client_module.list_rag_documents("ops")
    assert calls[1].params == {"namespace": "ops"}
    client_module.list_rag_documents()
    assert calls[2].params == {}
