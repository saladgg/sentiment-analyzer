"""Tests for the Prometheus /metrics endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

API = "/api/v1"


def _metric(text: str, name: str, **labels: str) -> float:
    """Value of the first sample of ``name`` whose labels include ``labels``."""
    for line in text.splitlines():
        if not line.startswith(name + "{") and not line.startswith(name + " "):
            continue
        if all(f'{k}="{v}"' in line for k, v in labels.items()):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"{name} {labels} not found")


def test_http_and_run_metrics(client, analyze_payload):
    client.post(f"{API}/analyze", json=analyze_payload)
    run_id = client.get(f"{API}/runs").json()["items"][0]["run_id"]
    client.get(f"{API}/runs/{run_id}")

    text = client.get("/metrics").text
    # Route templates, never raw ids, as labels.
    assert (
        _metric(text, "asa_http_requests_total", route="/api/v1/runs/{run_id}", status="200") == 1
    )
    assert run_id not in text
    assert _metric(text, "asa_analysis_runs_total", engine="rule_based", status="completed") == 1
    assert _metric(text, "asa_records_analyzed_total", engine="rule_based", outcome="ok") == 3
    assert _metric(text, "asa_engine_batch_duration_seconds_count", engine="rule_based") >= 1


def test_llm_usage_metrics(client, fake_llm):
    client.post(
        f"{API}/analyze",
        json={"source_type": "text", "sentiment_engine": "llm", "data": "good\ngood\nbad"},
    )
    text = client.get("/metrics").text
    assert _metric(text, "asa_llm_calls_total", source="api") == 2
    assert _metric(text, "asa_llm_calls_total", source="cache") == 1
    assert _metric(text, "asa_llm_tokens_total", kind="prompt") == 20
    assert _metric(text, "asa_llm_cost_usd_total") == 0.002


def test_metrics_require_a_key_when_auth_is_on(make_app):
    with TestClient(make_app(api_keys=["k"], rate_limit_per_minute=1)) as c:
        assert c.get("/metrics").status_code == 401
        # Scrapes are not rate limited.
        for _ in range(3):
            assert c.get("/metrics", headers={"X-API-Key": "k"}).status_code == 200


def test_metrics_not_in_openapi(client):
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]
