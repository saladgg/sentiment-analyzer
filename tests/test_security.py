"""Tests for API-key scopes and rate limiting."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentiment_analyzer.core.security import RateLimiter

API = "/api/v1"


class TestReadOnlyKeys:
    @pytest.fixture
    def client(self, make_app):
        with TestClient(make_app(api_keys=["admin"], api_read_keys=["viewer"])) as c:
            yield c

    def test_read_key_can_read(self, client):
        assert client.get(f"{API}/runs", headers={"X-API-Key": "viewer"}).status_code == 200
        assert client.get(f"{API}/engines", headers={"X-API-Key": "viewer"}).status_code == 200

    def test_read_key_cannot_write(self, client, analyze_payload):
        response = client.post(
            f"{API}/analyze", json=analyze_payload, headers={"X-API-Key": "viewer"}
        )
        assert response.status_code == 403
        assert response.json()["code"] == "forbidden"
        assert (
            client.delete(f"{API}/rag/documents/x", headers={"X-API-Key": "viewer"}).status_code
            == 403
        )

    def test_full_key_can_write(self, client, analyze_payload):
        response = client.post(
            f"{API}/analyze", json=analyze_payload, headers={"X-API-Key": "admin"}
        )
        assert response.status_code == 200

    def test_unknown_key(self, client):
        assert client.get(f"{API}/runs", headers={"X-API-Key": "nope"}).status_code == 401

    def test_read_keys_alone_are_rejected_at_startup(self, make_settings):
        from sentiment_analyzer.main import create_app

        with pytest.raises(RuntimeError, match="ASA_API_READ_KEYS"):
            create_app(make_settings(api_read_keys=["viewer"]))


class TestRateLimiter:
    def test_token_bucket(self):
        now = [0.0]
        limiter = RateLimiter(per_minute=2, clock=lambda: now[0])
        assert limiter.acquire("a") == 0
        assert limiter.acquire("a") == 0
        wait = limiter.acquire("a")
        assert wait == pytest.approx(30.0)  # one token refills every 30 s
        assert limiter.acquire("b") == 0  # clients are independent
        now[0] = 30.0
        assert limiter.acquire("a") == 0

    def test_prunes_idle_clients(self, monkeypatch):
        now = [0.0]
        limiter = RateLimiter(per_minute=60, clock=lambda: now[0])
        monkeypatch.setattr(RateLimiter, "_MAX_CLIENTS", 2)
        for client in ("a", "b", "c"):
            limiter.acquire(client)
        now[0] = 120.0
        limiter.acquire("d")
        assert set(limiter._buckets) == {"d"}

    def test_api_returns_429_with_retry_after(self, make_app):
        with TestClient(make_app(rate_limit_per_minute=2)) as c:
            assert c.get(f"{API}/runs").status_code == 200
            assert c.get(f"{API}/engines").status_code == 200
            limited = c.get(f"{API}/runs")
            assert limited.status_code == 429
            assert limited.json()["code"] == "rate_limited"
            assert int(limited.headers["retry-after"]) >= 1
            # Health probes are never rate limited.
            assert c.get("/healthz").status_code == 200

    def test_limits_are_per_api_key(self, make_app):
        with TestClient(make_app(api_keys=["k1", "k2"], rate_limit_per_minute=1)) as c:
            assert c.get(f"{API}/runs", headers={"X-API-Key": "k1"}).status_code == 200
            assert c.get(f"{API}/runs", headers={"X-API-Key": "k1"}).status_code == 429
            assert c.get(f"{API}/runs", headers={"X-API-Key": "k2"}).status_code == 200

    def test_zero_disables(self, make_app):
        with TestClient(make_app(rate_limit_per_minute=0)) as c:
            assert all(c.get(f"{API}/runs").status_code == 200 for _ in range(5))
