"""
API-key authentication and per-client rate limiting.

Clients send ``X-API-Key: <key>``.

- ``ASA_API_KEYS``: full access.
- ``ASA_API_READ_KEYS``: read-only access, i.e. ``GET``/``HEAD`` requests
  (listing and exporting runs, listing engines and documents).

When no keys are configured the API is open, which
``Settings.validate_for_startup`` only permits in ``dev``.

Rate limiting (``ASA_RATE_LIMIT_PER_MINUTE``) is a token bucket per client,
where a client is its API key (hashed) or, without auth, its IP address.
Buckets live in process memory, which matches the single-process
deployment model (see ``storage/runs.py``).
"""

import hashlib
import math
import secrets
import threading
import time
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from pydantic import SecretStr

from sentiment_analyzer.core.config import Settings

API_KEY_HEADER = "X-API-Key"
READ_ONLY_METHODS = frozenset({"GET", "HEAD"})

_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _matches(candidate: bytes, keys: list[SecretStr]) -> bool:
    # Constant-time comparison against every configured key.
    found = False
    for key in keys:
        found |= secrets.compare_digest(candidate, key.get_secret_value().encode())
    return found


def _unauthorized(detail: str = "Missing or invalid API key") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": API_KEY_HEADER},
    )


def client_identity(request: Request, api_key: str | None) -> str:
    """Stable, non-secret identifier for logs and rate limiting."""
    if api_key:
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:12]
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


def require_api_key(
    request: Request,
    api_key: Annotated[str | None, Depends(_api_key_header)],
) -> None:
    """Reject the request unless it carries a key allowed for its method."""
    settings = _settings(request)
    request.state.client_id = client_identity(request, api_key)
    if not settings.auth_enabled:
        return
    if not api_key:
        raise _unauthorized()

    candidate = api_key.encode()
    if _matches(candidate, settings.api_keys):
        return
    if _matches(candidate, settings.api_read_keys):
        if request.method in READ_ONLY_METHODS:
            return
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This API key is read-only",
        )
    raise _unauthorized()


class RateLimiter:
    """Thread-safe token buckets: ``per_minute`` capacity, refilled continuously."""

    _MAX_CLIENTS = 10_000

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def acquire(self, client: str) -> float:
        """Take one token. Returns 0 if allowed, else seconds until a token frees up."""
        now = self._clock()
        with self._lock:
            tokens, updated = self._buckets.get(client, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - updated) * self.rate)
            if tokens >= 1:
                self._buckets[client] = (tokens - 1, now)
                self._prune(now)
                return 0.0
            self._buckets[client] = (tokens, now)
            return (1 - tokens) / self.rate

    def _prune(self, now: float) -> None:
        if len(self._buckets) <= self._MAX_CLIENTS:
            return
        # Drop buckets that have been idle long enough to be full again.
        full_after = self.capacity / self.rate
        self._buckets = {k: v for k, v in self._buckets.items() if now - v[1] < full_after}


def enforce_rate_limit(request: Request) -> None:
    """Dependency: 429 with ``Retry-After`` when the client is over its limit."""
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    client = getattr(request.state, "client_id", None) or client_identity(request, None)
    wait = limiter.acquire(client)
    if wait > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({int(limiter.capacity)} requests/minute)",
            headers={"Retry-After": str(max(1, math.ceil(wait)))},
        )
