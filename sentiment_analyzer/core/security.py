"""
API-key authentication.

Clients send ``X-API-Key: <key>``. Keys are configured with
``ASA_API_KEYS`` (comma-separated). When no keys are configured the API
is open, which ``Settings.validate_for_startup`` only permits in ``dev``.
"""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from sentiment_analyzer.core.config import Settings

API_KEY_HEADER = "X-API-Key"

_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def require_api_key(
    request: Request,
    api_key: Annotated[str | None, Depends(_api_key_header)],
) -> None:
    """Reject the request unless it carries a configured API key."""
    settings = _settings(request)
    if not settings.auth_enabled:
        return

    if api_key:
        candidate = api_key.encode()
        for configured in settings.api_keys:
            # Constant-time comparison of each configured key.
            if secrets.compare_digest(candidate, configured.get_secret_value().encode()):
                return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key",
        headers={"WWW-Authenticate": API_KEY_HEADER},
    )
