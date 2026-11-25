"""
HTTP middleware: request IDs, access logging, and request-size limits.

Implemented as pure ASGI middleware so streaming bodies are counted
without being buffered twice.
"""

import json
import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sentiment_analyzer.core.logging import logger

REQUEST_ID_HEADER = "x-request-id"
API_PREFIXES = ("/api/v1", "/api")


def route_template(scope: Scope) -> str:
    """
    Matched route as a template (``/api/v1/runs/{run_id}``), for metric labels.

    Templates keep label cardinality bounded; unmatched paths collapse to
    ``"unmatched"``. FastAPI records the route without its router prefix,
    so the API prefix is re-added from the request path.
    """
    template = getattr(scope.get("route"), "path", None)
    if not template:
        return "unmatched"
    path = scope.get("path", "")
    for prefix in API_PREFIXES:
        if path.startswith(prefix + "/") and not template.startswith(prefix + "/"):
            return prefix + template
    return template


class RequestContextMiddleware:
    """
    Assign every request an ID (honouring a sane incoming ``X-Request-ID``),
    bind it to the logging context, return it in the response, and log one
    access line per request.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(REQUEST_ID_HEADER.encode(), b"").decode("latin-1")
        request_id = incoming if 0 < len(incoming) <= 128 and incoming.isprintable() else ""
        request_id = request_id or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = message.setdefault("headers", [])
                if not any(k.lower() == REQUEST_ID_HEADER.encode() for k, _ in headers):
                    headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            path = scope.get("path", "")
            elapsed = time.perf_counter() - started
            template = route_template(scope)
            metrics = getattr(getattr(scope.get("app"), "state", None), "metrics", None)
            if metrics is not None and path != "/metrics":
                method = str(scope.get("method"))
                metrics.http_requests.labels(method, template, str(status_code)).inc()
                metrics.http_latency.labels(method, template).observe(elapsed)
            if path not in ("/healthz", "/readyz", "/metrics"):
                logger.info(
                    "http_request",
                    method=scope.get("method"),
                    path=path,
                    status=status_code,
                    duration_ms=round(elapsed * 1000, 1),
                    client=(scope.get("state") or {}).get("client_id"),
                )
            structlog.contextvars.clear_contextvars()


class BodySizeLimitMiddleware:
    """
    Reject request bodies larger than ``max_bytes`` with 413.

    Checks ``Content-Length`` up front and also counts streamed chunks, so
    chunked uploads without a length header are bounded too.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_bytes: int,
        overrides: dict[str, int] | None = None,
    ):
        """``overrides`` maps a path fragment (e.g. ``"/jobs/"``) to a larger limit."""
        self.app = app
        self.max_bytes = max_bytes
        self.overrides = overrides or {}

    def _limit_for(self, path: str) -> int:
        for fragment, limit in self.overrides.items():
            if fragment in path:
                return limit
        return self.max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_bytes = self._limit_for(scope.get("path", ""))
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        request_id = str((scope.get("state") or {}).get("request_id") or "")
        if declared is not None and declared.isdigit() and int(declared) > max_bytes:
            await self._reject(send, request_id, max_bytes)
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await self._reject(send, request_id, max_bytes)

    async def _reject(self, send: Send, request_id: str, max_bytes: int) -> None:
        body = json.dumps(
            {
                "detail": f"Request body exceeds the {max_bytes} byte limit",
                "code": "input_too_large",
                "request_id": request_id or None,
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _BodyTooLarge(Exception):
    pass
