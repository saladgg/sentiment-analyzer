"""
Domain exceptions and their HTTP mapping.

Services raise these; ``register_exception_handlers`` turns them, and
FastAPI's own HTTP and validation errors, into one JSON envelope:
``{"detail": ..., "code": str, "request_id": str}``. ``detail`` is a string,
except for request-validation errors (422 ``validation_error``), where it
keeps FastAPI's list of per-field errors.
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from sentiment_analyzer.core.logging import logger


class AppError(Exception):
    """Base class for errors that map to a specific HTTP status."""

    status_code = 400
    code = "bad_request"

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class InvalidInputError(AppError):
    status_code = 422
    code = "invalid_input"


class InputTooLargeError(AppError):
    status_code = 413
    code = "input_too_large"


class UnsupportedInputError(AppError):
    status_code = 415
    code = "unsupported_input"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class EngineUnavailableError(AppError):
    status_code = 503
    code = "engine_unavailable"


class EngineOutputError(AppError):
    """An engine returned output that does not satisfy the engine contract."""

    status_code = 502
    code = "engine_output_invalid"


_HTTP_CODES = {
    401: "unauthorized",
    403: "forbidden",
    429: "rate_limited",
    404: "not_found",
    405: "method_not_allowed",
}


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _envelope(
    request: Request,
    status_code: int,
    detail: Any,
    code: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    response_headers = dict(headers or {})
    if request_id:
        # Set here too: a 500 is rendered outside RequestContextMiddleware.
        response_headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "code": code, "request_id": request_id},
        headers=response_headers,
    )


async def _handle_app_error(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, AppError):  # pragma: no cover - registered for AppError only
        return await _handle_unexpected(request, exc)
    logger.info(
        "request_rejected",
        code=exc.code,
        status=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
    )
    return _envelope(request, exc.status_code, exc.detail, exc.code)


async def _handle_http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101 - registered for it only
    code = _HTTP_CODES.get(exc.status_code, "http_error")
    return _envelope(request, exc.status_code, exc.detail, code, getattr(exc, "headers", None))


async def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)  # noqa: S101 - registered for it only
    return _envelope(request, 422, jsonable_encoder(exc.errors()), "validation_error")


async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error", path=request.url.path)
    return _envelope(request, 500, "Internal server error", "internal_error")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(Exception, _handle_unexpected)
