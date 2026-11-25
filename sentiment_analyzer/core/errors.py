"""
Domain exceptions and their HTTP mapping.

Services raise these; ``register_exception_handlers`` turns them into
JSON responses of the shape ``{"detail": str, "code": str}`` so routes
never need their own try/except blocks.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def _handle_app_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    logger.info(
        "request_rejected",
        code=exc.code,
        status=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": exc.code, "request_id": _request_id(request)},
    )


async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error", path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "code": "internal_error",
            "request_id": _request_id(request),
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(Exception, _handle_unexpected)
