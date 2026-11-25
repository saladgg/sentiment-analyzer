"""
Structured logging configuration.

Structured logs are critical for:
- Debugging multi-step AI pipelines
- Auditing decisions
- Production observability

Every log line, including uvicorn's and third-party libraries' stdlib
logs, is rendered as one JSON object. Request-scoped values bound with
``structlog.contextvars`` (e.g. ``request_id``) are added automatically.
"""

import logging
import sys

import structlog

_SHARED_PROCESSORS: list = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog and route stdlib logging through the same JSON renderer."""
    log_level = logging.getLevelName(level.upper())

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.processors.StackInfoRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_SHARED_PROCESSORS,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level)

    # uvicorn installs its own handlers; make it propagate to ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = []
        uv_logger.propagate = True
    # Our middleware logs requests with a request_id; drop the duplicate.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


logger = structlog.get_logger("sentiment_analyzer")
