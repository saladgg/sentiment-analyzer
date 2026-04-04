"""
Structured logging configuration.

Structured logs are critical for:
- Debugging multi-step AI pipelines
- Auditing decisions
- Production observability

This module standardizes logging across the application.
"""

import logging

import structlog


def configure_logging():
    """
    Configure structlog for JSON-style structured logging.
    """
    logging.basicConfig(level=logging.INFO)

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )


logger = structlog.get_logger()
