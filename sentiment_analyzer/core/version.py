"""Application version, read from the installed package metadata."""

from functools import cache
from importlib.metadata import PackageNotFoundError, version


@cache
def app_version() -> str:
    try:
        return version("sentiment-analyzer")
    except PackageNotFoundError:  # running from a source checkout without install
        return "0.0.0+unknown"
