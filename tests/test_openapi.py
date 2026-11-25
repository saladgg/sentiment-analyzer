"""
API contract snapshot.

Fails when the OpenAPI schema changes, so contract changes are reviewed
deliberately. After an intended change, regenerate with:

    UPDATE_SNAPSHOTS=1 uv run pytest tests/test_openapi.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SNAPSHOT = Path(__file__).parent / "snapshots" / "openapi.json"


def test_openapi_matches_snapshot(client):
    schema = client.get("/openapi.json").json()
    schema["info"].pop("version", None)  # changes on every release
    rendered = json.dumps(schema, indent=2, sort_keys=True) + "\n"

    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOT.write_text(rendered)

    assert rendered == SNAPSHOT.read_text(), (
        "OpenAPI schema changed. If intended, regenerate with "
        "UPDATE_SNAPSHOTS=1 uv run pytest tests/test_openapi.py"
    )
