"""
Integration tests for the sentiment analysis API endpoints.
"""

from fastapi.testclient import TestClient
from sentiment_analyzer.main import app

client = TestClient(app)


def test_analyze_json():
    """Test the JSON analysis endpoint returns expected record structure."""
    response = client.post(
        "/api/analyze",
        json={
            "source_type": "json",
            "sentiment_engine": "rule_based",
            "data": [
                {"text": "good product"},
                {"text": "bad experience"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert "records" in payload
    assert len(payload["records"]) == 2

    record = payload["records"][0]
    assert "contributing_fields" in record
