"""Tests for background analysis jobs (``/api/v1/jobs``)."""

from __future__ import annotations

import io
import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sentiment_analyzer.sentiment.rule_based import RuleBasedSentimentEngine
from sentiment_analyzer.services.jobs import RESTART_REASON
from sentiment_analyzer.storage.runs import RunStore

API = "/api/v1"
TERMINAL = {"completed", "failed", "cancelled"}


def wait_for(client: TestClient, job_id: str, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"{API}/jobs/{job_id}").json()
        if job["status"] in TERMINAL:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {job}")


@pytest.fixture
def gate(monkeypatch):
    """Make rule_based block until ``gate.set()`` so jobs stay queued/running."""
    event = threading.Event()
    original = RuleBasedSentimentEngine.analyze_batch

    def _blocking(self, texts, contexts=None):
        assert event.wait(10), "test gate never opened"
        return original(self, texts, contexts)

    monkeypatch.setattr(RuleBasedSentimentEngine, "analyze_batch", _blocking)
    yield event
    event.set()


class TestSubmitAndPoll:
    def test_json_job_completes_with_a_run(self, client):
        response = client.post(
            f"{API}/jobs/analyze",
            json={"source_type": "text", "data": "great\nterrible\nfine"},
        )
        assert response.status_code == 202
        job = response.json()
        assert job["status"] in {"queued", "running", "completed"}
        assert job["records_total"] == 3

        done = wait_for(client, job["job_id"])
        assert done["status"] == "completed"
        assert done["progress"] == 1.0
        assert done["records_done"] == 3
        assert done["finished_at"] is not None

        run = client.get(f"{API}/runs/{done['run_id']}").json()
        assert run["records_total"] == 3

    def test_csv_job(self, client):
        csv = "comment\n" + "".join(f"great product number {i}\n" for i in range(600))
        response = client.post(
            f"{API}/jobs/analyze/csv",
            files={"file": ("big.csv", io.BytesIO(csv.encode()), "text/csv")},
        )
        assert response.status_code == 202
        done = wait_for(client, response.json()["job_id"])
        assert done["status"] == "completed"
        assert done["records_done"] == 600

    def test_excel_job_rejects_wrong_extension(self, client):
        response = client.post(
            f"{API}/jobs/analyze/excel",
            files={"file": ("x.csv", io.BytesIO(b"a\nb"), "text/csv")},
        )
        assert response.status_code == 415

    def test_list_and_filter(self, client):
        job_id = client.post(
            f"{API}/jobs/analyze", json={"source_type": "text", "data": "good"}
        ).json()["job_id"]
        wait_for(client, job_id)
        page = client.get(f"{API}/jobs", params={"status": "completed"}).json()
        assert page["total"] == 1
        assert page["items"][0]["job_id"] == job_id
        assert client.get(f"{API}/jobs", params={"status": "failed"}).json()["total"] == 0

    def test_unknown_job(self, client):
        assert client.get(f"{API}/jobs/nope").status_code == 404


class TestValidationHappensBeforeQueueing:
    def test_bad_input_is_rejected_synchronously(self, client):
        response = client.post(
            f"{API}/jobs/analyze",
            json={"source_type": "json", "text_fields": ["missing"], "data": [{"a": "hello"}]},
        )
        assert response.status_code == 422
        assert client.get(f"{API}/jobs").json()["total"] == 0

    def test_job_limits_are_larger_than_sync_limits(self, make_app):
        payload = {"source_type": "text", "data": "\n".join(["good one"] * 5)}
        with TestClient(make_app(max_records=3, max_job_records=10)) as c:
            sync = c.post(f"{API}/analyze", json=payload)
            assert sync.status_code == 413
            assert "background job" in sync.json()["detail"]
            assert c.post(f"{API}/jobs/analyze", json=payload).status_code == 202

    def test_job_record_limit(self, make_app):
        with TestClient(make_app(max_job_records=2)) as c:
            response = c.post(
                f"{API}/jobs/analyze", json={"source_type": "text", "data": "a b c\nd e f\ng h"}
            )
        assert response.status_code == 413

    def test_paid_engine_job_limit(self, make_app, fake_llm):
        with TestClient(make_app(max_llm_records=1, max_llm_job_records=2)) as c:
            ok = c.post(
                f"{API}/jobs/analyze",
                json={"source_type": "text", "sentiment_engine": "llm", "data": "a\nb"},
            )
            assert ok.status_code == 202
            wait_for(c, ok.json()["job_id"])
            too_many = c.post(
                f"{API}/jobs/analyze",
                json={"source_type": "text", "sentiment_engine": "llm", "data": "a\nb\nc"},
            )
            assert too_many.status_code == 413

    def test_use_rag_without_rag(self, client, fake_llm):
        response = client.post(
            f"{API}/jobs/analyze",
            json={"source_type": "text", "sentiment_engine": "llm", "use_rag": True, "data": "x"},
        )
        assert response.status_code == 422

    def test_job_uploads_get_the_larger_body_limit(self, make_app):
        body = "comment\n" + "".join(f"great product {i}\n" for i in range(20))  # ~350 B
        with TestClient(make_app(max_upload_bytes=100, max_job_upload_bytes=10_000)) as c:
            files = {"file": ("x.csv", io.BytesIO(body.encode()), "text/csv")}
            assert c.post(f"{API}/analyze/csv", files=files).status_code == 413
            files = {"file": ("x.csv", io.BytesIO(body.encode()), "text/csv")}
            assert c.post(f"{API}/jobs/analyze/csv", files=files).status_code == 202


class TestQueueAndCancel:
    def test_queue_limit_and_cancel(self, make_app, gate):
        with TestClient(make_app(job_workers=1, max_queued_jobs=1)) as c:
            payload = {"source_type": "text", "data": "good"}
            running = c.post(f"{API}/jobs/analyze", json=payload).json()
            # Wait until the worker has picked the first job up.
            deadline = time.monotonic() + 5
            while c.get(f"{API}/jobs/{running['job_id']}").json()["status"] != "running":
                assert time.monotonic() < deadline
                time.sleep(0.01)

            queued = c.post(f"{API}/jobs/analyze", json=payload)
            assert queued.status_code == 202
            full = c.post(f"{API}/jobs/analyze", json=payload)
            assert full.status_code == 429
            assert full.json()["code"] == "job_queue_full"

            # Running jobs can't be cancelled; queued ones can.
            assert c.delete(f"{API}/jobs/{running['job_id']}").status_code == 409
            cancelled = c.delete(f"{API}/jobs/{queued.json()['job_id']}")
            assert cancelled.status_code == 200
            assert cancelled.json()["status"] == "cancelled"
            assert cancelled.json()["finished_at"] is not None

            gate.set()
            assert wait_for(c, running["job_id"])["status"] == "completed"
            # Finished jobs can't be cancelled either.
            assert c.delete(f"{API}/jobs/{running['job_id']}").status_code == 409
            metrics = c.get("/metrics").text
            assert 'asa_jobs{status="queued"} 0.0' in metrics
            assert 'asa_jobs{status="running"} 0.0' in metrics


class TestFailures:
    def test_run_time_app_error_fails_the_job(self, client, monkeypatch):
        from sentiment_analyzer.core.errors import EngineUnavailableError
        from sentiment_analyzer.services.analysis_service import AnalysisService

        def _boom(self, *args, **kwargs):
            raise EngineUnavailableError("engine went away")

        monkeypatch.setattr(AnalysisService, "run_prepared", _boom)
        job_id = client.post(
            f"{API}/jobs/analyze", json={"source_type": "text", "data": "good"}
        ).json()["job_id"]
        job = wait_for(client, job_id)
        assert job["status"] == "failed"
        assert job["error"] == "engine went away"
        assert job["run_id"] is None

    def test_unexpected_error_is_not_leaked(self, client, monkeypatch):
        from sentiment_analyzer.services.analysis_service import AnalysisService

        def _crash(self, *args, **kwargs):
            raise RuntimeError("secret internals")

        monkeypatch.setattr(AnalysisService, "run_prepared", _crash)
        job_id = client.post(
            f"{API}/jobs/analyze", json={"source_type": "text", "data": "good"}
        ).json()["job_id"]
        job = wait_for(client, job_id)
        assert job["status"] == "failed"
        assert "secret" not in job["error"]


class TestRecovery:
    def test_unfinished_jobs_fail_after_restart(self, make_app, tmp_db_path):
        store = RunStore(tmp_db_path)
        store.create_job(
            "left-queued", engine="rule_based", source_type="json", records_total=5, client_id=None
        )
        store.create_job(
            "left-running", engine="rule_based", source_type="json", records_total=5, client_id=None
        )
        store.update_job("left-running", status="running")
        store.close()

        with TestClient(make_app()) as c:
            for job_id in ("left-queued", "left-running"):
                job = c.get(f"{API}/jobs/{job_id}").json()
                assert job["status"] == "failed"
                assert job["error"] == RESTART_REASON

    def test_old_finished_jobs_are_pruned(self, tmp_db_path):
        store = RunStore(tmp_db_path)
        store.create_job("old", engine="x", source_type="json", records_total=1, client_id=None)
        store.update_job("old", status="completed")
        store.conn.execute(
            "UPDATE jobs SET finished_at = finished_at - INTERVAL 40 DAY WHERE job_id = 'old'"
        )
        store.create_job("new", engine="x", source_type="json", records_total=1, client_id=None)
        store.update_job("new", status="completed")
        assert store.prune_jobs(30) == 1
        assert store.get_job("old") is None
        assert store.get_job("new") is not None

    def test_update_job_rejects_unknown_fields(self, tmp_db_path):
        store = RunStore(tmp_db_path)
        store.create_job("j", engine="x", source_type="json", records_total=1, client_id=None)
        with pytest.raises(ValueError):
            store.update_job("j", engine="other")
