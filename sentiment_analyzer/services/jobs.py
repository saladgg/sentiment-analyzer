"""
Background analysis jobs.

Large inputs are validated synchronously (so bad requests still get an
immediate 4xx), then queued and run by a small in-process worker pool.
Clients poll ``GET /api/v1/jobs/{job_id}`` for status and progress; a
completed job points at its ``run_id``.

Design constraints (single-process deployment, see ``storage/runs.py``):

- Job metadata is persisted in DuckDB; the *input* is held in memory only.
  A restart therefore cannot resume queued or running jobs: they are
  marked ``failed`` ("interrupted by restart") at the next startup.
- ``ASA_JOB_WORKERS`` jobs run concurrently (default 1) and share the
  process with the API, so a CPU-heavy job slows requests down.
"""

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any

from sentiment_analyzer.analytics.time_aggregation import Granularity
from sentiment_analyzer.core.config import Settings
from sentiment_analyzer.core.errors import (
    AppError,
    ConflictError,
    InvalidInputError,
    NotFoundError,
)
from sentiment_analyzer.core.logging import logger
from sentiment_analyzer.core.metrics import Metrics
from sentiment_analyzer.services.analysis_service import AnalysisService, PreparedInput
from sentiment_analyzer.storage.runs import RunStore

RESTART_REASON = "Interrupted by a server restart; resubmit the job"
# Persist progress at most this often (seconds) to limit write load.
_PROGRESS_INTERVAL = 1.0


class JobQueueFullError(AppError):
    status_code = 429
    code = "job_queue_full"


class JobManager:
    """Queues prepared inputs and runs them on a bounded thread pool."""

    def __init__(
        self,
        settings: Settings,
        service: AnalysisService,
        store: RunStore,
        metrics: Metrics | None = None,
    ):
        self.settings = settings
        self.service = service
        self.store = store
        self.metrics = metrics
        self._pool = ThreadPoolExecutor(
            max_workers=settings.job_workers, thread_name_prefix="asa-job"
        )
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()
        self._counts = {"queued": 0, "running": 0}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def recover(self) -> None:
        """Startup: fail jobs a previous process left unfinished; prune old ones."""
        failed = self.store.fail_unfinished_jobs(RESTART_REASON)
        pruned = self.store.prune_jobs(self.settings.job_retention_days)
        if failed or pruned:
            logger.info("jobs_recovered", interrupted=failed, pruned=pruned)
        self._publish_counts()

    def shutdown(self) -> bool:
        """
        Stop accepting work: cancel queued jobs and wait up to
        ``ASA_JOB_SHUTDOWN_TIMEOUT`` seconds for running ones.

        Returns True when no job is still running. Jobs cut off here stay
        ``queued``/``running`` in the table and are failed by ``recover`` at
        the next startup.
        """
        with self._lock:
            pending = list(self._futures.values())
        self._pool.shutdown(wait=False, cancel_futures=True)
        _, still_running = wait(pending, timeout=self.settings.job_shutdown_timeout)
        if still_running:
            logger.warning("jobs_still_running_at_shutdown", count=len(still_running))
        return not still_running

    # ------------------------------------------------------------------
    # Submission & control
    # ------------------------------------------------------------------
    def submit(
        self,
        prepared: PreparedInput,
        *,
        use_rag: bool | None,
        granularity: Granularity,
        client_id: str | None,
    ) -> dict[str, Any]:
        if use_rag and self.service.enricher is None:
            # Same check run_prepared does, but before the job is queued.
            raise InvalidInputError("RAG was requested but is disabled (ASA_RAG_ENABLED=false)")

        with self._lock:
            if self._counts["queued"] >= self.settings.max_queued_jobs:
                raise JobQueueFullError(
                    f"{self._counts['queued']} jobs are already queued "
                    "(ASA_MAX_QUEUED_JOBS); try again later"
                )
            job_id = str(uuid.uuid4())
            job = self.store.create_job(
                job_id,
                engine=prepared.engine_name,
                source_type=prepared.source_type,
                records_total=len(prepared.raw_records),
                client_id=client_id,
            )
            self._counts["queued"] += 1
            self._futures[job_id] = self._pool.submit(
                self._execute, job_id, prepared, use_rag, granularity
            )
        self._publish_counts()
        logger.info(
            "job_queued",
            job_id=job_id,
            engine=prepared.engine_name,
            records=len(prepared.raw_records),
        )
        return job

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued job. Running and finished jobs can't be cancelled."""
        job = self.get(job_id)
        with self._lock:
            future = self._futures.get(job_id)
            if job["status"] != "queued" or future is None or not future.cancel():
                raise ConflictError(f"Job {job_id!r} is {job['status']} and cannot be cancelled")
            self._futures.pop(job_id, None)
            self._counts["queued"] -= 1
        self.store.update_job(job_id, status="cancelled")
        self._publish_counts()
        logger.info("job_cancelled", job_id=job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if job is None:
            raise NotFoundError(f"Job {job_id!r} not found")
        return job

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _execute(
        self,
        job_id: str,
        prepared: PreparedInput,
        use_rag: bool | None,
        granularity: Granularity,
    ) -> None:
        with self._lock:
            self._counts["queued"] -= 1
            self._counts["running"] += 1
        self._publish_counts()
        self.store.update_job(job_id, status="running")
        last_write = 0.0

        def progress(done: int, total: int) -> None:
            nonlocal last_write
            now = time.monotonic()
            if done == total or now - last_write >= _PROGRESS_INTERVAL:
                self.store.update_job(job_id, records_done=done, records_total=total)
                last_write = now

        try:
            result = self.service.run_prepared(
                prepared, use_rag=use_rag, granularity=granularity, progress=progress
            )
        except AppError as exc:
            self.store.update_job(job_id, status="failed", error=exc.detail)
            logger.warning("job_failed", job_id=job_id, code=exc.code, detail=exc.detail)
        except Exception:  # noqa: BLE001 - any crash must fail the job, not the worker
            self.store.update_job(
                job_id, status="failed", error="Internal error while running the job"
            )
            logger.exception("job_crashed", job_id=job_id)
        else:
            self.store.update_job(job_id, status="completed", run_id=result["run_id"])
            logger.info("job_completed", job_id=job_id, run_id=result["run_id"])
        finally:
            with self._lock:
                self._counts["running"] -= 1
                self._futures.pop(job_id, None)
            self._publish_counts()

    def _publish_counts(self) -> None:
        if self.metrics:
            for status, count in self._counts.items():
                self.metrics.jobs.labels(status).set(count)
