"""
Prometheus metrics.

Each app instance owns its own ``CollectorRegistry`` (so tests can build
many apps), exposed at ``GET /metrics`` in the Prometheus text format.

Label values are always bounded: HTTP paths use the route *template*
(``/api/v1/runs/{run_id}``), never the raw URL.
"""

from dataclasses import dataclass, field

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)


@dataclass
class Metrics:
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self) -> None:
        r = self.registry
        self.http_requests = Counter(
            "asa_http_requests_total",
            "HTTP requests by method, route template and status code.",
            ["method", "route", "status"],
            registry=r,
        )
        self.http_latency = Histogram(
            "asa_http_request_duration_seconds",
            "HTTP request latency by method and route template.",
            ["method", "route"],
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.runs = Counter(
            "asa_analysis_runs_total",
            "Completed analysis runs by engine and status.",
            ["engine", "status"],
            registry=r,
        )
        self.records = Counter(
            "asa_records_analyzed_total",
            "Records analyzed by engine and outcome (ok/failed).",
            ["engine", "outcome"],
            registry=r,
        )
        self.inference_latency = Histogram(
            "asa_engine_batch_duration_seconds",
            "Time spent in an engine's analyze_batch call.",
            ["engine"],
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.rag_latency = Histogram(
            "asa_rag_retrieval_duration_seconds",
            "Time spent retrieving RAG context for a batch.",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.llm_tokens = Counter(
            "asa_llm_tokens_total",
            "LLM tokens consumed, by kind (prompt/completion).",
            ["kind"],
            registry=r,
        )
        self.llm_cost = Counter(
            "asa_llm_cost_usd_total",
            "Estimated LLM spend in USD (litellm pricing; unknown models count 0).",
            registry=r,
        )
        self.llm_calls = Counter(
            "asa_llm_calls_total",
            "LLM engine requests by source (api/cache).",
            ["source"],
            registry=r,
        )
        self.jobs = Gauge(
            "asa_jobs",
            "Background analysis jobs by status.",
            ["status"],
            registry=r,
        )
