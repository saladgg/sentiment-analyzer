# Sentiment Engines

SentimentAnalyzer ships three engines behind one interface. Clients pick one
per request with `sentiment_engine`; the default is
`ASA_DEFAULT_SENTIMENT_ENGINE` (`rule_based`). `GET /api/v1/engines` lists
them with their capabilities and status.

## The contract

All engines extend `SentimentEngine` from
[`sentiment/base.py`](../sentiment_analyzer/sentiment/base.py) and return a
`SentimentResult`:

```python
@dataclass(frozen=True)
class SentimentResult:
    label: Literal["positive", "neutral", "negative"]
    polarity: float       # signed sentiment, clamped to [-1, 1]; 0 = neutral
    confidence: float     # how sure the engine is of `label`, clamped to [0, 1]
    drivers: dict[str, float] = {}   # engine-specific explanatory signals


class SentimentEngine(ABC):
    name: ClassVar[str]           # registry key, e.g. "rule_based"
    display_name: ClassVar[str]   # for UIs
    description: ClassVar[str]    # for UIs
    uses_context: ClassVar[bool] = False   # reads RAG context?
    is_paid: ClassVar[bool] = False        # each call costs money?

    @abstractmethod
    def analyze(self, text: str, context: str | None = None) -> SentimentResult: ...

    def analyze_batch(self, texts, contexts=None) -> list[SentimentResult | Exception]: ...

    def config_snapshot(self) -> dict[str, Any]: ...
```

- `SentimentResult` validates the label and clamps `polarity` and
  `confidence`, so every engine's output is on the same scale.
- `analyze_batch` returns one item per text: a `SentimentResult`, or the
  exception raised for that text. One bad input never discards the rest of
  the batch. The default implementation loops over `analyze`; engines
  override it when they can batch more efficiently.
- `config_snapshot` is stored with every run for reproducibility.
- `uses_context`: RAG enrichment only runs for engines that set it. For
  others, retrieval is skipped entirely.
- `is_paid`: paid engines are limited to `ASA_MAX_LLM_RECORDS` records per run
  and per evaluation call, and per-field contributions are not computed for
  them (that would double the number of billed calls).

In API responses, `polarity` is reported as `score`, and `drivers` as
`drivers`. Per-field scores are reported separately in
`contributing_fields` (see [data-model.md](data-model.md#recordsentiment)).

## The registry

[`sentiment/registry.py`](../sentiment_analyzer/sentiment/registry.py) holds
an `EngineRegistry`, created once per process in `main.lifespan`:

- Engines are built **lazily** on first use and cached for the process
  lifetime, behind a lock. A `rule_based` request never imports torch.
- `get(name)` raises `EngineUnavailableError` (HTTP `503`) for unknown
  engines and for engines that fail to load (missing `ml` extra, missing
  `ASA_LLM_MODEL`, model download failure).
- `describe()` returns static metadata for every engine without loading any:
  `name`, `display_name`, `description`, `uses_context`, `is_paid`,
  `configured` (for `llm`: whether `ASA_LLM_MODEL` is set), `loaded`, and
  `default`.

## Available engines

| `sentiment_engine` | Class                        | File                                                                              | `uses_context` | `is_paid` |
| ------------------ | ---------------------------- | --------------------------------------------------------------------------------- | -------------- | --------- |
| `rule_based`       | `RuleBasedSentimentEngine`   | [sentiment/rule_based.py](../sentiment_analyzer/sentiment/rule_based.py)          | no             | no        |
| `hf_transformer`   | `HuggingFaceSentimentEngine` | [sentiment/hf_transformer.py](../sentiment_analyzer/sentiment/hf_transformer.py)  | no             | no        |
| `llm`              | `LLMSentimentEngine`         | [sentiment/llm.py](../sentiment_analyzer/sentiment/llm.py)                        | yes            | yes       |

### Rule-based

A deterministic lexicon engine. No model weights, no network. Use it as the
audit-safe baseline and as a reference when comparing other engines.

- **Tokenizer** — lower-cases the text and extracts word tokens with
  `[a-z]+(?:'[a-z]+)?`, so punctuation is ignored (`great!` matches `great`)
  and contractions stay whole (`don't`).
- **Lexicons** — small positive and negative word sets in the module.
- **Negation** — a negator directly before a sentiment word flips it
  (`not good` counts as negative). Negators are `not`, `no`, `never`,
  `nothing`, `hardly`, `without`, `cannot`, and any token ending in `n't`.

```
polarity   = (positive_hits - negative_hits) / (positive_hits + negative_hits)
confidence = |polarity|
```

With no hits, or hits that cancel out, the label is `neutral` with polarity
and confidence `0`.

`drivers`:

```python
{"positive_hits": float, "negative_hits": float, "negated_hits": float}
```

The `context` argument is ignored by design.

### HuggingFace transformer

Runs a transformers `text-classification` pipeline. It needs the `ml` extra
(`uv sync --extra ml`, or `pip install 'sentiment-analyzer[ml]'`); without it,
loading the engine fails with a `503` that says so. torch and transformers
are imported only when the engine is first built.

Configuration: `ASA_HF_SENTIMENT_MODEL` (default
`distilbert-base-uncased-finetuned-sst-2-english`), `ASA_HF_MODEL_REVISION`
(pin a commit for reproducible results), `ASA_HF_BATCH_SIZE`,
`ASA_HF_NEUTRAL_BAND`.

**Label mapping.** The pipeline is called with `top_k=None` to get the full
class distribution. Each model label is mapped to a polarity by
`label_polarity`:

| Model label                                   | Polarity                          |
| --------------------------------------------- | --------------------------------- |
| contains `pos` / `neg` / `neu`                | `1` / `-1` / `0`                  |
| `N star(s)` (1-5)                             | `(N - 3) / 2`                     |
| `LABEL_n` on a 2-class model                  | `LABEL_0 = -1`, `LABEL_1 = 1`     |
| `LABEL_n` on a 3-class model                  | `LABEL_0 = -1`, `LABEL_1 = 0`, `LABEL_2 = 1` |

A label that matches none of these raises an error for that record. This
covers binary, three-class and 1-5 star models.

**Scores.**

- `polarity` is the probability-weighted label polarity. For a binary model
  that is `2 * p(positive) - 1`, so uncertain predictions sit near `0`.
- `confidence` is the top-class probability.
- `label` follows the polarity of the top class.

**Neutral band.** Binary models never predict neutral. With
`ASA_HF_NEUTRAL_BAND=x`, a binary-model prediction whose `p(positive)` lies
within `0.5 ± x` is labelled `neutral`. It has no effect on models that have
a neutral class. The default `0.0` disables it.

**Batching and truncation.** `analyze_batch` sends the whole batch through
the pipeline with `batch_size=ASA_HF_BATCH_SIZE` and `truncation=True`, so
texts longer than the model's maximum sequence length are truncated rather
than failing. If the batch call fails, the engine retries text by text so
only the failing texts get an error.

`drivers`:

```python
{"model_confidence": float, "p_positive": float, "p_negative": float, ...}
```

There is one `p_<label>` entry per model label (lower-cased, spaces replaced
by `_`).

`config_snapshot` records the model, revision and neutral band. The engine
ignores `context`.

### LLM

Provider-agnostic engine backed by [litellm](https://docs.litellm.ai/).
Configure it with `ASA_LLM_MODEL` (required; there is no default) and, where
needed, `ASA_LLM_API_KEY`. Until a model is set the engine reports
`configured: false` and requests for it return `503`.

| Provider  | `ASA_LLM_MODEL` example             |
| --------- | ----------------------------------- |
| OpenAI    | `gpt-4o`                            |
| Anthropic | `anthropic/claude-sonnet-4-5`       |
| Groq      | `groq/llama-3.1-8b-instant`         |
| Ollama    | `ollama/llama3` (local, no API key) |
| Azure     | `azure/your-deployment-name`        |

Each call uses `temperature=0`, `max_tokens=ASA_LLM_MAX_TOKENS`,
`timeout=ASA_LLM_TIMEOUT` and `num_retries=ASA_LLM_NUM_RETRIES`. When litellm
reports that the model supports `response_format`, JSON mode is requested.

**Prompt-injection hardening.** Record text and RAG context are untrusted:

- The user message wraps the record in `<text>...</text>` and any context in
  `<context>...</context>`. Any `<text>`, `</text>`, `<context>` or
  `</context>` tag inside the input is stripped first, so input cannot close
  or open a block.
- The system prompt states that everything inside those tags is data, never
  instructions.
- The response is validated against a strict schema (below). An injected
  instruction can at most change one record's label; it cannot change the
  output format or affect other records.

**Output validation.** The model must return a JSON object with `label`
(`positive`/`neutral`/`negative`, case-insensitive), `score` (`-1..1`),
optional `confidence` (`0..1`) and optional `drivers` (string → number).
The parser accepts plain JSON, JSON in a markdown fence, or a JSON object
embedded in prose. `score` and `confidence` are clamped to at most `[-1, 1]`
(a negative `confidence` is rejected) and non-numeric drivers are dropped. Anything else (empty response, invalid JSON, invalid
label) raises `EngineOutputError`, which becomes that record's `error`. If
`confidence` is missing, `|score|` is used.

**Cache.** Results are cached in memory per process, keyed by
`(text, context)`, for up to 1024 entries (least recently used are evicted).
Repeated texts are not billed twice.

**Concurrency.** `analyze_batch` runs up to `ASA_LLM_CONCURRENCY` calls in
parallel. Failures stay per record.

**Limits.** As a paid engine, runs are limited to `ASA_MAX_LLM_RECORDS`
records (default 500), and per-field contributions are not computed.

**Environment.** Importing the module sets `LITELLM_MODE=PRODUCTION` (unless
already set) so litellm does not load `./.env` into `os.environ`.

`config_snapshot` records the model and `max_tokens`. This is the only engine
that uses RAG `context`.

## Choosing an engine

| When you want...                                     | Use              |
| ---------------------------------------------------- | ---------------- |
| Reproducibility, no network, fast                    | `rule_based`     |
| Better accuracy locally, no per-call cost            | `hf_transformer` |
| Nuance, sarcasm handling, domain context via RAG     | `llm`            |
| Measuring the effect of RAG                          | `llm` with `POST /api/v1/runs/ab` |
| Comparing two engines on the same input              | Two runs, then `GET /api/v1/runs/compare` |

Use `POST /api/v1/evaluation/engine` with labelled texts to measure an
engine's accuracy on your own data (see
[api-reference.md](api-reference.md#post-evaluationengine)).

## Adding a new engine

1. **Implement the contract.** Create a module under
   [`sentiment_analyzer/sentiment/`](../sentiment_analyzer/sentiment) and
   subclass `SentimentEngine`. Set `name`, `display_name`, `description`, and
   `uses_context` / `is_paid` if they apply. Implement `analyze` to return a
   `SentimentResult`. Override `analyze_batch` if the engine can batch, and
   `config_snapshot` to record anything that affects results.
2. **Register it** in
   [`sentiment/registry.py`](../sentiment_analyzer/sentiment/registry.py):
   add the class to the `ENGINE_CLASSES` tuple, and a factory to
   `EngineRegistry.__init__` that builds it from `Settings`:
   ```python
   self._factories = {
       ...,
       MyEngine.name: lambda: MyEngine(settings.my_engine_option),
   }
   ```
   If it needs configuration to be usable, extend `is_configured`.
3. **Allow the name** in the API and settings: add it to the `EngineName`
   enum in [`api/schemas.py`](../sentiment_analyzer/api/schemas.py) and to
   `EngineSetting` in [`core/config.py`](../sentiment_analyzer/core/config.py).
   Add any new settings as described in
   [configuration.md](configuration.md#adding-a-new-setting).
4. **Frontend.** The Analyze page reads engines from `GET /engines`, so the
   new engine appears automatically. Optionally add it to the fallback maps in
   [`frontend/theme.py`](../frontend/theme.py), used only when the API is
   unreachable.
5. **Test it** in
   [`tests/test_sentiment_engines.py`](../tests/test_sentiment_engines.py):
   label/polarity/confidence semantics, batch failure isolation, and the
   config snapshot. Stub out network or model calls, as `conftest.py` does
   for the HF pipeline and `litellm.completion`.
