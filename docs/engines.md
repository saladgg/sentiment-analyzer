# Sentiment Engines

SentimentAnalyzer ships three pluggable engines, all of which conform to a
single interface. Engines are selected per-request via the
`sentiment_engine` field. The default is configured by
`ASA_DEFAULT_SENTIMENT_ENGINE`.

## The contract

All engines extend `SentimentEngine` from
[`sentiment/base.py`](../sentiment_analyzer/sentiment/base.py):

```python
class SentimentEngine(ABC):
    @abstractmethod
    def analyze(self, text: str, context: str | None = None) -> dict:
        """
        Returns:
            {
                "label":   "positive" | "neutral" | "negative",
                "score":   float in [-1.0, 1.0],
                "drivers": dict[str, float]
            }
        """
```

`drivers` is engine-specific — it is the explainability surface that flows
into `RecordSentiment.contributing_fields` in the API response.

## Available engines

| `sentiment_engine` value | Class                       | File                                                                 |
| ------------------------ | --------------------------- | -------------------------------------------------------------------- |
| `rule_based`             | `RuleBasedSentimentEngine`  | [sentiment/rule_based.py](../sentiment_analyzer/sentiment/rule_based.py) |
| `hf_transformer`         | `HuggingFaceSentimentEngine`| [sentiment/hf_transformer.py](../sentiment_analyzer/sentiment/hf_transformer.py) |
| `llm`                    | `LLMSentimentEngine`        | [sentiment/llm_stub.py](../sentiment_analyzer/sentiment/llm_stub.py) |

All three are instantiated eagerly inside
[`AnalysisService.__init__`](../sentiment_analyzer/services/analysis_service.py).

### Rule-based

A deterministic lexical engine using two small word lists (positive and
negative). Best used as:

- The audit-safe baseline (no model weights, no network).
- A debugging reference when comparing against transformer / LLM outputs.

Score formula:

```
score = (pos_hits - neg_hits) / (pos_hits + neg_hits)   if hits > 0
        0.0                                              otherwise
```

`drivers` returned:

```python
{
    "lexical_rules": abs(score),
    "positive_hits": int,
    "negative_hits": int,
}
```

It ignores the `context` argument by design — RAG should never affect
deterministic baselines.

### HuggingFace transformer

Loads a `text-classification` pipeline at construction time. The default model
(`distilbert-base-uncased-finetuned-sst-2-english`) emits two labels:
`POSITIVE` / `NEGATIVE`, so the `score` is signed accordingly:

```python
score = +confidence  if label == "positive"
score = -confidence  if label == "negative"
```

`drivers` returned:

```python
{ "model_confidence": float }
```

The HF engine **also ignores `context`** today. If you want RAG-aware
transformer output, you would need to prepend context to the input text or
fine-tune a model that accepts it.

To swap the model, change `ASA_HF_SENTIMENT_MODEL`. Anything that the
`text-classification` pipeline accepts will work; the engine assumes
two-label `POSITIVE`/`NEGATIVE` outputs.

### LLM

Provider-agnostic engine backed by [litellm](https://docs.litellm.ai/). It
formats a system prompt that asks the model to return JSON with the engine's
expected keys, then parses the response.

The LLM engine is the only one that **uses** the optional RAG `context`:

```python
user_message = f"TEXT:\n{text}"
if context:
    user_message += f"\n\nCONTEXT:\n{context}"
```

Configure with `ASA_LLM_API_KEY` and `ASA_LLM_MODEL`. The model id format is
litellm's — examples:

| Provider  | `ASA_LLM_MODEL`                     |
| --------- | ----------------------------------- |
| Anthropic | `claude-sonnet-4-20250514`          |
| OpenAI    | `gpt-4o`                            |
| Groq      | `groq/llama-3.1-8b-instant`         |
| Ollama    | `ollama/llama3` (local, no API key) |
| Azure     | `azure/your-deployment-name`        |

Failure modes to be aware of:

- The engine calls `json.loads` on the raw model output. Models that wrap JSON
  in prose will throw — pick a model that respects the system prompt's
  "return ONLY the JSON object" instruction, or wrap it for additional
  parsing tolerance.
- Latency is meaningfully higher than the other engines. The frontend
  HTTP client uses a 120-second timeout
  ([api_client.py](../frontend/api_client.py)).

## Choosing an engine

| When you want…                                         | Use              |
| ------------------------------------------------------ | ---------------- |
| Reproducibility, no network, fast                      | `rule_based`     |
| Strong baseline accuracy, no provider dependencies     | `hf_transformer` |
| Nuance, sarcasm handling, RAG-aware reasoning          | `llm`            |
| Comparing a baseline run to a RAG-enriched run         | `rule_based` then `llm`, then `GET /api/runs/compare` |

## Adding a new engine

1. **Implement the contract.** Create a new module under
   [`sentiment_analyzer/sentiment/`](../sentiment_analyzer/sentiment) and
   subclass `SentimentEngine`. Return the `{label, score, drivers}` dict.
2. **Register it.** Add an entry in
   [`AnalysisService.__init__`](../sentiment_analyzer/services/analysis_service.py):
   ```python
   self.sentiment_engines = {
       "rule_based": RuleBasedSentimentEngine(),
       "hf_transformer": HuggingFaceSentimentEngine(settings.hf_sentiment_model),
       "llm": LLMSentimentEngine(),
       "my_engine": MyEngine(),  # ← here
   }
   ```
3. **Surface it in the frontend.** Add labels and descriptions to
   [`frontend/theme.py`](../frontend/theme.py) so it appears in the Analyze
   page selector.
4. **Test it.** Add a case in
   [`tests/test_sentiment_engines.py`](../tests/test_sentiment_engines.py)
   that asserts the contract — same input always returns the right shape.
