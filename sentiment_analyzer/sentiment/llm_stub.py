"""
LLM-backed sentiment engine using litellm.

This engine provides:
- High semantic understanding via any LLM provider
- Support for RAG context enrichment
- Structured JSON output for consistent parsing

Supports any provider that litellm supports (OpenAI, Anthropic,
Ollama, Azure, Groq, Mistral, etc.) by setting the model string
and corresponding API key in the environment.

Examples:
    ASA_LLM_MODEL=gpt-4o                    # OpenAI
    ASA_LLM_MODEL=claude-sonnet-4-20250514   # Anthropic
    ASA_LLM_MODEL=ollama/llama3              # Local Ollama
    ASA_LLM_MODEL=groq/llama-3.1-8b-instant # Groq
"""

import json

import litellm
from sentiment_analyzer.core.config import settings
from sentiment_analyzer.sentiment.base import SentimentEngine

SYSTEM_PROMPT = (
    "You are a sentiment analysis engine. "
    "Analyze the given text and return a JSON object with exactly these keys:\n"
    '- "label": one of "positive", "negative", or "neutral"\n'
    '- "score": a float between -1.0 (most negative) and 1.0 (most positive)\n'
    '- "drivers": an object mapping contributing factors to their float weight (0.0 to 1.0)\n'
    "Return ONLY the JSON object, no other text."
)


class LLMSentimentEngine(SentimentEngine):
    """
    LLM-backed sentiment engine.

    Uses litellm for provider-agnostic LLM access.
    Supports both text and optional RAG context.
    """

    def __init__(self):
        self.model = settings.llm_model
        if settings.llm_api_key:
            litellm.api_key = settings.llm_api_key

    def analyze(self, text: str, context: str | None = None) -> dict:
        user_message = f"TEXT:\n{text}"
        if context:
            user_message += f"\n\nCONTEXT:\n{context}"

        response = litellm.completion(
            model=self.model,
            max_tokens=256,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        raw = response.choices[0].message.content
        result = json.loads(raw)

        return {
            "label": result["label"],
            "score": float(result["score"]),
            "drivers": {k: float(v) for k, v in result["drivers"].items()},
        }
