"""
Shared constants for consistent theming across pages.
"""

# Sentiment label -> colour mapping (used in charts & badges)
SENTIMENT_COLORS: dict[str, str] = {
    "positive": "#10b981",  # emerald-500
    "neutral": "#6366f1",  # indigo-500
    "negative": "#ef4444",  # red-500
}

ENGINE_LABELS: dict[str, str] = {
    "rule_based": "Rule-Based (Lexical)",
    "hf_transformer": "HuggingFace Transformer",
    "llm": "LLM (Claude / OpenAI / Ollama)",
}

ENGINE_DESCRIPTIONS: dict[str, str] = {
    "rule_based": "Fast, deterministic lexical matching. Best for auditable results.",
    "hf_transformer": "DistilBERT fine-tuned on SST-2. Good balance of speed and accuracy.",
    "llm": "Provider-agnostic LLM via litellm. Most nuanced but slowest.",
}
