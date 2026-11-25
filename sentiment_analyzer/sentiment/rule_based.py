"""
Simple deterministic rule-based sentiment engine.

Used as:
- Baseline
- Debugging reference
- Deterministic audit-safe engine
"""

import re

from sentiment_analyzer.sentiment.base import Label, SentimentEngine, SentimentResult

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "happy",
    "love",
    "loved",
    "wonderful",
    "fantastic",
    "amazing",
    "best",
    "like",
    "liked",
    "perfect",
    "awesome",
    "nice",
    "pleased",
    "recommend",
    "helpful",
}
NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "hate",
    "hated",
    "poor",
    "worst",
    "awful",
    "horrible",
    "disappointing",
    "disappointed",
    "dislike",
    "ugly",
    "broken",
    "useless",
    "rude",
    "slow",
    "late",
}
NEGATORS = {"not", "no", "never", "nothing", "hardly", "without", "cannot"}

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens, ignoring punctuation (``great!`` -> ``great``)."""
    return _TOKEN_RE.findall(text.lower())


def _is_negator(token: str) -> bool:
    return token in NEGATORS or token.endswith("n't")


class RuleBasedSentimentEngine(SentimentEngine):
    """
    Deterministic lexicon-based engine with one-token negation.

    ``polarity = (positive_hits - negative_hits) / total_hits``.
    ``confidence = |polarity|``: how one-sided the lexicon hits are
    (``0`` when there are no hits or they cancel out).

    A negator directly before a sentiment word flips it
    (``not good`` counts as negative). Ignores RAG context by design.
    """

    name = "rule_based"
    display_name = "Rule-Based (Lexical)"
    description = "Fast, deterministic lexical matching. Best for auditable results."

    def analyze(self, text: str, context: str | None = None) -> SentimentResult:
        tokens = tokenize(text)
        pos_hits = neg_hits = negated = 0

        for i, token in enumerate(tokens):
            if token in POSITIVE_WORDS:
                polarity = 1
            elif token in NEGATIVE_WORDS:
                polarity = -1
            else:
                continue

            if i > 0 and _is_negator(tokens[i - 1]):
                polarity = -polarity
                negated += 1

            if polarity > 0:
                pos_hits += 1
            else:
                neg_hits += 1

        total_hits = pos_hits + neg_hits
        drivers = {
            "positive_hits": float(pos_hits),
            "negative_hits": float(neg_hits),
            "negated_hits": float(negated),
        }

        if total_hits == 0:
            return SentimentResult(label="neutral", polarity=0.0, confidence=0.0, drivers=drivers)

        score = (pos_hits - neg_hits) / total_hits
        label: Label
        if score > 0:
            label = "positive"
        elif score < 0:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(
            label=label,
            polarity=score,
            confidence=abs(score),
            drivers=drivers,
        )
