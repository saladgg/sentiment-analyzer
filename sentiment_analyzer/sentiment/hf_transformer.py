"""
Transformer-based sentiment engine using HuggingFace.

This engine provides:
- Strong baseline accuracy
- Deterministic inference (pin ``ASA_HF_MODEL_REVISION`` for a fixed model)
- Calibrated polarity from the full class distribution

Works with binary (positive/negative), three-class
(negative/neutral/positive) and 1–5 star models.
"""

import re
from collections.abc import Sequence
from typing import Any

from sentiment_analyzer.sentiment.base import BatchItem, Label, SentimentEngine, SentimentResult

_STAR_RE = re.compile(r"([1-5])\s*star")
_GENERIC_RE = re.compile(r"label_(\d+)")


def _load_pipeline(model_name: str, revision: str | None) -> Any:
    """Build the transformers pipeline. Imported lazily: torch is heavy."""
    from transformers import pipeline

    return pipeline(task="text-classification", model=model_name, revision=revision)


def label_polarity(label: str, num_labels: int) -> float:
    """
    Map a model label to a polarity in ``[-1, 1]``.

    Handles named labels (``POSITIVE``, ``neg``, ``Neutral``), star
    ratings (``4 stars``) and generic ``LABEL_n`` labels for binary and
    three-class models (ordered negative → positive by convention).
    """
    lowered = label.lower()
    if "pos" in lowered:
        return 1.0
    if "neg" in lowered:
        return -1.0
    if "neu" in lowered:
        return 0.0
    if star := _STAR_RE.search(lowered):
        return (int(star.group(1)) - 3) / 2
    if (generic := _GENERIC_RE.fullmatch(lowered)) and num_labels in (2, 3):
        index = int(generic.group(1))
        if index < num_labels:
            return {2: [-1.0, 1.0], 3: [-1.0, 0.0, 1.0]}[num_labels][index]
    raise ValueError(f"Cannot map model label {label!r} to a sentiment polarity")


class HuggingFaceSentimentEngine(SentimentEngine):
    """
    Transformer-based sentiment engine.

    ``polarity`` is the probability-weighted label polarity (for a binary
    model, ``2 * p(positive) - 1``), so uncertain predictions sit near 0.
    ``confidence`` is the top-class probability.
    """

    name = "hf_transformer"
    display_name = "HuggingFace Transformer"
    description = "Local transformer model (default: DistilBERT SST-2). Balanced speed/accuracy."

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        batch_size: int = 16,
        neutral_band: float = 0.0,
    ):
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.neutral_band = neutral_band
        self.pipeline = _load_pipeline(model_name, revision)

    def _run(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        outputs = self.pipeline(texts, batch_size=self.batch_size, truncation=True, top_k=None)
        # A single input may come back un-nested.
        if outputs and isinstance(outputs[0], dict):
            outputs = [outputs]
        return outputs

    def _to_result(self, scores: list[dict[str, Any]]) -> SentimentResult:
        num_labels = len(scores)
        probs = {s["label"]: float(s["score"]) for s in scores}
        polarities = {label: label_polarity(label, num_labels) for label in probs}

        polarity = sum(p * polarities[label] for label, p in probs.items())
        top_label, confidence = max(probs.items(), key=lambda kv: kv[1])
        top_polarity = polarities[top_label]
        has_neutral_class = any(v == 0.0 for v in polarities.values())

        label: Label
        if top_polarity > 0:
            label = "positive"
        elif top_polarity < 0:
            label = "negative"
        else:
            label = "neutral"

        # Binary models never predict neutral; use the band around 0.5.
        if not has_neutral_class and abs(polarity) <= 2 * self.neutral_band:
            label = "neutral"

        drivers = {"model_confidence": confidence}
        drivers.update({f"p_{lbl.lower().replace(' ', '_')}": p for lbl, p in probs.items()})
        return SentimentResult(
            label=label, polarity=polarity, confidence=confidence, drivers=drivers
        )

    def analyze(self, text: str, context: str | None = None) -> SentimentResult:
        """Analyze one text. RAG context is ignored by this engine."""
        return self._to_result(self._run([text])[0])

    def analyze_batch(
        self,
        texts: Sequence[str],
        contexts: Sequence[str | None] | None = None,
    ) -> list[BatchItem]:
        """Run the whole batch through the pipeline; fall back per text on failure."""
        if not texts:
            return []
        try:
            outputs = self._run(list(texts))
        except Exception:  # noqa: BLE001 - isolate the failing text below
            return super().analyze_batch(texts, contexts)

        results: list[BatchItem] = []
        for scores in outputs:
            try:
                results.append(self._to_result(scores))
            except Exception as exc:  # noqa: BLE001 - reported per record
                results.append(exc)
        return results

    def config_snapshot(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "model": self.model_name,
            "revision": self.revision,
            "neutral_band": self.neutral_band,
        }
