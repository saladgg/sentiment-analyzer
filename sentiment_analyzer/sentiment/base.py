"""
Sentiment engine interface.

Every engine returns a :class:`SentimentResult` with the same semantics,
so results are comparable across engines:

- ``label``: one of ``positive`` / ``neutral`` / ``negative``
- ``polarity``: signed sentiment in ``[-1, 1]`` (``0`` = neutral)
- ``confidence``: how sure the engine is about ``label``, in ``[0, 1]``
- ``drivers``: engine-specific numeric signals that explain the result
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

Label = Literal["positive", "neutral", "negative"]
LABELS: tuple[Label, ...] = ("positive", "neutral", "negative")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class SentimentResult:
    """Normalized output of a sentiment engine for one text."""

    label: Label
    polarity: float
    confidence: float
    drivers: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"Invalid sentiment label: {self.label!r}")
        object.__setattr__(self, "polarity", _clamp(float(self.polarity), -1.0, 1.0))
        object.__setattr__(self, "confidence", _clamp(float(self.confidence), 0.0, 1.0))
        object.__setattr__(self, "drivers", {k: float(v) for k, v in self.drivers.items()})


# A batch item is either a result or the exception raised for that text,
# so one bad input never discards the rest of the batch.
BatchItem = SentimentResult | Exception


class SentimentEngine(ABC):
    """Base class for sentiment engines."""

    #: Registry key, e.g. ``"rule_based"``.
    name: ClassVar[str]
    #: Human-readable name for UIs.
    display_name: ClassVar[str]
    #: One-line description for UIs.
    description: ClassVar[str]
    #: Whether the engine reads RAG context. Enrichment is skipped otherwise.
    uses_context: ClassVar[bool] = False
    #: Whether each call costs money (paid API). Paid engines get stricter limits.
    is_paid: ClassVar[bool] = False

    @abstractmethod
    def analyze(self, text: str, context: str | None = None) -> SentimentResult:
        """Analyze one text, optionally with RAG context."""
        raise NotImplementedError

    def analyze_batch(
        self,
        texts: Sequence[str],
        contexts: Sequence[str | None] | None = None,
    ) -> list[BatchItem]:
        """
        Analyze many texts.

        The default loops over :meth:`analyze`; engines override this
        when they can batch more efficiently.
        """
        ctxs = list(contexts) if contexts is not None else [None] * len(texts)
        results: list[BatchItem] = []
        for text, ctx in zip(texts, ctxs, strict=True):
            try:
                results.append(self.analyze(text, context=ctx))
            except Exception as exc:  # noqa: BLE001 - reported per record
                results.append(exc)
        return results

    def config_snapshot(self) -> dict[str, Any]:
        """Engine configuration recorded with each run for reproducibility."""
        return {"engine": self.name}
