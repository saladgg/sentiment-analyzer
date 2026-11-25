"""
Evaluation harness for sentiment systems.

Computes standard classification metrics of predicted labels against
ground-truth labels: accuracy, per-class precision/recall/F1, macro-F1
and the confusion matrix.
"""

from collections import Counter
from collections.abc import Sequence


class EvaluationHarness:
    """Lightweight, dependency-free classification metrics."""

    @staticmethod
    def accuracy(predictions: Sequence[str], labels: Sequence[str]) -> float:
        """Share of predictions equal to their label. Inputs must be aligned."""
        if len(predictions) != len(labels):
            raise ValueError("predictions and labels must have the same length")
        correct = sum(p == label for p, label in zip(predictions, labels, strict=True))
        return correct / len(labels) if labels else 0.0

    def report(self, predictions: Sequence[str], labels: Sequence[str]) -> dict:
        """Full classification report."""
        accuracy = self.accuracy(predictions, labels)
        classes = sorted(set(labels) | set(predictions))
        pairs = Counter(zip(labels, predictions, strict=True))

        per_class = {}
        for cls in classes:
            tp = pairs[(cls, cls)]
            predicted = sum(n for (_, p), n in pairs.items() if p == cls)
            actual = sum(n for (t, _), n in pairs.items() if t == cls)
            precision = tp / predicted if predicted else 0.0
            recall = tp / actual if actual else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[cls] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": actual,
            }

        supported = [m["f1"] for c, m in per_class.items() if m["support"] > 0]
        return {
            "total_samples": len(labels),
            "accuracy": round(accuracy, 4),
            "macro_f1": round(sum(supported) / len(supported), 4) if supported else 0.0,
            "per_class": per_class,
            "confusion_matrix": {
                "labels": classes,
                # rows = actual label, columns = predicted label
                "matrix": [[pairs[(t, p)] for p in classes] for t in classes],
            },
        }
