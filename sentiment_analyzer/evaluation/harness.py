"""
Evaluation harness for sentiment systems.

Provides:
- Precision measurement (against labeled data)
- Drift detection (distribution changes)
- Stability analysis (repeatability across runs)
"""

import statistics


class EvaluationHarness:
    """
    Lightweight evaluation framework.

    Designed to run offline or post-analysis.
    """

    def precision(self, predictions: list[str], labels: list[str]) -> float:
        """
        Compute simple classification accuracy.

        Assumes labels are aligned.
        """
        correct = sum(p == label for p, label in zip(predictions, labels, strict=True))
        return correct / len(labels) if labels else 0.0

    def drift(self, scores_run_a: list[float], scores_run_b: list[float]) -> float:
        """
        Detect sentiment drift using mean score difference.
        """
        if not scores_run_a or not scores_run_b:
            return 0.0

        return abs(statistics.mean(scores_run_a) - statistics.mean(scores_run_b))

    def stability(self, scores: list[float]) -> float:
        """
        Measure output stability using score variance.

        Lower variance = more stable system.
        """
        if len(scores) < 2:
            return 0.0

        return statistics.variance(scores)
