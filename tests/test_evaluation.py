"""
Tests for the evaluation harness metrics.
"""

import pytest
from sentiment_analyzer.evaluation.harness import EvaluationHarness


@pytest.fixture
def harness() -> EvaluationHarness:
    return EvaluationHarness()


class TestPrecision:
    def test_returns_value_in_unit_interval(self, harness):
        accuracy = harness.precision(
            ["positive", "negative", "positive"],
            ["positive", "negative", "negative"],
        )
        assert 0.0 <= accuracy <= 1.0

    def test_perfect_match(self, harness):
        assert harness.precision(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_no_match(self, harness):
        assert harness.precision(["a", "a"], ["b", "b"]) == 0.0

    def test_empty_inputs_return_zero(self, harness):
        assert harness.precision([], []) == 0.0

    def test_misaligned_lengths_raise(self, harness):
        with pytest.raises(ValueError):
            harness.precision(["a", "b"], ["a"])


class TestDrift:
    def test_returns_zero_for_identical_distributions(self, harness):
        assert harness.drift([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == 0.0

    def test_returns_absolute_mean_delta(self, harness):
        assert harness.drift([1.0, 1.0], [0.0, 0.0]) == pytest.approx(1.0)

    def test_handles_negative_drift_as_absolute(self, harness):
        assert harness.drift([0.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0)

    @pytest.mark.parametrize("a,b", [([], [0.1]), ([0.1], []), ([], [])])
    def test_empty_runs_return_zero(self, harness, a, b):
        assert harness.drift(a, b) == 0.0


class TestStability:
    def test_returns_zero_for_single_value(self, harness):
        assert harness.stability([0.5]) == 0.0

    def test_returns_zero_for_empty_input(self, harness):
        assert harness.stability([]) == 0.0

    def test_constant_scores_have_zero_variance(self, harness):
        assert harness.stability([0.5, 0.5, 0.5]) == 0.0

    def test_variance_increases_with_spread(self, harness):
        tight = harness.stability([0.4, 0.5, 0.6])
        wide = harness.stability([0.1, 0.5, 0.9])
        assert wide > tight
