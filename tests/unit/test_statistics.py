from __future__ import annotations

import pytest

from evoshift.config import EvolutionConfig
from evoshift.evaluation import (
    PromotionGate,
    paired_bootstrap_ci,
    paired_cluster_bootstrap_ci,
)


def _gate(**overrides: object) -> PromotionGate:
    config = EvolutionConfig(
        min_validation_examples=4,
        bootstrap_samples=500,
        confidence_level=0.90,
        min_mean_gain=0.03,
        min_ci_lower_bound=-0.01,
        max_regression_rate=0.15,
        max_protected_slice_regression=0.05,
        max_cost_increase_ratio=0.30,
        **overrides,
    )
    return PromotionGate(config)


def test_paired_bootstrap_ci_is_paired_reproducible_and_contains_mean() -> None:
    deltas = [-0.1, 0.1, 0.2, 0.3, 0.4]
    first = paired_bootstrap_ci(deltas, 1000, 0.95, 7)
    second = paired_bootstrap_ci(deltas, 1000, 0.95, 7)
    assert first == second
    assert first[0] < sum(deltas) / len(deltas) < first[1]
    assert paired_bootstrap_ci([0.2] * 5, 100, 0.95, 1) == (0.2, 0.2)


def test_paired_bootstrap_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        paired_bootstrap_ci([])
    with pytest.raises(ValueError, match="samples"):
        paired_bootstrap_ci([0.1], samples=0)
    with pytest.raises(ValueError, match="confidence"):
        paired_bootstrap_ci([0.1], confidence=1.0)


def test_paired_cluster_bootstrap_is_reproducible_and_seed_balanced() -> None:
    deltas = {
        11: [1.0, 1.0],
        22: [-1.0, -1.0, -1.0, -1.0],
    }

    first = paired_cluster_bootstrap_ci(deltas, samples=1000, confidence=0.95, seed=7)
    second = paired_cluster_bootstrap_ci(deltas, samples=1000, confidence=0.95, seed=7)

    assert first == second
    assert first[0] == pytest.approx(0.0)
    assert first[1] <= first[0] <= first[2]


def test_paired_cluster_bootstrap_rejects_bad_clusters() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        paired_cluster_bootstrap_ci({})
    with pytest.raises(ValueError, match="has no paired deltas"):
        paired_cluster_bootstrap_ci({1: []})
    with pytest.raises(ValueError, match="finite"):
        paired_cluster_bootstrap_ci({1: [float("nan")]})
    with pytest.raises(ValueError, match="positive"):
        paired_cluster_bootstrap_ci({1: [0.0]}, samples=0)


def test_promotion_gate_promotes_verified_gain() -> None:
    decision = _gate().decide(
        "memory-1",
        "memory",
        control_scores=[0.4, 0.5, 0.6, 0.7, 0.5, 0.4, 0.6, 0.5],
        candidate_scores=[0.5, 0.6, 0.7, 0.8, 0.6, 0.5, 0.7, 0.6],
        control_costs=[1.0] * 8,
        candidate_costs=[1.1] * 8,
        protected_mask=[True, True, False, False, False, False, False, False],
    )
    assert decision.promote is True
    assert all(decision.gate_checks.values())
    assert decision.result.mean_delta == pytest.approx(0.1)
    assert decision.result.cost_delta_ratio == pytest.approx(0.1)


def test_promotion_gate_rejects_worst_protected_slice_even_with_global_gain() -> None:
    control = [0.5] * 10
    candidate = [0.4] + [0.6] * 9
    decision = _gate().decide(
        "memory-2",
        "memory",
        control,
        candidate,
        protected_mask=[True] + [False] * 9,
    )
    assert decision.result.mean_delta == pytest.approx(0.08)
    assert decision.result.regression_rate == 0.1
    assert decision.result.protected_slice_regression == pytest.approx(0.1)
    assert decision.gate_checks["protected_slice"] is False
    assert decision.promote is False


def test_promotion_gate_rejects_cost_increase_and_too_few_examples() -> None:
    costly = _gate().decide(
        "policy-1",
        "policy",
        [0.2, 0.2, 0.2, 0.2],
        [0.4, 0.4, 0.4, 0.4],
        control_costs=[1.0] * 4,
        candidate_costs=[1.5] * 4,
    )
    assert costly.gate_checks["cost"] is False
    assert costly.promote is False

    zero_cost_control = _gate().decide(
        "policy-2",
        "policy",
        [0.2, 0.2, 0.2, 0.2],
        [0.4, 0.4, 0.4, 0.4],
        control_costs=[0.0] * 4,
        candidate_costs=[0.01] * 4,
    )
    assert zero_cost_control.gate_checks["cost"] is False

    undersized = _gate().decide(
        "memory-3",
        "memory",
        [0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5],
    )
    assert undersized.gate_checks["minimum_examples"] is False
    assert undersized.promote is False


def test_promotion_gate_validates_pair_shapes() -> None:
    gate = _gate()
    with pytest.raises(ValueError, match="same length"):
        gate.decide("x", "memory", [0.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="provided together"):
        gate.decide("x", "memory", [0.0], [1.0], control_costs=[1.0])
    with pytest.raises(ValueError, match="protected_mask"):
        gate.decide("x", "memory", [0.0], [1.0], protected_mask=[True, False])
