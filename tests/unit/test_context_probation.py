import pytest

from evoshift.config import EvolutionConfig
from evoshift.evolution import ContextLocalProbation, ContextLocalProvisionalLane
from evoshift.schemas import MemoryItem


def _config(**updates: object) -> EvolutionConfig:
    return EvolutionConfig(
        paired_replay=True,
        future_audit_enabled=True,
        dynamic_feedback_trust_enabled=True,
        context_local_probation_fast_path_enabled=True,
        **updates,
    )


def _memory() -> MemoryItem:
    return MemoryItem(
        memory_id="candidate",
        status="probation",
        trigger="refund context",
        scope="refund",
        directive="apply the verified current rule",
    )


def test_context_probation_is_exact_context_and_bounded() -> None:
    probation = ContextLocalProbation(
        _config(
            context_local_probation_fast_path_max_age=3,
            context_local_probation_fast_path_max_uses=2,
        )
    )
    lease = probation.register(
        _memory(),
        source="portal",
        context="Refund:Case",
        signal="approve",
        context_observations=2,
        episode_index=10,
    )
    assert lease is not None
    assert (
        probation.match(
            source="portal",
            context="other",
            signal="approve",
            context_observations=3,
            episode_index=11,
        )
        is None
    )
    assert (
        probation.match(
            source="portal",
            context="refund:case",
            signal="deny",
            context_observations=3,
            episode_index=11,
        )
        is None
    )
    assert (
        probation.match(
            source="portal",
            context="refund:case",
            signal="approve",
            context_observations=3,
            episode_index=10,
        )
        is None
    )
    assert (
        probation.match(
            source="portal",
            context="refund:case",
            signal="approve",
            context_observations=2,
            episode_index=11,
        )
        is None
    )
    assert (
        probation.match(
            source="portal",
            context="refund:case",
            signal="approve",
            context_observations=3,
            episode_index=11,
        )
        == lease
    )
    probation.consume(lease)
    second = probation.match(
        source="portal",
        context="refund:case",
        signal="approve",
        context_observations=4,
        episode_index=12,
    )
    assert second is not None
    probation.consume(second)
    assert (
        probation.match(
            source="portal",
            context="refund:case",
            signal="approve",
            context_observations=5,
            episode_index=13,
        )
        is None
    )
    assert probation.snapshot()["interventions"] == 2


def test_context_probation_expires_and_discard_is_memory_scoped() -> None:
    probation = ContextLocalProbation(_config(context_local_probation_fast_path_max_age=1))
    lease = probation.register(
        _memory(),
        source="portal",
        context="case",
        signal="approve",
        context_observations=1,
        episode_index=10,
    )
    assert lease is not None
    assert probation.expire(11) == 0
    assert probation.expire(12) == 1
    assert probation.snapshot()["pending"] == 0
    lease = probation.register(
        _memory(),
        source="portal",
        context="case",
        signal="approve",
        context_observations=1,
        episode_index=20,
    )
    assert lease is not None
    probation.discard("candidate")
    assert probation.pending() == ()


def test_context_probation_rejects_stale_lease_consumption() -> None:
    probation = ContextLocalProbation(_config())
    lease = probation.register(
        _memory(),
        source="portal",
        context="case",
        signal="approve",
        context_observations=1,
        episode_index=10,
    )
    assert lease is not None
    probation.discard("candidate")
    with pytest.raises(ValueError, match="no longer pending"):
        probation.consume(lease)


def test_context_probation_rejects_empty_signal_instead_of_creating_wildcard() -> None:
    probation = ContextLocalProbation(_config())
    assert (
        probation.register(
            _memory(),
            source="portal",
            context="case",
            signal="  ",
            context_observations=2,
            episode_index=10,
        )
        is None
    )
    assert probation.snapshot()["registrations"] == 0


def test_provisional_lane_requires_mature_source_and_confirmed_change() -> None:
    lane = ContextLocalProvisionalLane(
        _config(
            context_local_provisional_lane_enabled=True,
            context_local_provisional_lane_min_source_trust=0.8,
            context_local_provisional_lane_min_context_observations=3,
        )
    )
    memory = _memory()
    common = {
        "source": "portal",
        "context": "case",
        "signal": "deny",
        "source_posterior_mean": 0.9,
        "context_observations": 3,
        "pending_observations": 0,
        "reason": "dynamic_confirmed_change",
        "episode_index": 10,
    }
    assert (
        lane.register_from_assessment(
            memory, **{**common, "source_posterior_mean": 0.79}
        )
        is None
    )
    assert lane.register_from_assessment(
        memory,
        **{
            **common,
            "pending_observations": 1,
            "reason": "dynamic_pending_change",
        },
    ) is None
    assert (
        lane.register_from_assessment(memory, **{**common, "context_observations": 2}) is None
    )
    lease = lane.register_from_assessment(memory, **common)
    assert lease is not None
    assert lane.match(
        source="portal",
        context="case",
        signal="deny",
        context_observations=4,
        episode_index=11,
    ) == lease


def test_provisional_lane_is_bounded_and_discardable() -> None:
    lane = ContextLocalProvisionalLane(
        _config(
            context_local_provisional_lane_max_age=1,
            context_local_provisional_lane_max_uses=1,
        )
    )
    lease = lane.register_from_assessment(
        _memory(),
        source="portal",
        context="case",
        signal="deny",
        source_posterior_mean=0.9,
        context_observations=2,
        pending_observations=0,
        reason="dynamic_confirmed_change",
        episode_index=10,
    )
    assert lease is not None
    lane.consume(lease)
    assert lane.match(
        source="portal",
        context="case",
        signal="deny",
        context_observations=3,
        episode_index=11,
    ) is None
    assert lane.snapshot()["exhaustions"] == 1
