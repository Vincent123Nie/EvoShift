import pytest

from evoshift.config import EvolutionConfig
from evoshift.evolution import ContextLocalProbation
from evoshift.schemas import MemoryItem


def _config(**updates: object) -> EvolutionConfig:
    return EvolutionConfig(
        paired_replay=True,
        future_audit_enabled=True,
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
        episode_index=10,
    )
    assert lease is not None
    assert (
        probation.match(source="portal", context="other", signal="approve", episode_index=11)
        is None
    )
    assert (
        probation.match(source="portal", context="refund:case", signal="deny", episode_index=11)
        is None
    )
    assert (
        probation.match(source="portal", context="refund:case", signal="approve", episode_index=10)
        is None
    )
    assert (
        probation.match(source="portal", context="refund:case", signal="approve", episode_index=11)
        == lease
    )
    probation.consume(lease)
    second = probation.match(
        source="portal", context="refund:case", signal="approve", episode_index=12
    )
    assert second is not None
    probation.consume(second)
    assert (
        probation.match(source="portal", context="refund:case", signal="approve", episode_index=13)
        is None
    )
    assert probation.snapshot()["interventions"] == 2


def test_context_probation_expires_and_discard_is_memory_scoped() -> None:
    probation = ContextLocalProbation(_config(context_local_probation_fast_path_max_age=1))
    lease = probation.register(
        _memory(), source="portal", context="case", signal="approve", episode_index=10
    )
    assert lease is not None
    assert probation.expire(11) == 0
    assert probation.expire(12) == 1
    assert probation.snapshot()["pending"] == 0
    lease = probation.register(
        _memory(), source="portal", context="case", signal="approve", episode_index=20
    )
    assert lease is not None
    probation.discard("candidate")
    assert probation.pending() == ()


def test_context_probation_rejects_stale_lease_consumption() -> None:
    probation = ContextLocalProbation(_config())
    lease = probation.register(
        _memory(), source="portal", context="case", signal="approve", episode_index=10
    )
    assert lease is not None
    probation.discard("candidate")
    with pytest.raises(ValueError, match="no longer pending"):
        probation.consume(lease)
