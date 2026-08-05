from evoshift.config import EvolutionConfig
from evoshift.evolution import RetirementProbation
from evoshift.schemas import MemoryItem, MemoryStatus


def _config(**updates: object) -> EvolutionConfig:
    return EvolutionConfig(
        active_audit_enabled=True,
        active_audit_retirement_probation_enabled=True,
    ).model_copy(update=updates)


def _retired(memory_id: str = "retired") -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        version=2,
        status=MemoryStatus.RETIRED,
        trigger="refund rule",
        scope="customer_support/refund_policy",
        directive="Use the retired refund rule.",
    )


def test_retirement_probation_requires_later_exact_context_confirmation() -> None:
    probation = RetirementProbation(_config())
    pending = probation.register(
        source="portal",
        context="refund:any:days_8_14",
        memory=_retired(),
        restored_predecessors=(),
        episode_index=10,
        mechanism="active_causal",
        phase_index=2,
        oracle_stale=True,
        early_retirement=False,
    )

    assert pending is not None
    assert probation.pending_memory_keys() == {("retired", 2)}
    assert (
        probation.match(
            source="portal",
            context="refund:any:days_8_14",
            episode_index=10,
            retired_memory_versions=[("retired", 2)],
        )
        is None
    )
    matched = probation.match(
        source="portal",
        context="refund:any:days_8_14",
        episode_index=11,
        retired_memory_versions=[("retired", 2)],
    )
    assert matched == pending
    probation.defer(matched)
    assert probation.pending_memory_keys() == {("retired", 2)}
    matched = probation.match(
        source="portal",
        context="refund:any:days_8_14",
        episode_index=12,
        retired_memory_versions=[("retired", 2)],
    )
    assert matched == pending
    assert probation.qualifies(0.75)
    probation.resolve(matched, confirmed=True)
    assert probation.snapshot()["confirmations"] == 1
    assert probation.snapshot()["deferrals"] == 1
    assert probation.pending_memory_keys() == set()


def test_retirement_probation_invalidates_changed_version_and_expires() -> None:
    probation = RetirementProbation(_config(active_audit_retirement_probation_max_age=2))
    pending = probation.register(
        source="portal",
        context="context",
        memory=_retired(),
        restored_predecessors=(),
        episode_index=10,
        mechanism="posterior_utility",
        phase_index=1,
        oracle_stale=False,
        early_retirement=False,
    )
    assert pending is not None
    assert (
        probation.match(
            source="portal",
            context="context",
            episode_index=11,
            retired_memory_versions=[("retired", 3)],
        )
        is None
    )
    assert probation.snapshot()["invalidations"] == 1

    expiring = RetirementProbation(_config(active_audit_retirement_probation_max_age=2))
    expiring_pending = expiring.register(
        source="portal",
        context="context",
        memory=_retired("expiring"),
        restored_predecessors=(),
        episode_index=10,
        mechanism="active_causal",
        phase_index=1,
        oracle_stale=True,
        early_retirement=True,
    )
    assert expiring_pending is not None
    assert expiring.expire(12) == ()
    assert expiring.expire(13) == (expiring_pending,)
    assert expiring.snapshot()["pending"] == 0
