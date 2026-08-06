import pytest

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


def _active_snapshot(memory_id: str = "retired") -> MemoryItem:
    return _retired(memory_id).model_copy(
        update={
            "status": MemoryStatus.ACTIVE,
            "beta": 3.0,
            "use_count": 2,
        }
    )


def test_retirement_probation_requires_later_exact_context_confirmation() -> None:
    probation = RetirementProbation(_config())
    snapshot = _active_snapshot()
    pending = probation.register(
        source="portal",
        context="refund:any:days_8_14",
        memory=_retired(),
        active_snapshot=snapshot,
        restored_predecessors=(),
        episode_index=10,
        mechanism="active_causal",
        phase_index=2,
        oracle_stale=True,
        early_retirement=False,
    )

    assert pending is not None
    assert pending.active_snapshot == snapshot
    assert pending.active_snapshot.status == MemoryStatus.ACTIVE
    assert probation.pending_memory_keys() == {("retired", 2)}
    assert (
        probation.match(
            source="portal",
            context="refund:any:days_8_14",
            episode_index=10,
            available_memory_versions=[("retired", 2)],
        )
        is None
    )
    matched = probation.match(
        source="portal",
        context="refund:any:days_8_14",
        episode_index=11,
        available_memory_versions=[("retired", 2)],
    )
    assert matched == pending
    probation.defer(matched)
    assert probation.pending_memory_keys() == {("retired", 2)}
    matched = probation.match(
        source="portal",
        context="refund:any:days_8_14",
        episode_index=12,
        available_memory_versions=[("retired", 2)],
    )
    assert matched == pending
    assert probation.qualifies(0.75)
    probation.resolve(matched, confirmed=True)
    assert probation.snapshot()["confirmations"] == 1
    assert probation.snapshot()["deferrals"] == 1
    assert probation.pending_memory_keys() == set()


def test_context_scoped_probation_suppresses_only_the_matching_observable_key() -> None:
    probation = RetirementProbation(
        _config(active_audit_retirement_probation_context_scoped=True)
    )
    active = _active_snapshot()
    pending = probation.register(
        source="portal",
        context="refund:any:days_8_14",
        memory=active,
        active_snapshot=active,
        restored_predecessors=(),
        episode_index=10,
        mechanism="active_causal",
        phase_index=2,
        oracle_stale=False,
        early_retirement=False,
    )

    assert pending is not None
    assert probation.suppressed_memory_versions(
        source="portal",
        context="refund:any:days_8_14",
    ) == {("retired", 2)}
    assert (
        probation.suppressed_memory_versions(
            source="portal",
            context="refund:premium:days_15_30",
        )
        == set()
    )
    assert (
        probation.match(
            source="portal",
            context="refund:premium:days_15_30",
            episode_index=11,
            available_memory_versions=[("retired", 2)],
        )
        is None
    )
    assert probation.match(
        source="portal",
        context="refund:any:days_8_14",
        episode_index=11,
        available_memory_versions=[("retired", 2)],
    ) == pending
    assert probation.snapshot()["context_scoped"] is True


def test_context_scoped_probation_quarantine_releases_on_change_point() -> None:
    probation = RetirementProbation(
        _config(active_audit_retirement_probation_context_scoped=True)
    )
    active = _active_snapshot()
    pending = probation.register(
        source="portal",
        context="refund:any:days_8_14",
        memory=active,
        active_snapshot=active,
        restored_predecessors=(),
        episode_index=10,
        mechanism="active_causal",
        phase_index=2,
        oracle_stale=False,
        early_retirement=False,
    )
    assert pending is not None
    matched = probation.match(
        source="portal",
        context="refund:any:days_8_14",
        episode_index=11,
        available_memory_versions=[("retired", 2)],
    )
    assert matched == pending
    probation.quarantine(matched)
    assert probation.suppressed_memory_versions(
        source="portal",
        context="refund:any:days_8_14",
    ) == {("retired", 2)}
    released = probation.release_context(
        source="portal",
        context="refund:any:days_8_14",
    )
    assert released == (pending,)
    assert probation.suppressed_memory_versions(
        source="portal",
        context="refund:any:days_8_14",
    ) == set()
    assert probation.snapshot()["quarantine_releases"] == 1


def test_retirement_probation_invalidates_changed_version_and_expires() -> None:
    probation = RetirementProbation(_config(active_audit_retirement_probation_max_age=2))
    pending = probation.register(
        source="portal",
        context="context",
        memory=_retired(),
        active_snapshot=_active_snapshot(),
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
            available_memory_versions=[("retired", 3)],
        )
        is None
    )
    assert probation.snapshot()["invalidations"] == 1

    expiring = RetirementProbation(_config(active_audit_retirement_probation_max_age=2))
    expiring_pending = expiring.register(
        source="portal",
        context="context",
        memory=_retired("expiring"),
        active_snapshot=_active_snapshot("expiring"),
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


def test_same_context_matches_distinct_retirement_transactions_fifo() -> None:
    probation = RetirementProbation(_config())
    first = probation.register(
        source="portal",
        context="refund:any:days_8_14",
        memory=_retired("first"),
        active_snapshot=_active_snapshot("first"),
        restored_predecessors=(),
        episode_index=10,
        mechanism="posterior_utility",
        phase_index=1,
        oracle_stale=False,
        early_retirement=False,
    )
    second = probation.register(
        source="portal",
        context="refund:any:days_8_14",
        memory=_retired("second"),
        active_snapshot=_active_snapshot("second"),
        restored_predecessors=(),
        episode_index=11,
        mechanism="active_causal",
        phase_index=1,
        oracle_stale=False,
        early_retirement=False,
    )

    assert first is not None
    assert second is not None
    assert probation.pending_memory_keys() == {("first", 2), ("second", 2)}
    matched_first = probation.match(
        source="portal",
        context="refund:any:days_8_14",
        episode_index=12,
        available_memory_versions=[("first", 2), ("second", 2)],
    )
    assert matched_first == first
    probation.resolve(matched_first, confirmed=True)

    matched_second = probation.match(
        source="portal",
        context="refund:any:days_8_14",
        episode_index=13,
        available_memory_versions=[("first", 2), ("second", 2)],
    )
    assert matched_second == second


@pytest.mark.parametrize(
    ("delta", "terminal_outcome", "counter"),
    [
        (0.80, "confirm", "confirmations"),
        (-0.80, "veto", "cancellations"),
    ],
)
def test_decisive_retirement_evidence_requires_temporally_separated_repetition(
    delta: float,
    terminal_outcome: str,
    counter: str,
) -> None:
    probation = RetirementProbation(
        _config(
            active_audit_retirement_probation_min_confirmations=2,
            active_audit_retirement_probation_min_evidence_span=2,
        )
    )
    pending = probation.register(
        source="portal",
        context="refund:any:days_8_14",
        memory=_retired(),
        active_snapshot=_active_snapshot(),
        restored_predecessors=(),
        episode_index=10,
        mechanism="posterior_utility",
        phase_index=1,
        oracle_stale=False,
        early_retirement=False,
    )
    assert pending is not None

    first_match = probation.match(
        source="portal",
        context="refund:any:days_8_14",
        episode_index=11,
        available_memory_versions=[("retired", 2)],
    )
    assert first_match is not None
    first = probation.observe(
        first_match,
        episode_index=11,
        trust=1.0,
        old_memory_applied=True,
        delta=delta,
    )
    assert first.outcome == "defer"
    assert first.evidence_recorded

    adjacent_match = probation.match(
        source="portal",
        context="refund:any:days_8_14",
        episode_index=12,
        available_memory_versions=[("retired", 2)],
    )
    assert adjacent_match is not None
    adjacent = probation.observe(
        adjacent_match,
        episode_index=12,
        trust=1.0,
        old_memory_applied=True,
        delta=delta,
    )
    assert adjacent.outcome == "defer"
    assert not adjacent.evidence_recorded
    assert adjacent.pending.confirmation_indices + adjacent.pending.veto_indices == (11,)
    assert probation.snapshot()["evidence_observations"] == 1
    assert probation.snapshot()["temporal_deferrals"] == 1

    separated_match = probation.match(
        source="portal",
        context="refund:any:days_8_14",
        episode_index=13,
        available_memory_versions=[("retired", 2)],
    )
    assert separated_match is not None
    separated = probation.observe(
        separated_match,
        episode_index=13,
        trust=1.0,
        old_memory_applied=True,
        delta=delta,
    )
    assert separated.outcome == terminal_outcome
    assert separated.evidence_recorded
    assert separated.pending.confirmation_indices + separated.pending.veto_indices == (11, 13)
    assert probation.snapshot()[counter] == 1
    assert probation.snapshot()["evidence_observations"] == 2
    assert probation.pending_memory_keys() == set()
