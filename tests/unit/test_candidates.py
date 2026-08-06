from evoshift.evolution import CandidateEvidencePool, candidate_signature
from evoshift.schemas import MemoryItem, MemoryStatus


def _candidate(episode_id: str = "e1") -> MemoryItem:
    return MemoryItem(
        memory_id="",
        trigger="Refund after day seven",
        scope="customer_support/refund_policy",
        directive="Approve eligible refunds through day fourteen.",
        provenance_episode_ids=[episode_id],
        source_domains=["customer_support/refund_policy"],
        confidence=0.9,
    )


def _different_candidate(episode_id: str = "e-other") -> MemoryItem:
    return MemoryItem(
        memory_id="",
        trigger="Premium refund after day fourteen",
        scope="customer_support/refund_policy",
        directive="Approve premium refunds through day thirty.",
        provenance_episode_ids=[episode_id],
        source_domains=["customer_support/refund_policy"],
        confidence=0.9,
    )


def test_candidate_pool_aggregates_evidence_and_enforces_cooldown() -> None:
    pool = CandidateEvidencePool(
        min_observations=2,
        min_trusted_observations=1,
        min_new_observations=1,
        cooldown_episodes=4,
    )

    first = pool.observe(_candidate("e1"), 10)
    assert pool.readiness(first, 10) == (False, "insufficient_observations")

    second = pool.observe(_candidate("e2"), 12)
    assert second is first
    assert second.observation_count == 2
    assert second.candidate.provenance_episode_ids == ["e1", "e2"]
    assert pool.readiness(second, 12) == (True, "ready")

    pool.mark_validated(second, 12)
    pool.observe(_candidate("e3"), 13)
    assert pool.readiness(second, 13) == (False, "candidate_cooldown")
    assert pool.readiness(second, 16) == (True, "ready")


def test_candidate_pool_suppresses_an_already_active_memory() -> None:
    pool = CandidateEvidencePool(
        min_observations=1,
        min_trusted_observations=1,
        min_new_observations=1,
        cooldown_episodes=0,
    )
    active = _candidate().model_copy(update={"memory_id": "memory", "status": MemoryStatus.ACTIVE})
    pool.seed_accepted([active])

    evidence = pool.observe(_candidate("e2"), 20)

    assert candidate_signature(active) == evidence.signature
    assert pool.readiness(evidence, 20) == (False, "duplicate_of_active_memory")


def test_candidate_pool_blocks_probation_and_reopens_after_future_rejection() -> None:
    pool = CandidateEvidencePool(
        min_observations=1,
        min_trusted_observations=1,
        min_new_observations=1,
        cooldown_episodes=0,
    )
    evidence = pool.observe(_candidate(), 3)
    pool.mark_validated(evidence, 3)
    pool.mark_probation(evidence)

    assert pool.readiness(evidence, 4) == (False, "candidate_in_probation")

    pool.mark_rejected(evidence)
    evidence = pool.observe(_candidate("e2"), 5)
    assert pool.readiness(evidence, 5) == (True, "ready")


def test_candidate_pool_reopens_a_causally_retired_memory_for_recurrence() -> None:
    pool = CandidateEvidencePool(
        min_observations=1,
        min_trusted_observations=1,
        min_new_observations=1,
        cooldown_episodes=0,
    )
    active = _candidate().model_copy(update={"memory_id": "memory", "status": MemoryStatus.ACTIVE})
    pool.seed_accepted([active])

    pool.mark_memory_retired(active.model_copy(update={"status": MemoryStatus.RETIRED}))
    evidence = pool.observe(_candidate("e2"), 20)

    assert pool.readiness(evidence, 20) == (True, "ready")


def test_candidate_pool_tracks_shadow_trust_and_can_require_trusted_evidence() -> None:
    pool = CandidateEvidencePool(
        min_observations=1,
        min_trusted_observations=1,
        min_new_observations=1,
        cooldown_episodes=0,
    )

    evidence = pool.observe(_candidate("shadow"), 1, trust=0.10, trusted=False)

    assert evidence.shadow_observation_count == 1
    assert evidence.trusted_observation_count == 0
    assert evidence.has_shadow_evidence is True
    assert evidence.mean_trust == 0.10
    assert pool.readiness(evidence, 1) == (False, "insufficient_trusted_observations")

    evidence = pool.observe(_candidate("trusted"), 2, trust=0.80, trusted=True)

    assert evidence.trusted_observation_count == 1
    assert evidence.shadow_observation_count == 1
    assert evidence.min_trust == 0.10
    assert evidence.max_trust == 0.80
    assert evidence.mean_trust == 0.45
    assert pool.readiness(evidence, 2) == (True, "ready")


def test_candidate_pool_can_require_mature_feedback_after_initial_consensus() -> None:
    pool = CandidateEvidencePool(
        min_observations=1,
        min_trusted_observations=1,
        min_mature_observations=1,
        min_new_observations=1,
        cooldown_episodes=0,
    )

    provisional = pool.observe(_candidate("initial"), 1, mature=False)
    assert provisional.mature_observation_count == 0
    assert pool.readiness(provisional, 1) == (
        False,
        "insufficient_mature_feedback_evidence",
    )

    mature = pool.observe(_candidate("stable"), 2, mature=True)
    assert mature is provisional
    assert mature.mature_observation_count == 1
    assert pool.readiness(mature, 2) == (True, "ready")


def test_shadow_eprocess_excludes_discovery_then_crosses_and_resets() -> None:
    pool = CandidateEvidencePool(
        min_observations=1,
        min_trusted_observations=0,
        min_new_observations=1,
        cooldown_episodes=4,
        shadow_eprocess_enabled=True,
        shadow_eprocess_null_match_probability=0.25,
        shadow_eprocess_alternative_match_probability=0.75,
        shadow_eprocess_alpha=0.05,
    )

    evidence = pool.observe(_candidate("s1"), 1, trust=0.10, trusted=False)
    assert evidence.shadow_e_value == 1.0
    assert evidence.shadow_eprocess_opportunities_since_validation == 0
    assert pool.readiness(evidence, 1, trusted=False) == (
        False,
        "shadow_eprocess_below_threshold",
    )

    pool.observe(_candidate("s2"), 2, trust=0.10, trusted=False)
    assert evidence.shadow_e_value == 3.0
    assert pool.readiness(evidence, 2, trusted=False) == (
        False,
        "shadow_eprocess_below_threshold",
    )

    pool.observe(_candidate("s3"), 3, trust=0.10, trusted=False)
    assert evidence.shadow_e_value == 9.0
    assert pool.readiness(evidence, 3, trusted=False) == (
        False,
        "shadow_eprocess_below_threshold",
    )

    pool.observe(_candidate("s4"), 4, trust=0.10, trusted=False)
    assert evidence.shadow_e_value == 27.0
    assert evidence.shadow_eprocess_crossings == 1
    assert evidence.shadow_eprocess_ready is True
    assert pool.shadow_eprocess_crossings == 1
    assert pool.readiness(evidence, 4, trusted=False) == (True, "ready")

    pool.observe(_different_candidate("before-validation"), 5, trust=0.10, trusted=False)
    assert evidence.shadow_e_value == 27.0
    assert evidence.shadow_eprocess_ready is True

    pool.mark_validated(evidence, 5, trusted=False)
    assert evidence.shadow_e_value == 1.0
    assert evidence.shadow_eprocess_ready is False
    assert evidence.shadow_eprocess_resets == 1
    assert evidence.shadow_eprocess_opportunities_since_validation == 0

    pool.observe(_different_candidate(), 6, trust=0.10, trusted=False)
    assert evidence.shadow_e_value == 1.0 / 3.0


def test_trusted_lane_bypasses_shadow_cooldown_and_uses_trusted_start_index() -> None:
    pool = CandidateEvidencePool(
        min_observations=1,
        min_trusted_observations=0,
        min_new_observations=1,
        cooldown_episodes=100,
    )
    evidence = pool.observe(_candidate("shadow"), 1, trust=0.10, trusted=False)
    pool.mark_validated(evidence, 1, trusted=False)

    evidence = pool.observe(_candidate("trusted"), 2, trust=0.80, trusted=True)

    assert evidence.first_shadow_episode_index == 1
    assert evidence.first_trusted_episode_index == 2
    assert pool.shadow_cooldown_would_block(evidence, 2) is True
    assert pool.readiness(evidence, 2, trusted=True) == (True, "ready")
