from evoshift.config import EvolutionConfig
from evoshift.evolution import (
    DormantMemoryRevival,
    dormant_candidate_keys,
    order_semantic_dormant_candidates,
)
from evoshift.memory import BM25MemoryRetriever
from evoshift.schemas import MemoryItem, MemoryStatus, PolicyGenome


def _config(**updates: object) -> EvolutionConfig:
    return EvolutionConfig(
        dynamic_feedback_trust_enabled=True,
        dormant_revival_enabled=True,
    ).model_copy(update=updates)


def _retired() -> MemoryItem:
    return MemoryItem(
        memory_id="retired",
        version=2,
        status=MemoryStatus.RETIRED,
        trigger="refund recurrence",
        scope="customer_support/refund_policy",
        directive="Restore the verified recurring refund rule.",
    )


def test_dormant_revival_requires_cooldown_two_gains_and_exact_context() -> None:
    config = _config(dormant_revival_min_retired_age=18)
    revival = DormantMemoryRevival(config)

    assert not revival.retired_long_enough(retired_index=10, episode_index=27)
    assert revival.retired_long_enough(retired_index=10, episode_index=28)
    assert revival.trust_is_probe_eligible(0.10)
    assert not revival.trust_is_probe_eligible(0.60)
    assert revival.qualifies(0.75)
    assert not revival.qualifies(0.5)

    pending = revival.register(
        source="portal",
        context="refund:any:days_8_14",
        memory=_retired(),
        episode_index=28,
        feedback_off=0.0,
        feedback_on=1.0,
    )
    assert pending is not None
    assert (
        revival.match(
            source="portal",
            context="refund:premium:days_15_30",
            episode_index=29,
            retired_latest_versions=[("retired", 2)],
        )
        is None
    )
    matched = revival.match(
        source="portal",
        context="refund:any:days_8_14",
        episode_index=30,
        retired_latest_versions=[("retired", 2)],
    )
    assert matched == pending
    revival.resolve(matched, confirmed=True)
    assert revival.snapshot()["confirmations"] == 1


def test_dormant_revival_invalidates_changed_version_and_expires() -> None:
    config = _config(dormant_revival_max_age=2)
    invalidated = DormantMemoryRevival(config)
    pending = invalidated.register(
        source="portal",
        context="context",
        memory=_retired(),
        episode_index=10,
        feedback_off=0.0,
        feedback_on=1.0,
    )
    assert pending is not None
    assert (
        invalidated.match(
            source="portal",
            context="context",
            episode_index=11,
            retired_latest_versions=[("retired", 3)],
        )
        is None
    )
    assert invalidated.snapshot()["invalidations"] == 1

    expiring = DormantMemoryRevival(config)
    expiring_pending = expiring.register(
        source="portal",
        context="context",
        memory=_retired(),
        episode_index=10,
        feedback_off=0.0,
        feedback_on=1.0,
    )
    assert expiring_pending is not None
    assert expiring.expire(12) == ()
    assert expiring.expire(13) == (expiring_pending,)
    assert expiring.snapshot()["pending"] == 0


def test_semantic_dormant_view_does_not_hide_older_applicable_rule_in_same_scope() -> None:
    applicable = _retired().model_copy(
        update={
            "memory_id": "refund-v2",
            "trigger": "premium refund between fifteen and thirty days",
            "directive": "Approve premium refunds through day thirty.",
        }
    )
    newer_but_wrong = _retired().model_copy(
        update={
            "memory_id": "refund-v3",
            "trigger": "standard refund after seven days",
            "directive": "Reject standard refunds after day seven.",
        }
    )
    retired_indices = {
        (applicable.memory_id, applicable.version): 80,
        (newer_but_wrong.memory_id, newer_but_wrong.version): 100,
    }

    legacy = dormant_candidate_keys(
        [applicable, newer_but_wrong],
        retired_indices,
        semantic_indexed=False,
    )
    semantic = dormant_candidate_keys(
        [applicable, newer_but_wrong],
        retired_indices,
        semantic_indexed=True,
    )

    assert legacy == {("refund-v3", 2)}
    assert semantic == {("refund-v2", 2), ("refund-v3", 2)}

    retriever = BM25MemoryRetriever()
    policy = PolicyGenome(top_k=3, utility_weight=0.0, exploration_weight=0.0)
    ranked = retriever.retrieve(
        "premium refund request on day twenty",
        [applicable, newer_but_wrong],
        policy,
    )
    ordered = order_semantic_dormant_candidates(ranked, retired_indices)
    assert ordered[0].item.memory_id == "refund-v2"


def test_semantic_dormant_order_uses_recency_only_for_equal_scores() -> None:
    older = _retired().model_copy(update={"memory_id": "older"})
    newer = _retired().model_copy(update={"memory_id": "newer"})
    policy = PolicyGenome(top_k=3, utility_weight=0.0, exploration_weight=0.0)
    tied = BM25MemoryRetriever().retrieve("refund rule", [older, newer], policy)

    ordered = order_semantic_dormant_candidates(
        tied,
        {("older", 2): 10, ("newer", 2): 20},
    )

    assert [item.item.memory_id for item in ordered] == ["newer", "older"]
