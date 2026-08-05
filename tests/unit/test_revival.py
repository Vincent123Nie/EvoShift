from evoshift.config import EvolutionConfig
from evoshift.evolution import DormantMemoryRevival
from evoshift.schemas import MemoryItem, MemoryStatus


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
