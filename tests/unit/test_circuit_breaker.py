from pathlib import Path

from evoshift.config import EvolutionConfig
from evoshift.evolution import ActiveMemoryAuditor, CausalCircuitBreaker
from evoshift.memory import MemoryManager
from evoshift.schemas import MemoryItem, MemoryStatus
from evoshift.storage import SQLiteStore


def _config(**updates: object) -> EvolutionConfig:
    return EvolutionConfig(
        active_audit_enabled=True,
        dynamic_feedback_trust_enabled=True,
        active_audit_circuit_breaker_enabled=True,
        active_audit_min_observations=2,
        active_audit_min_negative_observations=2,
        active_audit_retire_mean_delta=-0.5,
    ).model_copy(update=updates)


def _memory() -> MemoryItem:
    return MemoryItem(
        memory_id="suspect",
        version=3,
        status=MemoryStatus.ACTIVE,
        trigger="refund context",
        directive="Apply the newer refund rule.",
    )


def _provisional(config: EvolutionConfig, *, index: int = 10):
    return ActiveMemoryAuditor(config).observe(
        _memory(),
        episode_index=index,
        feedback_control=1.0,
        feedback_candidate=0.0,
        retirement_enabled=False,
    )


def test_circuit_breaker_matches_only_same_context_and_exact_active_version() -> None:
    config = _config()
    breaker = CausalCircuitBreaker(config)
    pending = breaker.register(
        source="portal",
        context="refund:any:days_8_14",
        observation=_provisional(config),
    )

    assert pending is not None
    assert (
        breaker.match(
            source="portal",
            context="refund:premium:days_15_30",
            episode_index=11,
            active_memory_versions=[("suspect", 3)],
        )
        is None
    )
    matched = breaker.match(
        source="portal",
        context="refund:any:days_8_14",
        episode_index=12,
        active_memory_versions=[("suspect", 3)],
    )

    assert matched == pending
    breaker.resolve(matched, confirmed=True)
    assert breaker.snapshot()["interventions"] == 1
    assert breaker.snapshot()["confirmations"] == 1


def test_circuit_breaker_expires_after_bounded_ttl() -> None:
    config = _config(active_audit_circuit_breaker_max_age=2)
    breaker = CausalCircuitBreaker(config)
    pending = breaker.register(
        source="portal",
        context="context",
        observation=_provisional(config),
    )

    assert pending is not None
    assert breaker.expire(12) == ()
    assert breaker.expire(13) == (pending,)
    assert breaker.snapshot()["expirations"] == 1
    assert breaker.snapshot()["pending"] == 0


def test_cancelled_canary_restores_exact_persistent_causal_state(tmp_path: Path) -> None:
    config = _config()
    auditor = ActiveMemoryAuditor(config)
    store = SQLiteStore(tmp_path / "state.sqlite3")
    manager = MemoryManager(store)
    original = _memory()
    store.save_memory(original)
    provisional = auditor.observe(
        original,
        episode_index=10,
        feedback_control=1.0,
        feedback_candidate=0.0,
        retirement_enabled=False,
    )
    manager.apply_active_audit(provisional.memory_after, retire=False)
    current = store.get_memory("suspect", version=3)
    assert current is not None

    restored = auditor.revert_observation(current, provisional)
    manager.apply_active_audit(restored, retire=False)

    final = store.get_memory("suspect", version=3)
    assert final == original
    assert final is not None and final.status == MemoryStatus.ACTIVE
    store.close()


def test_circuit_probe_uses_pre_registered_trust_and_delta_gates() -> None:
    config = _config()
    breaker = CausalCircuitBreaker(config)

    assert breaker.trust_is_probe_eligible(0.10)
    assert not breaker.trust_is_probe_eligible(0.60)
    assert breaker.qualifies(-0.75)
    assert not breaker.qualifies(-0.5)
