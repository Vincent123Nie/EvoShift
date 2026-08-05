from pathlib import Path

from evoshift.memory import BM25MemoryRetriever, MemoryManager
from evoshift.schemas import MemoryItem, MemoryStatus, PolicyGenome
from evoshift.storage import SQLiteStore


def _memory(memory_id: str, trigger: str, directive: str) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        status=MemoryStatus.ACTIVE,
        trigger=trigger,
        directive=directive,
    )


def test_bm25_retrieval_prefers_matching_memory() -> None:
    retriever = BM25MemoryRetriever()
    policy = PolicyGenome(top_k=1, exploration_weight=0.0, utility_weight=0.0)
    items = [
        _memory("math", "arithmetic addition", "Add carefully."),
        _memory("date", "calendar weekday", "Count days modulo seven."),
    ]

    result = retriever.retrieve("Which weekday is the calendar date?", items, policy)

    assert [entry.item.memory_id for entry in result] == ["date"]
    assert result[0].relevance == 1.0


def test_retrieval_enforces_provenance_domain_by_default() -> None:
    retriever = BM25MemoryRetriever()
    item = _memory(
        "causal",
        "abnormal action causes a later harmful event",
        "Check counterfactual dependence.",
    ).model_copy(update={"source_domains": ["causal_judgement"]})
    policy = PolicyGenome(top_k=1)

    in_domain = retriever.retrieve(
        "Does the earlier action cause the outcome?",
        [item],
        policy,
        domain="causal-judgement",
    )
    out_of_domain = retriever.retrieve(
        "Does the earlier action cause the outcome?",
        [item],
        policy,
        domain="formal_fallacies",
    )

    assert [entry.item.memory_id for entry in in_domain] == ["causal"]
    assert out_of_domain == []
    assert (
        retriever.novelty(
            "Does the earlier action cause the outcome?",
            [item],
            policy,
            domain="formal_fallacies",
        )
        == 1.0
    )


def test_cross_domain_transfer_is_an_explicit_policy_ablation() -> None:
    retriever = BM25MemoryRetriever()
    item = _memory("causal", "causal counterfactual", "Check causation.").model_copy(
        update={"source_domains": ["causal_judgement"]}
    )
    policy = PolicyGenome(top_k=1, allow_cross_domain_transfer=True)

    result = retriever.retrieve(
        "causal counterfactual",
        [item],
        policy,
        domain="formal_fallacies",
    )

    assert [entry.item.memory_id for entry in result] == ["causal"]


def test_utility_cannot_retrieve_a_zero_lexical_match_without_domain_evidence() -> None:
    retriever = BM25MemoryRetriever()
    item = _memory("date", "calendar weekday", "Count days modulo seven.")

    result = retriever.retrieve("multiply 7 by 8", [item], PolicyGenome(top_k=1))

    assert result == []


def test_manager_rolls_back_harmful_memory(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "memory.sqlite3")
    manager = MemoryManager(store)
    policy = PolicyGenome(rollback_min_uses=2, rollback_utility_threshold=0.6)
    item = _memory("bad", "all tasks", "Always answer zero.")
    store.save_memory(item)

    manager.record_outcome(["bad"], success=False, policy=policy)
    outcome = manager.record_outcome(["bad"], success=False, policy=policy)

    assert outcome.rolled_back[0].status == MemoryStatus.RETIRED
    assert store.get_memory("bad").status == MemoryStatus.RETIRED  # type: ignore[union-attr]
    store.close()


def test_confirmed_successor_supersedes_and_rollback_restores_prior_rule(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "memory.sqlite3")
    manager = MemoryManager(store)
    prior = _memory("prior", "refund policy", "Use the 14 day rule.")
    successor = _memory("successor", "refund policy", "Use the premium 30 day rule.").model_copy(
        update={"supersedes_memory_ids": ["prior"]}
    )
    store.save_memory(prior)

    activation = manager.activate(successor, 0.2, 0.1, 0.0)

    assert activation.active.status == MemoryStatus.ACTIVE
    assert [item.memory_id for item in activation.superseded] == ["prior"]
    assert store.get_memory("prior").status == MemoryStatus.SUPERSEDED  # type: ignore[union-attr]

    policy = PolicyGenome(rollback_min_uses=1, rollback_utility_threshold=0.6)
    outcome = manager.record_outcome(["successor"], success=False, policy=policy)

    assert [item.memory_id for item in outcome.rolled_back] == ["successor"]
    assert [item.memory_id for item in outcome.reactivated] == ["prior"]
    assert store.get_memory("prior").status == MemoryStatus.ACTIVE  # type: ignore[union-attr]
    store.close()


def test_active_causal_retirement_persists_ledger_and_restores_prior_rule(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "memory.sqlite3")
    manager = MemoryManager(store)
    prior = _memory("prior", "refund policy", "Use the 14 day rule.")
    successor = _memory("successor", "refund policy", "Use the premium 30 day rule.").model_copy(
        update={"supersedes_memory_ids": ["prior"]}
    )
    store.save_memory(prior)
    manager.activate(successor, 0.2, 0.1, 0.0)
    audited = successor.model_copy(
        update={
            "causal_audit_count": 2,
            "causal_negative_count": 2,
            "causal_delta_sum": -2.0,
            "causal_last_audit_index": 20,
        }
    )

    outcome = manager.apply_active_audit(audited, retire=True, restore_predecessors=True)

    assert [item.memory_id for item in outcome.rolled_back] == ["successor"]
    assert [item.memory_id for item in outcome.reactivated] == ["prior"]
    retired = store.get_memory("successor")
    assert retired is not None
    assert retired.status == MemoryStatus.RETIRED
    assert retired.causal_audit_count == 2
    assert store.get_memory("prior").status == MemoryStatus.ACTIVE  # type: ignore[union-attr]
    store.close()


def test_active_causal_retirement_does_not_eagerly_restore_predecessor_by_default(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "memory.sqlite3")
    manager = MemoryManager(store)
    prior = _memory("prior", "refund policy", "Use the 14 day rule.")
    successor = _memory("successor", "refund policy", "Use the premium 30 day rule.").model_copy(
        update={"supersedes_memory_ids": ["prior"]}
    )
    store.save_memory(prior)
    active = manager.activate(successor, 0.2, 0.1, 0.0).active

    outcome = manager.apply_active_audit(active, retire=True)

    assert outcome.reactivated == ()
    assert store.get_memory("prior").status == MemoryStatus.SUPERSEDED  # type: ignore[union-attr]
    store.close()


def test_confirmed_dormant_revival_resets_regime_specific_causal_ledger(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "memory.sqlite3")
    manager = MemoryManager(store)
    retired = _memory("rule", "refund policy", "Use the recurring rule.").model_copy(
        update={
            "status": MemoryStatus.RETIRED,
            "causal_audit_count": 2,
            "causal_negative_count": 2,
            "causal_delta_sum": -2.0,
            "causal_last_audit_index": 20,
        }
    )
    store.save_memory(retired)

    reactivated = manager.reactivate_retired(retired)

    assert reactivated is not None
    assert reactivated.status == MemoryStatus.ACTIVE
    assert reactivated.causal_audit_count == 0
    assert reactivated.causal_negative_count == 0
    assert reactivated.causal_delta_sum == 0.0
    assert reactivated.causal_last_audit_index is None
    store.close()


def test_dormant_revival_rejects_non_latest_retired_version(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "memory.sqlite3")
    manager = MemoryManager(store)
    retired = _memory("rule", "refund policy", "Use the recurring rule.").model_copy(
        update={"status": MemoryStatus.RETIRED}
    )
    store.save_memory(retired)
    store.save_memory(
        retired.model_copy(
            update={
                "version": 2,
                "status": MemoryStatus.REJECTED,
                "directive": "Rejected newer draft.",
            }
        )
    )

    assert manager.reactivate_retired(retired) is None
    assert store.get_memory("rule", version=1).status == MemoryStatus.RETIRED  # type: ignore[union-attr]
    store.close()


def test_probation_memory_is_retrievable_without_superseding_prior_rule(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "memory.sqlite3")
    manager = MemoryManager(store)
    prior = _memory("prior", "refund policy", "Use the 14 day rule.")
    successor = _memory("successor", "refund policy", "Use the premium 30 day rule.").model_copy(
        update={"supersedes_memory_ids": ["prior"]}
    )
    store.save_memory(prior)
    probation = manager.probation(successor, 0.2, 0.1, 0.0)

    assert probation.status == MemoryStatus.PROBATION
    assert {item.memory_id for item in manager.active()} == {"prior", "successor"}
    assert store.get_memory("prior").status == MemoryStatus.ACTIVE  # type: ignore[union-attr]
    store.close()


def test_probation_lifecycle_is_not_preempted_by_posterior_rollback(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "memory.sqlite3")
    manager = MemoryManager(store)
    probation = _memory("candidate", "refund policy", "Use the 14 day rule.").model_copy(
        update={"status": MemoryStatus.PROBATION}
    )
    store.save_memory(probation)
    policy = PolicyGenome(rollback_min_uses=1, rollback_utility_threshold=0.9)

    outcome = manager.record_outcome(["candidate"], success=False, policy=policy)

    assert outcome.rolled_back == ()
    assert store.get_memory("candidate").status == MemoryStatus.PROBATION  # type: ignore[union-attr]
    store.close()
