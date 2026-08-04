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
    rolled_back = manager.record_outcome(["bad"], success=False, policy=policy)

    assert rolled_back[0].status == MemoryStatus.RETIRED
    assert store.get_memory("bad").status == MemoryStatus.RETIRED  # type: ignore[union-attr]
    store.close()
