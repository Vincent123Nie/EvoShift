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
