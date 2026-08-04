from __future__ import annotations

from pathlib import Path

from evoshift.runtime.cache import SQLiteLLMCache, canonical_cache_key
from evoshift.schemas import GenerationResponse, LLMUsage


def response(text: str = "answer") -> GenerationResponse:
    return GenerationResponse(
        text=text,
        model="fake-model",
        usage=LLMUsage(input_tokens=3, output_tokens=2, total_tokens=5, cost_usd=0.01),
        response_id="resp-cache",
        raw={"output_text": text},
    )


def test_cache_key_is_canonical_and_namespaced() -> None:
    first = canonical_cache_key({"b": [2, 1], "a": {"x": "值"}}, namespace="experiment")
    second = canonical_cache_key({"a": {"x": "值"}, "b": [2, 1]}, namespace="experiment")
    other = canonical_cache_key({"a": {"x": "值"}, "b": [2, 1]}, namespace="other")
    assert first == second
    assert first != other
    assert first.startswith("v1:")


def test_sqlite_cache_round_trip_hits_delete_and_clear(tmp_path: Path) -> None:
    cache = SQLiteLLMCache(tmp_path / "nested" / "cache.sqlite3", namespace="run-a")
    try:
        key = cache.make_key({"model": "fake", "input": "hello"})
        assert cache.get(key) is None
        cache.put(key, response())
        restored = cache.get(key)
        assert restored == response()
        assert cache.stats().entries == 1
        assert cache.stats().total_hits == 1
        assert cache.delete(key) is True
        assert cache.delete(key) is False

        cache.put(cache.make_key({"n": 1}), response("one"))
        cache.put(cache.make_key({"n": 2}), response("two"))
        assert cache.clear() == 2
        assert cache.stats().entries == 0
    finally:
        cache.close()


def test_namespaces_are_isolated_in_shared_database(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite3"
    first = SQLiteLLMCache(path, namespace="first")
    second = SQLiteLLMCache(path, namespace="second")
    try:
        payload = {"same": "request"}
        first_key = first.make_key(payload)
        second_key = second.make_key(payload)
        first.put(first_key, response("first"))
        second.put(second_key, response("second"))
        assert first.get(first_key) == response("first")
        assert second.get(second_key) == response("second")
        first.clear()
        assert second.get(second_key) == response("second")
    finally:
        first.close()
        second.close()


def test_zero_ttl_expires_entries(tmp_path: Path) -> None:
    cache = SQLiteLLMCache(tmp_path / "ttl.sqlite3", ttl_seconds=0)
    try:
        key = cache.make_key({"request": 1})
        cache.put(key, response())
        assert cache.get(key) is None
        assert cache.stats().entries == 0
    finally:
        cache.close()
