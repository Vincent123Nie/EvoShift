from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Union

from pydantic import ValidationError

from evoshift.schemas import GenerationResponse

JsonMapping = Mapping[str, Any]


def canonical_cache_key(payload: JsonMapping, *, namespace: str = "default") -> str:
    """Create a stable, content-addressed key without provider credentials."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(namespace.encode("utf-8") + b"\0" + encoded).hexdigest()
    return f"v1:{digest}"


@dataclass(frozen=True)
class CacheStats:
    entries: int
    total_hits: int


class LLMCache(Protocol):
    def make_key(self, payload: JsonMapping) -> str: ...

    def get(self, key: str) -> GenerationResponse | None: ...

    def put(self, key: str, response: GenerationResponse) -> None: ...


class SQLiteLLMCache:
    """Small process-safe LLM response cache backed by SQLite.

    The cache stores normalized responses, never API keys. Callers are
    responsible for excluding secrets from the payload passed to ``make_key``.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        namespace: str = "default",
        ttl_seconds: float | None = None,
    ) -> None:
        raw_path = str(path)
        self.path = raw_path if raw_path == ":memory:" else str(Path(raw_path).expanduser())
        self.namespace = namespace
        self.ttl_seconds = ttl_seconds
        if ttl_seconds is not None and ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("PRAGMA busy_timeout = 5000")
            if self.path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (namespace, cache_key)
                )
                """
            )

    def make_key(self, payload: JsonMapping) -> str:
        return canonical_cache_key(payload, namespace=self.namespace)

    def get(self, key: str) -> GenerationResponse | None:
        now = time.time()
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT response_json, created_at
                FROM llm_cache
                WHERE namespace = ? AND cache_key = ?
                """,
                (self.namespace, key),
            ).fetchone()
            if row is None:
                return None
            if self.ttl_seconds is not None and now - float(row["created_at"]) > self.ttl_seconds:
                self._connection.execute(
                    "DELETE FROM llm_cache WHERE namespace = ? AND cache_key = ?",
                    (self.namespace, key),
                )
                return None
            try:
                response = GenerationResponse.model_validate_json(row["response_json"])
            except (ValidationError, ValueError):
                self._connection.execute(
                    "DELETE FROM llm_cache WHERE namespace = ? AND cache_key = ?",
                    (self.namespace, key),
                )
                return None
            self._connection.execute(
                """
                UPDATE llm_cache
                SET accessed_at = ?, hit_count = hit_count + 1
                WHERE namespace = ? AND cache_key = ?
                """,
                (now, self.namespace, key),
            )
            return response

    def put(self, key: str, response: GenerationResponse) -> None:
        now = time.time()
        serialized = response.model_dump_json()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO llm_cache (
                    namespace, cache_key, response_json, created_at, accessed_at, hit_count
                ) VALUES (?, ?, ?, ?, ?, 0)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    response_json = excluded.response_json,
                    created_at = excluded.created_at,
                    accessed_at = excluded.accessed_at,
                    hit_count = 0
                """,
                (self.namespace, key, serialized, now, now),
            )

    def delete(self, key: str) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM llm_cache WHERE namespace = ? AND cache_key = ?",
                (self.namespace, key),
            )
            return cursor.rowcount > 0

    def clear(self) -> int:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM llm_cache WHERE namespace = ?",
                (self.namespace,),
            )
            return max(0, cursor.rowcount)

    def stats(self) -> CacheStats:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS entries, COALESCE(SUM(hit_count), 0) AS total_hits
                FROM llm_cache
                WHERE namespace = ?
                """,
                (self.namespace,),
            ).fetchone()
        assert row is not None
        return CacheStats(entries=int(row["entries"]), total_hits=int(row["total_hits"]))

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteLLMCache:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
