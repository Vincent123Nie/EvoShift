from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from evoshift.schemas import (
    Episode,
    MemoryItem,
    MemoryStatus,
    PolicyGenome,
    PromotionDecision,
    RunManifest,
)

SCHEMA_VERSION = 1


class SQLiteStore:
    """Auditable SQLite state for runs, episodes, memories, and policy versions.

    The store deliberately persists structured summaries rather than hidden model
    chain-of-thought. Every mutation is wrapped in a transaction and the database
    runs in WAL mode so experiment readers do not block the online loop.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def _migrate(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    episode_index INTEGER NOT NULL,
                    domain TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    score REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_run_index
                    ON episodes(run_id, episode_index);
                CREATE INDEX IF NOT EXISTS idx_episodes_domain
                    ON episodes(run_id, domain, episode_index);

                CREATE TABLE IF NOT EXISTS memory_items (
                    memory_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(memory_id, version)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_status
                    ON memory_items(status, updated_at);

                CREATE TABLE IF NOT EXISTS policies (
                    version INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    parent_version INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS validations (
                    validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    candidate_type TEXT NOT NULL,
                    promoted INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS evolution_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                """
            )
            self._connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
                (SCHEMA_VERSION,),
            )

    def save_run(self, manifest: RunManifest, status: str = "running") -> None:
        payload = manifest.model_dump_json()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO runs(run_id, status, manifest_json)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    manifest_json=excluded.manifest_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (manifest.run_id, status, payload),
            )

    def get_run(self, run_id: str) -> Optional[RunManifest]:
        row = self._connection.execute(
            "SELECT manifest_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return RunManifest.model_validate_json(row[0]) if row else None

    def save_episode(self, episode: Episode) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes(
                    episode_id, run_id, episode_index, domain, phase,
                    success, score, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    episode.run_id,
                    episode.index,
                    episode.sample.domain,
                    episode.sample.phase,
                    int(episode.score.success),
                    episode.score.primary,
                    episode.model_dump_json(),
                ),
            )

    def list_episodes(
        self,
        run_id: str,
        limit: int = 0,
        domains: Optional[Sequence[str]] = None,
        newest_first: bool = False,
    ) -> List[Episode]:
        clauses = ["run_id=?"]
        params: List[Any] = [run_id]
        if domains:
            placeholders = ",".join("?" for _ in domains)
            clauses.append(f"domain IN ({placeholders})")
            params.extend(domains)
        order = "DESC" if newest_first else "ASC"
        query = (
            "SELECT payload_json FROM episodes WHERE "
            + " AND ".join(clauses)
            + f" ORDER BY episode_index {order}"
        )
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._connection.execute(query, params).fetchall()
        episodes = [Episode.model_validate_json(row[0]) for row in rows]
        if newest_first:
            return episodes
        return episodes

    def save_memory(self, item: MemoryItem) -> None:
        with self.transaction() as conn:
            self._save_memory_in_transaction(conn, item)

    def save_memories_atomic(self, items: Sequence[MemoryItem]) -> None:
        """Persist a related memory-state transition in one SQLite transaction."""

        if not items:
            return
        with self.transaction() as conn:
            for item in items:
                self._save_memory_in_transaction(conn, item)

    @staticmethod
    def _save_memory_in_transaction(
        conn: sqlite3.Connection,
        item: MemoryItem,
    ) -> None:
        if item.status == MemoryStatus.ACTIVE:
            rows = conn.execute(
                """
                SELECT version, payload_json FROM memory_items
                WHERE memory_id=? AND version<>? AND status='active'
                """,
                (item.memory_id, item.version),
            ).fetchall()
            for row in rows:
                previous = MemoryItem.model_validate_json(row["payload_json"])
                retired = previous.model_copy(update={"status": MemoryStatus.RETIRED})
                conn.execute(
                    """
                    UPDATE memory_items SET status='retired', payload_json=?,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE memory_id=? AND version=?
                    """,
                    (retired.model_dump_json(), item.memory_id, row["version"]),
                )
        conn.execute(
            """
            INSERT OR REPLACE INTO memory_items(
                memory_id, version, status, kind, payload_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                item.memory_id,
                item.version,
                item.status.value,
                item.kind.value,
                item.model_dump_json(),
            ),
        )

    def list_memories(self, statuses: Optional[Sequence[MemoryStatus]] = None) -> List[MemoryItem]:
        params: List[Any] = []
        subquery_filter = ""
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            subquery_filter = f"WHERE status IN ({placeholders})"
            params.extend(status.value for status in statuses)
        rows = self._connection.execute(
            f"""
            SELECT m.payload_json
            FROM memory_items m
            JOIN (
                SELECT memory_id, MAX(version) AS max_version
                FROM memory_items {subquery_filter} GROUP BY memory_id
            ) latest
              ON latest.memory_id=m.memory_id AND latest.max_version=m.version
            ORDER BY m.updated_at DESC, m.memory_id ASC
            """,
            params,
        ).fetchall()
        return [MemoryItem.model_validate_json(row[0]) for row in rows]

    def get_memory(self, memory_id: str, version: int = 0) -> Optional[MemoryItem]:
        if version > 0:
            row = self._connection.execute(
                "SELECT payload_json FROM memory_items WHERE memory_id=? AND version=?",
                (memory_id, version),
            ).fetchone()
        else:
            row = self._connection.execute(
                """
                SELECT payload_json FROM memory_items
                WHERE memory_id=? ORDER BY version DESC LIMIT 1
                """,
                (memory_id,),
            ).fetchone()
        return MemoryItem.model_validate_json(row[0]) if row else None

    def save_policy(
        self, policy: PolicyGenome, status: str = "active", parent_version: int = 0
    ) -> None:
        with self.transaction() as conn:
            if status == "active":
                conn.execute("UPDATE policies SET status='retired' WHERE status='active'")
            conn.execute(
                """
                INSERT OR REPLACE INTO policies(version, status, payload_json, parent_version)
                VALUES (?, ?, ?, ?)
                """,
                (policy.version, status, policy.model_dump_json(), parent_version or None),
            )

    def load_active_policy(self) -> Optional[PolicyGenome]:
        row = self._connection.execute(
            "SELECT payload_json FROM policies WHERE status='active' ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return PolicyGenome.model_validate_json(row[0]) if row else None

    def save_validation(self, run_id: str, decision: PromotionDecision) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO validations(
                    run_id, candidate_id, candidate_type, promoted, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    decision.result.candidate_id,
                    decision.result.candidate_type,
                    int(decision.promote),
                    decision.model_dump_json(),
                ),
            )

    def record_event(
        self, run_id: str, event_type: str, entity_id: str, payload: Dict[str, Any]
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO evolution_events(run_id, event_type, entity_id, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, event_type, entity_id, json.dumps(payload, sort_keys=True)),
            )

    def counts(self, run_id: str) -> Dict[str, int]:
        episode_count = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM episodes WHERE run_id=?", (run_id,)
            ).fetchone()[0]
        )
        validation_count = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM validations WHERE run_id=?", (run_id,)
            ).fetchone()[0]
        )
        memory_count = int(
            self._connection.execute(
                """
                SELECT COUNT(*) FROM memory_items m
                JOIN (
                    SELECT memory_id, MAX(version) max_version
                    FROM memory_items GROUP BY memory_id
                ) x ON x.memory_id=m.memory_id AND x.max_version=m.version
                WHERE m.status='active'
                """
            ).fetchone()[0]
        )
        return {
            "episodes": episode_count,
            "validations": validation_count,
            "active_memories": memory_count,
        }
