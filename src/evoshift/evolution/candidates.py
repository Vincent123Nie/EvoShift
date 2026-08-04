from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable

from evoshift.schemas import MemoryItem


def candidate_signature(candidate: MemoryItem) -> str:
    payload = {
        "kind": candidate.kind.value,
        "scope": " ".join(candidate.scope.casefold().split()),
        "trigger": " ".join(candidate.trigger.casefold().split()),
        "directive": " ".join(candidate.directive.casefold().split()),
        "anti_pattern": " ".join(candidate.anti_pattern.casefold().split()),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return f"candidate-{digest}"


@dataclass
class CandidateEvidence:
    signature: str
    candidate: MemoryItem
    observation_count: int
    first_episode_index: int
    last_episode_index: int
    last_validation_episode: int = -1
    last_validation_observation_count: int = 0
    accepted: bool = False


class CandidateEvidencePool:
    """Aggregate identical proposals and schedule bounded replay attempts."""

    def __init__(
        self,
        *,
        min_observations: int,
        min_new_observations: int,
        cooldown_episodes: int,
    ) -> None:
        self.min_observations = min_observations
        self.min_new_observations = min_new_observations
        self.cooldown_episodes = cooldown_episodes
        self._items: Dict[str, CandidateEvidence] = {}

    def seed_accepted(self, memories: Iterable[MemoryItem]) -> None:
        for memory in memories:
            signature = candidate_signature(memory)
            self._items[signature] = CandidateEvidence(
                signature=signature,
                candidate=memory,
                observation_count=max(1, len(memory.provenance_episode_ids)),
                first_episode_index=-1,
                last_episode_index=-1,
                accepted=True,
            )

    def observe(self, candidate: MemoryItem, episode_index: int) -> CandidateEvidence:
        signature = candidate_signature(candidate)
        existing = self._items.get(signature)
        if existing is None:
            evidence = CandidateEvidence(
                signature=signature,
                candidate=candidate,
                observation_count=1,
                first_episode_index=episode_index,
                last_episode_index=episode_index,
            )
            self._items[signature] = evidence
            return evidence

        provenance = list(
            dict.fromkeys(
                existing.candidate.provenance_episode_ids + candidate.provenance_episode_ids
            )
        )
        source_domains = list(
            dict.fromkeys(existing.candidate.source_domains + candidate.source_domains)
        )
        existing.candidate = candidate.model_copy(
            update={
                "provenance_episode_ids": provenance,
                "source_domains": source_domains,
                "confidence": max(existing.candidate.confidence, candidate.confidence),
            }
        )
        existing.observation_count += 1
        existing.last_episode_index = episode_index
        return existing

    def readiness(self, evidence: CandidateEvidence, episode_index: int) -> tuple[bool, str]:
        if evidence.accepted:
            return False, "duplicate_of_active_memory"
        if evidence.observation_count < self.min_observations:
            return False, "insufficient_observations"
        new_observations = evidence.observation_count - evidence.last_validation_observation_count
        if new_observations < self.min_new_observations:
            return False, "insufficient_new_evidence"
        if (
            evidence.last_validation_episode >= 0
            and episode_index - evidence.last_validation_episode < self.cooldown_episodes
        ):
            return False, "candidate_cooldown"
        return True, "ready"

    @staticmethod
    def mark_validated(evidence: CandidateEvidence, episode_index: int) -> None:
        evidence.last_validation_episode = episode_index
        evidence.last_validation_observation_count = evidence.observation_count

    @staticmethod
    def mark_accepted(evidence: CandidateEvidence) -> None:
        evidence.accepted = True


__all__ = [
    "CandidateEvidence",
    "CandidateEvidencePool",
    "candidate_signature",
]
