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
    evidence_source: str = ""
    evidence_context: str = ""
    last_validation_episode: int = -1
    last_validation_observation_count: int = 0
    first_trusted_episode_index: int | None = None
    first_shadow_episode_index: int | None = None
    last_trusted_validation_episode: int = -1
    last_trusted_validation_observation_count: int = 0
    last_shadow_validation_episode: int = -1
    last_shadow_validation_observation_count: int = 0
    trusted_observation_count: int = 0
    shadow_observation_count: int = 0
    trust_sum: float = 0.0
    min_trust: float = 1.0
    max_trust: float = 0.0
    shadow_e_value: float = 1.0
    shadow_eprocess_ready: bool = False
    shadow_eprocess_opportunities_since_validation: int = 0
    shadow_eprocess_crossings: int = 0
    shadow_eprocess_resets: int = 0
    accepted: bool = False
    probationary: bool = False

    @property
    def mean_trust(self) -> float:
        return self.trust_sum / self.observation_count if self.observation_count else 0.0

    @property
    def has_shadow_evidence(self) -> bool:
        return self.shadow_observation_count > 0


class CandidateEvidencePool:
    """Aggregate identical proposals and schedule bounded replay attempts."""

    def __init__(
        self,
        *,
        min_observations: int,
        min_trusted_observations: int,
        min_new_observations: int,
        cooldown_episodes: int,
        context_scoped: bool = False,
        shadow_eprocess_enabled: bool = False,
        shadow_eprocess_null_match_probability: float = 0.25,
        shadow_eprocess_alternative_match_probability: float = 0.75,
        shadow_eprocess_alpha: float = 0.05,
    ) -> None:
        if not 0.0 < shadow_eprocess_alpha < 1.0:
            raise ValueError("shadow e-process alpha must be in (0, 1)")
        if not (
            0.0
            < shadow_eprocess_null_match_probability
            < shadow_eprocess_alternative_match_probability
            < 1.0
        ):
            raise ValueError(
                "shadow e-process probabilities must satisfy 0 < null < alternative < 1"
            )
        self.min_observations = min_observations
        self.min_trusted_observations = min_trusted_observations
        self.min_new_observations = min_new_observations
        self.cooldown_episodes = cooldown_episodes
        self.context_scoped = context_scoped
        self.shadow_eprocess_enabled = shadow_eprocess_enabled
        self.shadow_eprocess_null_match_probability = shadow_eprocess_null_match_probability
        self.shadow_eprocess_alternative_match_probability = (
            shadow_eprocess_alternative_match_probability
        )
        self.shadow_eprocess_threshold = 1.0 / shadow_eprocess_alpha
        self.shadow_eprocess_opportunities = 0
        self.shadow_eprocess_crossings = 0
        self._items: Dict[tuple[str, str, str], CandidateEvidence] = {}
        self._accepted_signatures: set[str] = set()

    @staticmethod
    def _normalize_context_key(context_key: tuple[str, str] | None) -> tuple[str, str]:
        source, context = context_key or ("", "")
        return " ".join(str(source).casefold().split()), " ".join(str(context).casefold().split())

    def _item_key(
        self,
        signature: str,
        context_key: tuple[str, str] | None,
    ) -> tuple[str, str, str]:
        if not self.context_scoped:
            return signature, "", ""
        source, context = self._normalize_context_key(context_key)
        return signature, source, context

    def seed_accepted(self, memories: Iterable[MemoryItem]) -> None:
        for memory in memories:
            signature = candidate_signature(memory)
            self._items[(signature, "", "")] = CandidateEvidence(
                signature=signature,
                candidate=memory,
                observation_count=max(1, len(memory.provenance_episode_ids)),
                first_episode_index=-1,
                last_episode_index=-1,
                first_trusted_episode_index=-1,
                trusted_observation_count=max(1, len(memory.provenance_episode_ids)),
                trust_sum=float(max(1, len(memory.provenance_episode_ids))),
                min_trust=1.0,
                max_trust=1.0,
                accepted=True,
            )
            self._accepted_signatures.add(signature)

    def observe(
        self,
        candidate: MemoryItem,
        episode_index: int,
        *,
        trust: float = 1.0,
        trusted: bool = True,
        context_key: tuple[str, str] | None = None,
    ) -> CandidateEvidence:
        if not 0.0 <= trust <= 1.0:
            raise ValueError("candidate evidence trust must be in [0, 1]")
        signature = candidate_signature(candidate)
        source, context = self._normalize_context_key(context_key)
        item_key = self._item_key(signature, (source, context))
        existing = self._items.get(item_key)
        if not trusted and self.shadow_eprocess_enabled:
            for item in self._items.values():
                if item.has_shadow_evidence and not item.accepted and not item.probationary:
                    same_context = not self.context_scoped or (
                        item.evidence_source,
                        item.evidence_context,
                    ) == (source, context)
                    if same_context:
                        self._update_shadow_eprocess(item, matched=item.signature == signature)
        if existing is None:
            evidence = CandidateEvidence(
                signature=signature,
                candidate=candidate,
                observation_count=1,
                first_episode_index=episode_index,
                last_episode_index=episode_index,
                evidence_source=source,
                evidence_context=context,
                first_trusted_episode_index=episode_index if trusted else None,
                first_shadow_episode_index=episode_index if not trusted else None,
                trusted_observation_count=int(trusted),
                shadow_observation_count=int(not trusted),
                trust_sum=trust,
                min_trust=trust,
                max_trust=trust,
            )
            self._items[item_key] = evidence
            return evidence

        provenance = list(
            dict.fromkeys(
                existing.candidate.provenance_episode_ids + candidate.provenance_episode_ids
            )
        )
        source_domains = list(
            dict.fromkeys(existing.candidate.source_domains + candidate.source_domains)
        )
        supersedes = list(
            dict.fromkeys(
                existing.candidate.supersedes_memory_ids + candidate.supersedes_memory_ids
            )
        )
        existing.candidate = candidate.model_copy(
            update={
                "provenance_episode_ids": provenance,
                "source_domains": source_domains,
                "supersedes_memory_ids": supersedes,
                "valid_from_episode_id": existing.candidate.valid_from_episode_id,
                "valid_from_index": min(
                    existing.candidate.valid_from_index,
                    candidate.valid_from_index,
                ),
                "confidence": max(existing.candidate.confidence, candidate.confidence),
            }
        )
        existing.observation_count += 1
        existing.trusted_observation_count += int(trusted)
        existing.shadow_observation_count += int(not trusted)
        if trusted and existing.first_trusted_episode_index is None:
            existing.first_trusted_episode_index = episode_index
        if not trusted and existing.first_shadow_episode_index is None:
            existing.first_shadow_episode_index = episode_index
        existing.trust_sum += trust
        existing.min_trust = min(existing.min_trust, trust)
        existing.max_trust = max(existing.max_trust, trust)
        existing.last_episode_index = episode_index
        return existing

    def readiness(
        self,
        evidence: CandidateEvidence,
        episode_index: int,
        *,
        trusted: bool = True,
    ) -> tuple[bool, str]:
        if evidence.accepted or evidence.signature in self._accepted_signatures:
            return False, "duplicate_of_active_memory"
        if evidence.probationary:
            return False, "candidate_in_probation"
        if evidence.observation_count < self.min_observations:
            return False, "insufficient_observations"
        if evidence.trusted_observation_count < self.min_trusted_observations:
            return False, "insufficient_trusted_observations"
        if not trusted and self.shadow_eprocess_enabled and not evidence.shadow_eprocess_ready:
            return False, "shadow_eprocess_below_threshold"
        observation_count = (
            evidence.trusted_observation_count if trusted else evidence.shadow_observation_count
        )
        last_validation_observation_count = (
            evidence.last_trusted_validation_observation_count
            if trusted
            else evidence.last_shadow_validation_observation_count
        )
        new_observations = observation_count - last_validation_observation_count
        if new_observations < self.min_new_observations:
            return False, "insufficient_new_evidence"
        last_validation_episode = (
            evidence.last_trusted_validation_episode
            if trusted
            else evidence.last_shadow_validation_episode
        )
        if (
            last_validation_episode >= 0
            and episode_index - last_validation_episode < self.cooldown_episodes
        ):
            return False, "candidate_cooldown"
        return True, "ready"

    def mark_validated(
        self,
        evidence: CandidateEvidence,
        episode_index: int,
        *,
        trusted: bool = True,
    ) -> None:
        evidence.last_validation_episode = episode_index
        evidence.last_validation_observation_count = evidence.observation_count
        if trusted:
            evidence.last_trusted_validation_episode = episode_index
            evidence.last_trusted_validation_observation_count = evidence.trusted_observation_count
        else:
            evidence.last_shadow_validation_episode = episode_index
            evidence.last_shadow_validation_observation_count = evidence.shadow_observation_count
            if self.shadow_eprocess_enabled:
                evidence.shadow_e_value = 1.0
                evidence.shadow_eprocess_ready = False
                evidence.shadow_eprocess_opportunities_since_validation = 0
                evidence.shadow_eprocess_resets += 1

    def shadow_cooldown_would_block(
        self,
        evidence: CandidateEvidence,
        episode_index: int,
    ) -> bool:
        return (
            evidence.last_shadow_validation_episode >= 0
            and episode_index - evidence.last_shadow_validation_episode < self.cooldown_episodes
        )

    def _update_shadow_eprocess(
        self,
        evidence: CandidateEvidence,
        *,
        matched: bool,
    ) -> None:
        if evidence.shadow_eprocess_ready:
            return
        previous = evidence.shadow_e_value
        if matched:
            multiplier = (
                self.shadow_eprocess_alternative_match_probability
                / self.shadow_eprocess_null_match_probability
            )
        else:
            multiplier = (1.0 - self.shadow_eprocess_alternative_match_probability) / (
                1.0 - self.shadow_eprocess_null_match_probability
            )
        evidence.shadow_e_value *= multiplier
        evidence.shadow_eprocess_opportunities_since_validation += 1
        self.shadow_eprocess_opportunities += 1
        if previous < self.shadow_eprocess_threshold <= evidence.shadow_e_value:
            evidence.shadow_eprocess_ready = True
            evidence.shadow_eprocess_crossings += 1
            self.shadow_eprocess_crossings += 1

    def mark_accepted(self, evidence: CandidateEvidence) -> None:
        evidence.accepted = True
        evidence.probationary = False
        self._accepted_signatures.add(evidence.signature)

    @staticmethod
    def mark_probation(evidence: CandidateEvidence) -> None:
        evidence.probationary = True

    @staticmethod
    def mark_rejected(evidence: CandidateEvidence) -> None:
        evidence.accepted = False
        evidence.probationary = False

    def mark_memory_retired(self, memory: MemoryItem) -> None:
        signature = candidate_signature(memory)
        self._accepted_signatures.discard(signature)
        for evidence in self._items.values():
            if evidence.signature == signature:
                self.mark_rejected(evidence)


__all__ = [
    "CandidateEvidence",
    "CandidateEvidencePool",
    "candidate_signature",
]
