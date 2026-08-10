from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
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


def candidate_cluster_signature(candidate: MemoryItem) -> str:
    """Return the opaque, code-derived observable anchor for a candidate.

    Older/manual candidates do not carry an anchor; using their exact signature
    preserves the pre-existing behavior while keeping the hierarchical path
    opt-in and fail-closed.
    """

    return candidate.evidence_cluster_key or candidate_signature(candidate)


def observable_candidate_cluster_key(
    *,
    domain: str,
    feedback_source: str,
    feedback_context: str,
    feedback_signal: str,
) -> str:
    """Hash only learner-visible feedback anchors into an opaque key."""

    payload = {
        "domain": " ".join(str(domain).casefold().split()),
        "feedback_source": " ".join(str(feedback_source).casefold().split()) or "unspecified",
        "feedback_context": " ".join(str(feedback_context).casefold().split()),
        "feedback_signal": " ".join(str(feedback_signal).casefold().split()),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"cluster-{digest}"


@dataclass
class CandidateEvidence:
    signature: str
    cluster_signature: str
    candidate: MemoryItem
    observation_count: int
    first_episode_index: int
    last_episode_index: int
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


@dataclass
class CandidateClusterEvidence:
    signature: str
    first_episode_index: int
    last_episode_index: int
    shadow_observation_count: int = 0
    shadow_e_value: float = 1.0
    shadow_eprocess_ready: bool = False
    shadow_eprocess_opportunities_since_validation: int = 0
    shadow_eprocess_crossings: int = 0
    shadow_eprocess_resets: int = 0
    accepted: bool = False
    probationary: bool = False
    member_signatures: set[str] = field(default_factory=set)


class CandidateEvidencePool:
    """Aggregate identical proposals and schedule bounded replay attempts."""

    def __init__(
        self,
        *,
        min_observations: int,
        min_trusted_observations: int,
        min_new_observations: int,
        cooldown_episodes: int,
        shadow_eprocess_enabled: bool = False,
        shadow_eprocess_null_match_probability: float = 0.25,
        shadow_eprocess_alternative_match_probability: float = 0.75,
        shadow_eprocess_alpha: float = 0.05,
        shadow_hierarchical_eprocess_enabled: bool = False,
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
        self.shadow_eprocess_enabled = shadow_eprocess_enabled
        self.shadow_hierarchical_eprocess_enabled = shadow_hierarchical_eprocess_enabled
        self.shadow_eprocess_null_match_probability = shadow_eprocess_null_match_probability
        self.shadow_eprocess_alternative_match_probability = (
            shadow_eprocess_alternative_match_probability
        )
        self.shadow_eprocess_threshold = 1.0 / shadow_eprocess_alpha
        self.shadow_eprocess_opportunities = 0
        self.shadow_eprocess_crossings = 0
        self.shadow_cluster_eprocess_opportunities = 0
        self.shadow_cluster_eprocess_crossings = 0
        self._items: Dict[str, CandidateEvidence] = {}
        self._clusters: Dict[str, CandidateClusterEvidence] = {}

    def seed_accepted(self, memories: Iterable[MemoryItem]) -> None:
        for memory in memories:
            signature = candidate_signature(memory)
            cluster_signature = candidate_cluster_signature(memory)
            cluster = self._clusters.setdefault(
                cluster_signature,
                CandidateClusterEvidence(
                    signature=cluster_signature,
                    first_episode_index=-1,
                    last_episode_index=-1,
                    accepted=True,
                ),
            )
            cluster.member_signatures.add(signature)
            self._items[signature] = CandidateEvidence(
                signature=signature,
                cluster_signature=cluster_signature,
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

    def observe(
        self,
        candidate: MemoryItem,
        episode_index: int,
        *,
        trust: float = 1.0,
        trusted: bool = True,
    ) -> CandidateEvidence:
        if not 0.0 <= trust <= 1.0:
            raise ValueError("candidate evidence trust must be in [0, 1]")
        signature = candidate_signature(candidate)
        cluster_signature = candidate_cluster_signature(candidate)
        cluster = self._clusters.get(cluster_signature)
        if cluster is None:
            cluster = CandidateClusterEvidence(
                signature=cluster_signature,
                first_episode_index=episode_index,
                last_episode_index=episode_index,
            )
            self._clusters[cluster_signature] = cluster
        cluster.member_signatures.add(signature)
        cluster.last_episode_index = episode_index
        existing = self._items.get(signature)
        if not trusted and self.shadow_eprocess_enabled:
            for item in self._items.values():
                if item.has_shadow_evidence and not item.accepted and not item.probationary:
                    self._update_shadow_eprocess(item, matched=item.signature == signature)
            if self.shadow_hierarchical_eprocess_enabled:
                for cluster_item in self._clusters.values():
                    if (
                        cluster_item.shadow_observation_count > 0
                        and not cluster_item.accepted
                        and not cluster_item.probationary
                    ):
                        self._update_cluster_eprocess(
                            cluster_item,
                            matched=cluster_item.signature == cluster_signature,
                        )
                cluster.shadow_observation_count += 1
            else:
                cluster.shadow_observation_count += 1
        if existing is None:
            evidence = CandidateEvidence(
                signature=signature,
                cluster_signature=cluster_signature,
                candidate=candidate,
                observation_count=1,
                first_episode_index=episode_index,
                last_episode_index=episode_index,
                first_trusted_episode_index=episode_index if trusted else None,
                first_shadow_episode_index=episode_index if not trusted else None,
                trusted_observation_count=int(trusted),
                shadow_observation_count=int(not trusted),
                trust_sum=trust,
                min_trust=trust,
                max_trust=trust,
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
        if evidence.accepted:
            return False, "duplicate_of_active_memory"
        if evidence.probationary:
            return False, "candidate_in_probation"
        cluster = self._clusters.get(evidence.cluster_signature)
        if not trusted and cluster is not None and cluster.accepted:
            return False, "cluster_duplicate_of_active_memory"
        if not trusted and cluster is not None and cluster.probationary:
            return False, "cluster_in_probation"
        if evidence.observation_count < self.min_observations:
            return False, "insufficient_observations"
        if evidence.trusted_observation_count < self.min_trusted_observations:
            return False, "insufficient_trusted_observations"
        if not trusted and self.shadow_eprocess_enabled and not evidence.shadow_eprocess_ready:
            cluster_ready = (
                self.shadow_hierarchical_eprocess_enabled
                and cluster is not None
                and cluster.shadow_eprocess_ready
            )
            if not cluster_ready:
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
                cluster = self._clusters.get(evidence.cluster_signature)
                if cluster is not None:
                    cluster.shadow_e_value = 1.0
                    cluster.shadow_eprocess_ready = False
                    cluster.shadow_eprocess_opportunities_since_validation = 0
                    cluster.shadow_eprocess_resets += 1

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

    def _update_cluster_eprocess(
        self,
        evidence: CandidateClusterEvidence,
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
        self.shadow_cluster_eprocess_opportunities += 1
        if previous < self.shadow_eprocess_threshold <= evidence.shadow_e_value:
            evidence.shadow_eprocess_ready = True
            evidence.shadow_eprocess_crossings += 1
            self.shadow_cluster_eprocess_crossings += 1

    def mark_accepted(self, evidence: CandidateEvidence) -> None:
        evidence.accepted = True
        evidence.probationary = False
        self._set_cluster_state(evidence, accepted=True, probationary=False)

    def mark_probation(self, evidence: CandidateEvidence) -> None:
        evidence.probationary = True
        self._set_cluster_state(evidence, accepted=False, probationary=True)

    def mark_rejected(self, evidence: CandidateEvidence) -> None:
        evidence.accepted = False
        evidence.probationary = False
        cluster = self._clusters.get(evidence.cluster_signature)
        if cluster is not None and not any(
            item.cluster_signature == evidence.cluster_signature
            and (item.accepted or item.probationary)
            for item in self._items.values()
        ):
            cluster.accepted = False
            cluster.probationary = False

    def _set_cluster_state(
        self,
        evidence: CandidateEvidence,
        *,
        accepted: bool,
        probationary: bool,
    ) -> None:
        cluster = self._clusters.get(evidence.cluster_signature)
        if cluster is not None:
            cluster.accepted = accepted
            cluster.probationary = probationary

    def mark_memory_retired(self, memory: MemoryItem) -> None:
        evidence = self._items.get(candidate_signature(memory))
        if evidence is not None:
            self.mark_rejected(evidence)

    def cluster_snapshot(self) -> dict[str, object]:
        return {
            "count": len(self._clusters),
            "shadow_eprocess_opportunities": self.shadow_cluster_eprocess_opportunities,
            "shadow_eprocess_crossings": self.shadow_cluster_eprocess_crossings,
            "clusters": {
                key: {
                    "opportunities": value.shadow_eprocess_opportunities_since_validation,
                    "crossings": value.shadow_eprocess_crossings,
                    "shadow_e_value": value.shadow_e_value,
                    "ready": value.shadow_eprocess_ready,
                    "members": sorted(value.member_signatures),
                    "accepted": value.accepted,
                    "probationary": value.probationary,
                }
                for key, value in sorted(self._clusters.items())
            },
        }

    def cluster_evidence(self, signature: str) -> CandidateClusterEvidence | None:
        return self._clusters.get(signature)

    def cluster_e_value(self, signature: str) -> float | None:
        evidence = self._clusters.get(signature)
        return evidence.shadow_e_value if evidence is not None else None


__all__ = [
    "CandidateClusterEvidence",
    "CandidateEvidence",
    "CandidateEvidencePool",
    "candidate_cluster_signature",
    "candidate_signature",
    "observable_candidate_cluster_key",
]
