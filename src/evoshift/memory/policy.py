from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Iterable

from pydantic import ValidationError

from evoshift.errors import EvolutionError
from evoshift.schemas import FailureRecord, FailureType, PolicyGenome, PolicyPatch, ShiftReport


def apply_policy_patch(base: PolicyGenome, patch: PolicyPatch) -> PolicyGenome:
    if patch.base_version != base.version:
        raise EvolutionError(
            f"stale policy patch: base={patch.base_version}, active={base.version}"
        )
    data = base.model_dump()
    data.update(patch.changes)
    data["version"] = base.version + 1
    try:
        return PolicyGenome.model_validate(data)
    except ValidationError as exc:
        raise EvolutionError(f"policy patch violates bounded genome schema: {exc}") from exc


def propose_bounded_policy_patch(
    base: PolicyGenome,
    failures: Iterable[FailureRecord],
    shift: ShiftReport,
) -> PolicyPatch:
    """Deterministic slow-loop mutation; it changes data, never executable code."""

    counts = Counter(failure.failure_type for failure in failures)
    changes: dict[str, object]
    hypothesis: str
    if counts[FailureType.RETRIEVAL_MISS] + counts[FailureType.WRITE_MISS] > 0:
        changes = {
            "top_k": min(20, base.top_k + 1),
            "relevance_weight": min(2.0, round(base.relevance_weight + 0.08, 4)),
        }
        hypothesis = "Broaden retrieval after drift-associated write/retrieval misses."
    elif counts[FailureType.RANKING_ERROR] > 0:
        changes = {
            "utility_weight": min(2.0, round(base.utility_weight + 0.08, 4)),
            "mmr_lambda": min(1.0, round(base.mmr_lambda + 0.05, 4)),
        }
        hypothesis = "Increase learned utility and relevance pressure after ranking errors."
    elif counts[FailureType.CONFLICT_ERROR] > 0:
        changes = {
            "top_k": max(1, base.top_k - 1),
            "dedup_similarity_threshold": max(
                0.0, round(base.dedup_similarity_threshold - 0.04, 4)
            ),
        }
        hypothesis = "Reduce conflicting context and merge near-duplicate memories earlier."
    else:
        direction = 1 if shift.novelty >= 0.5 else -1
        changes = {
            "exploration_weight": min(
                2.0, max(0.0, round(base.exploration_weight + direction * 0.05, 4))
            )
        }
        hypothesis = "Adjust exploration in response to detected stream novelty."
    fingerprint = hashlib.sha256(
        json.dumps(
            {"base": base.version, "changes": changes, "episode": shift.episode_index},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return PolicyPatch(
        patch_id=f"policy-{fingerprint}",
        base_version=base.version,
        changes=changes,
        hypothesis=hypothesis,
        failure_ids=[failure.failure_id for failure in failures],
        confidence=0.6,
    )
