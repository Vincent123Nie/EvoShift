from __future__ import annotations

import platform
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from evoshift.agents import MemoryAgent
from evoshift.agents.baselines import behavior_for
from evoshift.audit import state_fingerprint
from evoshift.benchmarks.base import BenchmarkAdapter, sample_fingerprint
from evoshift.config import EvoShiftConfig
from evoshift.evaluation import compute_stream_metrics, score_feedback_sample, score_sample
from evoshift.evolution import (
    CandidateEvidencePool,
    ExperienceCritic,
    FeedbackTrustModel,
    PageHinkleyShiftDetector,
)
from evoshift.evolution.replay import ReplayVerifier
from evoshift.memory import MemoryManager, apply_policy_patch
from evoshift.memory.policy import propose_bounded_policy_patch
from evoshift.providers.factory import create_client
from evoshift.runtime import BudgetLedger, RunArtifacts, SQLiteLLMCache
from evoshift.runtime.artifacts import git_state
from evoshift.schemas import (
    Episode,
    FailureRecord,
    MemoryItem,
    MemoryStatus,
    PolicyGenome,
    PromotionDecision,
    RunManifest,
    RunMode,
    ShiftReport,
)
from evoshift.storage import SQLiteStore


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    run_dir: Path
    metrics: Dict[str, Any]
    final_policy: PolicyGenome
    decisions: List[PromotionDecision]


def _new_run_id(algorithm: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{algorithm}-{uuid.uuid4().hex[:8]}"


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


class EvoShiftRunner:
    """Prequential predict-evaluate-evolve loop with verified promotion gates."""

    def __init__(
        self,
        config: EvoShiftConfig,
        adapter: BenchmarkAdapter,
        *,
        workdir: Optional[Path] = None,
        client: Optional[Any] = None,
        initial_memories: Sequence[MemoryItem] = (),
        initial_policy: Optional[PolicyGenome] = None,
        frozen_audit: bool = False,
        source_run_id: str = "",
        source_state_hash: str = "",
        source_dataset_hash: str = "",
    ):
        self.config = config
        self.adapter = adapter
        self.workdir = (workdir or Path.cwd()).resolve()
        self.injected_client = client
        self.initial_memories = tuple(item.model_copy(deep=True) for item in initial_memories)
        self.initial_policy = initial_policy.model_copy(deep=True) if initial_policy else None
        self.frozen_audit = frozen_audit
        self.source_run_id = source_run_id
        self.source_state_hash = source_state_hash
        self.source_dataset_hash = source_dataset_hash
        if self.frozen_audit and not self.source_run_id:
            raise ValueError("frozen audit requires source_run_id provenance")
        if any(item.status != MemoryStatus.ACTIVE for item in self.initial_memories):
            raise ValueError("initial memories must all be active")

    async def run(self) -> ExperimentResult:
        samples = self.adapter.load()
        if not samples:
            raise ValueError("benchmark produced no samples")
        dataset_hash = sample_fingerprint(samples)
        if (
            self.frozen_audit
            and self.source_dataset_hash
            and dataset_hash == self.source_dataset_hash
        ):
            raise ValueError("held-out audit dataset matches the source training stream")
        run_label = "audit" if self.frozen_audit else self.config.algorithm.value
        run_id = _new_run_id(run_label)
        artifacts = RunArtifacts(_resolve(self.workdir, self.config.storage.runs_dir), run_id)
        commit, dirty = git_state(self.workdir)
        manifest = RunManifest(
            run_id=run_id,
            git_commit=commit,
            git_dirty=dirty,
            config_hash=self.config.fingerprint(),
            dataset_hash=dataset_hash,
            model=self.config.provider.resolved_model(),
            algorithm=self.config.algorithm,
            run_mode=(RunMode.FROZEN_AUDIT if self.frozen_audit else RunMode.PREQUENTIAL),
            source_run_id=self.source_run_id,
            source_state_hash=self.source_state_hash,
            source_dataset_hash=self.source_dataset_hash,
            seed=self.config.evaluation.seed,
            python_version=platform.python_version(),
            platform=platform.platform(),
        )
        artifacts.initialize(manifest, self.config)
        state_path = (
            artifacts.run_dir / "state.sqlite3"
            if self.config.storage.isolate_runs
            else _resolve(self.workdir, self.config.storage.database_path)
        )
        store = SQLiteStore(state_path)
        cache: Optional[SQLiteLLMCache] = None
        if self.config.storage.cache_enabled:
            cache = SQLiteLLMCache(
                _resolve(self.workdir, self.config.storage.cache_path),
                namespace=f"{self.config.provider.kind}:{self.config.provider.resolved_model()}",
            )
        budget = BudgetLedger.from_config(self.config.budget)
        client = self.injected_client or create_client(
            self.config.provider, cache=cache, budget=budget
        )
        owns_client = self.injected_client is None

        try:
            store.save_run(manifest)
            policy = self.initial_policy or self.config.policy
            store.save_policy(policy)
            for item in self.initial_memories:
                store.save_memory(item)
            memory = MemoryManager(store)
            agent = MemoryAgent(client, self.config.provider, memory)
            critic = ExperienceCritic(client, self.config.provider, self.config.evolution)
            trust_model = FeedbackTrustModel(self.config.evolution)
            detectors: Dict[str, PageHinkleyShiftDetector] = {}
            regime_starts: Dict[str, int] = {}
            verifier = ReplayVerifier(
                agent,
                self.config.evolution,
                protected_phases=self.config.benchmark.protected_phases,
            )
            candidate_pool = CandidateEvidencePool(
                min_observations=self.config.evolution.candidate_min_observations,
                min_new_observations=(self.config.evolution.candidate_min_new_observations),
                cooldown_episodes=self.config.evolution.candidate_cooldown_episodes,
            )
            candidate_pool.seed_accepted(memory.active())
            behavior = behavior_for(self.config.algorithm)
            episodes: List[Episode] = []
            failures: List[FailureRecord] = []
            decisions: List[PromotionDecision] = []
            rollbacks = 0
            feedback_quarantined = 0
            candidate_observations = 0
            candidate_deferred = 0
            candidate_replay_attempts = 0
            candidate_duplicate_active = 0
            shift_detection_events = 0
            policy_evolution_suppressed_by_memory = 0

            for index, sample in enumerate(samples):
                memory_promoted_this_episode = False
                active_before = memory.active()
                prediction = await agent.solve(
                    sample,
                    policy,
                    use_memory=behavior.use_memory,
                    self_refine=behavior.self_refine,
                )
                score = score_sample(sample, prediction.output.answer)
                feedback_score = score_feedback_sample(sample, prediction.output.answer)
                selected_ids = [item.item.memory_id for item in prediction.retrieved]
                novelty = (
                    1.0 - max(item.relevance for item in prediction.retrieved)
                    if active_before and prediction.retrieved
                    else (1.0 if active_before else 0.0)
                )
                assessment = trust_model.assess(sample)
                feedback_eligible = (
                    assessment.trust >= self.config.evolution.min_feedback_trust_for_candidate
                )
                detector = detectors.setdefault(
                    sample.domain,
                    PageHinkleyShiftDetector(self.config.shift),
                )
                if assessment.trust >= self.config.evolution.min_feedback_trust_for_drift:
                    shift = detector.update(
                        feedback_score.primary,
                        novelty,
                        index,
                        sample.domain,
                    )
                    if shift.detected:
                        regime_starts[sample.domain] = index
                        shift_detection_events += 1
                else:
                    shift = ShiftReport(
                        detector="feedback_trust_gate",
                        domain=sample.domain,
                        episode_index=index,
                        novelty=novelty,
                        reason=(
                            f"feedback quarantined: trust={assessment.trust:.3f} "
                            f"source={assessment.source}"
                        ),
                    )
                episode = Episode(
                    episode_id=f"ep-{uuid.uuid4().hex[:16]}",
                    run_id=run_id,
                    index=index,
                    sample=sample,
                    output=prediction.output,
                    score=score,
                    feedback_score=feedback_score,
                    feedback_trust=assessment.trust,
                    feedback_eligible=feedback_eligible,
                    feedback_trust_reason=assessment.reason,
                    selected_memory_ids=selected_ids,
                    policy_version=policy.version,
                    usage=prediction.usage,
                    shift=shift,
                )
                episodes.append(episode)
                store.save_episode(episode)
                artifacts.append_episode(episode)

                if (
                    not self.frozen_audit
                    and assessment.trust
                    >= self.config.evolution.min_feedback_trust_for_memory_update
                ):
                    credited_ids = prediction.output.applied_memory_ids or selected_ids
                    retired = memory.record_outcome(credited_ids, feedback_score.success, policy)
                    for item in retired:
                        rollbacks += 1
                        payload = {"reason": "posterior utility below rollback threshold"}
                        store.record_event(
                            run_id,
                            "memory_rollback",
                            item.memory_id,
                            payload,
                        )
                elif not self.frozen_audit:
                    feedback_quarantined += 1
                    store.record_event(
                        run_id,
                        "feedback_quarantined",
                        episode.episode_id,
                        {
                            "source": assessment.source,
                            "trust": assessment.trust,
                            "reason": assessment.reason,
                        },
                    )

                should_extract = feedback_eligible and (
                    not feedback_score.success
                    or (
                        policy.learn_from_success_every > 0
                        and (index + 1) % policy.learn_from_success_every == 0
                    )
                )
                if (
                    self.config.evolution.enabled
                    and not self.frozen_audit
                    and behavior.generate_experience
                    and should_extract
                ):
                    failure = await critic.analyze(episode, memory.active(), shift)
                    failures.append(failure)
                    artifacts.append_failure(failure)
                    store.record_event(
                        run_id,
                        "failure_attribution",
                        failure.failure_id,
                        failure.model_dump(mode="json"),
                    )
                    if failure.proposed_memory is not None:
                        if not behavior.verify_before_promotion:
                            candidate = memory.stage(failure.proposed_memory, policy)
                            store.record_event(
                                run_id,
                                "memory_staged",
                                f"{candidate.memory_id}@v{candidate.version}",
                                {"status": candidate.status.value},
                            )
                            if candidate.status != MemoryStatus.REJECTED:
                                active = memory.activate(candidate, 0.0, 0.0, 0.0)
                                store.record_event(
                                    run_id,
                                    "memory_promoted_unverified",
                                    f"{active.memory_id}@v{active.version}",
                                    {"algorithm": self.config.algorithm.value},
                                )
                        else:
                            candidate_observations += 1
                            evidence = candidate_pool.observe(
                                failure.proposed_memory,
                                index,
                            )
                            ready, readiness_reason = candidate_pool.readiness(
                                evidence,
                                index,
                            )
                            if not ready:
                                candidate_deferred += 1
                                if readiness_reason == "duplicate_of_active_memory":
                                    candidate_duplicate_active += 1
                                store.record_event(
                                    run_id,
                                    "candidate_deferred",
                                    evidence.signature,
                                    {
                                        "reason": readiness_reason,
                                        "observation_count": evidence.observation_count,
                                        "last_validation_observation_count": (
                                            evidence.last_validation_observation_count
                                        ),
                                    },
                                )
                            else:
                                regime_start = regime_starts.get(sample.domain)
                                replay_buffer, _ = verifier.memory_buffer(
                                    evidence.candidate,
                                    episodes,
                                    regime_start_index=regime_start,
                                )
                                if (
                                    len(replay_buffer)
                                    < self.config.evolution.min_validation_examples
                                ):
                                    candidate_deferred += 1
                                    store.record_event(
                                        run_id,
                                        "candidate_deferred",
                                        evidence.signature,
                                        {
                                            "reason": "insufficient_trusted_replay_buffer",
                                            "observation_count": evidence.observation_count,
                                            "replay_count": len(replay_buffer),
                                            "regime_start_index": regime_start,
                                        },
                                    )
                                else:
                                    candidate = memory.stage(evidence.candidate, policy)
                                    store.record_event(
                                        run_id,
                                        "memory_staged",
                                        f"{candidate.memory_id}@v{candidate.version}",
                                        {
                                            "status": candidate.status.value,
                                            "candidate_signature": evidence.signature,
                                            "observation_count": evidence.observation_count,
                                        },
                                    )
                                    candidate_pool.mark_validated(evidence, index)
                                    if candidate.status != MemoryStatus.REJECTED:
                                        candidate_replay_attempts += 1
                                        decision = await verifier.validate_memory(
                                            candidate,
                                            episodes,
                                            policy,
                                            regime_start_index=regime_start,
                                        )
                                        decisions.append(decision)
                                        store.save_validation(run_id, decision)
                                        artifacts.append_decision(decision)
                                        if decision.promote:
                                            memory.activate(
                                                candidate,
                                                decision.result.mean_delta,
                                                decision.result.ci_low,
                                                decision.result.regression_rate,
                                            )
                                            candidate_pool.mark_accepted(evidence)
                                            memory_promoted_this_episode = True
                                        else:
                                            memory.reject(candidate)

                if (
                    shift.detected
                    and memory_promoted_this_episode
                    and not self.config.evolution.policy_evolve_if_memory_promoted
                ):
                    policy_evolution_suppressed_by_memory += 1

                if (
                    shift.detected
                    and not self.frozen_audit
                    and self.config.evolution.enabled
                    and self.config.evolution.policy_evolution_enabled
                    and self.config.evolution.policy_evolve_on_shift
                    and (
                        self.config.evolution.policy_evolve_if_memory_promoted
                        or not memory_promoted_this_episode
                    )
                    and behavior.evolve_policy
                    and failures
                    and len(episodes) >= self.config.evolution.min_validation_examples
                ):
                    recent_failures = failures[-self.config.evolution.validation_window :]
                    patch = propose_bounded_policy_patch(policy, recent_failures, shift)
                    challenger = apply_policy_patch(policy, patch)
                    regime_start = regime_starts.get(sample.domain)
                    replay_buffer, _ = verifier.policy_buffer(
                        episodes,
                        regime_start_index=regime_start,
                    )
                    if len(replay_buffer) < self.config.evolution.min_validation_examples:
                        store.record_event(
                            run_id,
                            "policy_patch_deferred",
                            patch.patch_id,
                            {
                                "reason": "insufficient_trusted_replay_buffer",
                                "replay_count": len(replay_buffer),
                                "regime_start_index": regime_start,
                            },
                        )
                        continue
                    decision = await verifier.validate_policy(
                        challenger,
                        policy,
                        episodes,
                        regime_start_index=regime_start,
                    )
                    decisions.append(decision)
                    store.save_validation(run_id, decision)
                    artifacts.append_decision(decision)
                    store.record_event(
                        run_id,
                        "policy_patch_evaluated",
                        patch.patch_id,
                        {
                            "patch": patch.model_dump(mode="json"),
                            "decision": decision.model_dump(mode="json"),
                        },
                    )
                    if decision.promote:
                        store.save_policy(
                            challenger,
                            status="active",
                            parent_version=policy.version,
                        )
                        policy = challenger
                    else:
                        store.save_policy(
                            challenger, status="rejected", parent_version=policy.version
                        )

            final_memories = tuple(memory.active())
            final_state_hash = state_fingerprint(policy, final_memories)
            if (
                self.frozen_audit
                and self.source_state_hash
                and final_state_hash != self.source_state_hash
            ):
                raise RuntimeError("frozen audit state changed during evaluation")

            metrics = compute_stream_metrics(
                episodes,
                recovery_fraction=self.config.evaluation.recovery_fraction,
                promotions=None if self.frozen_audit else decisions,
            )
            if self.frozen_audit:
                metrics["promotion_precision"] = None
            memory_decisions = [
                decision for decision in decisions if decision.result.candidate_type == "memory"
            ]
            policy_decisions = [
                decision for decision in decisions if decision.result.candidate_type == "policy"
            ]
            promoted = sum(int(decision.promote) for decision in decisions)
            promoted_memories = sum(int(decision.promote) for decision in memory_decisions)
            promoted_policies = sum(int(decision.promote) for decision in policy_decisions)
            metrics.update(
                {
                    "algorithm": self.config.algorithm.value,
                    "run_mode": (
                        RunMode.FROZEN_AUDIT.value
                        if self.frozen_audit
                        else RunMode.PREQUENTIAL.value
                    ),
                    "model": self.config.provider.resolved_model(),
                    "config_hash": self.config.fingerprint(),
                    "dataset_hash": dataset_hash,
                    "evolution": {
                        "validation_decisions": len(decisions),
                        "candidates_evaluated": len(decisions),
                        "candidates_promoted": promoted,
                        "candidates_rejected": len(decisions) - promoted,
                        "memory_candidates_evaluated": len(memory_decisions),
                        "memory_candidates_promoted": promoted_memories,
                        "memory_candidates_rejected": (len(memory_decisions) - promoted_memories),
                        "policy_candidates_evaluated": len(policy_decisions),
                        "policy_candidates_promoted": promoted_policies,
                        "policy_candidates_rejected": (len(policy_decisions) - promoted_policies),
                        "policy_evolution_suppressed_by_memory": (
                            policy_evolution_suppressed_by_memory
                        ),
                        "candidate_observations": candidate_observations,
                        "candidate_replay_attempts": candidate_replay_attempts,
                        "candidate_deferred": candidate_deferred,
                        "candidate_duplicate_active": candidate_duplicate_active,
                        "feedback_quarantined": feedback_quarantined,
                        "detector_domains": len(detectors),
                        "shift_detection_events": shift_detection_events,
                        "domains_with_detected_shift": len(regime_starts),
                        "memory_rollbacks": rollbacks,
                        "final_active_memories": len(final_memories),
                        "final_policy_version": policy.version,
                    },
                    "audit": {
                        "frozen": self.frozen_audit,
                        "source_run_id": self.source_run_id,
                        "source_state_hash": self.source_state_hash,
                        "source_dataset_hash": self.source_dataset_hash,
                        "initial_active_memories": len(self.initial_memories),
                        "final_state_hash": final_state_hash,
                        "state_unchanged": (
                            not self.frozen_audit
                            or not self.source_state_hash
                            or final_state_hash == self.source_state_hash
                        ),
                    },
                }
            )
            finished = manifest.model_copy(update={"finished_at": datetime.now(timezone.utc)})
            store.save_run(finished, status="completed")
            snapshot = budget.snapshot().as_dict()
            artifacts.finalize(
                finished,
                metrics,
                snapshot,
                {
                    "store_counts": store.counts(run_id),
                    "final_policy": policy.model_dump(mode="json"),
                    "active_memories": [item.model_dump(mode="json") for item in final_memories],
                    "promotion_decisions": [
                        decision.model_dump(mode="json") for decision in decisions
                    ],
                    "notes": {
                        "resource_metrics": "foreground task calls only",
                        "budget_ledger": "all provider calls including critic and replay",
                        "promotion_precision": (
                            "not applicable in frozen audit"
                            if self.frozen_audit
                            else "replay-estimated; held-out audit required for claims"
                        ),
                        "state_mutation": (
                            "disabled"
                            if self.frozen_audit
                            else "enabled according to algorithm configuration"
                        ),
                    },
                },
            )
            return ExperimentResult(run_id, artifacts.run_dir, metrics, policy, decisions)
        finally:
            if owns_client:
                await client.aclose()
            if cache is not None:
                cache.close()
            store.close()


__all__ = ["EvoShiftRunner", "ExperimentResult"]
