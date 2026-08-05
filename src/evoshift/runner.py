from __future__ import annotations

import platform
import statistics
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
from evoshift.evaluation import (
    compute_stream_metrics,
    promotion_precision,
    score_feedback_sample,
    score_sample,
)
from evoshift.evolution import (
    ActiveMemoryAuditor,
    CandidateEvidencePool,
    ExperienceCritic,
    FeedbackTrustModel,
    FutureAuditOutcome,
    FutureCounterfactualAuditor,
    PageHinkleyShiftDetector,
)
from evoshift.evolution.candidates import CandidateEvidence
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
            behavior = behavior_for(self.config.algorithm)
            agent = MemoryAgent(client, self.config.provider, memory)
            critic = ExperienceCritic(client, self.config.provider, self.config.evolution)
            trust_model = FeedbackTrustModel(
                self.config.evolution,
                dynamic_enabled=(
                    self.config.evolution.dynamic_feedback_trust_enabled
                    and behavior.dynamic_feedback_trust
                ),
            )
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
            future_auditor = (
                FutureCounterfactualAuditor(self.config.evolution)
                if self.config.evolution.future_audit_enabled
                and behavior.verify_before_promotion
                and behavior.future_audit
                else None
            )
            active_auditor = (
                ActiveMemoryAuditor(self.config.evolution)
                if self.config.evolution.active_audit_enabled
                and behavior.verify_before_promotion
                and behavior.use_memory
                and behavior.active_audit
                else None
            )
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
            supersession_events = 0
            memory_reactivations = 0
            future_audit_registered = 0
            future_audit_completed = 0
            future_audit_confirmed = 0
            future_audit_rolled_back = 0
            future_audit_expired = 0
            future_audit_outcomes: List[FutureAuditOutcome] = []
            audit_evidence: Dict[str, CandidateEvidence] = {}
            active_audit_eligible = 0
            active_audit_observations = 0
            active_audit_retirements = 0
            active_audit_correct_retirements = 0
            active_audit_false_retirements = 0
            active_audit_control_requests = 0
            active_audit_control_input_tokens = 0
            active_audit_control_output_tokens = 0
            active_audit_oracle_deltas: List[float] = []
            active_audit_retirement_latencies: List[int] = []
            stale_memory_opportunities = 0
            stale_active_memory_opportunities = 0
            active_memory_applications = 0
            stale_active_memory_applications = 0
            harmful_active_memory_exposure = 0
            stale_pair_opportunities: Dict[tuple[str, int], int] = {}
            stale_pair_first_active_index: Dict[tuple[str, int], int] = {}
            stale_pair_active_at_last_opportunity: Dict[tuple[str, int], bool] = {}
            active_audit_reactivations = 0
            active_audit_correct_reactivations = 0
            memory_reacquisitions = 0
            correct_memory_reacquisitions = 0

            def note_reacquisition(candidate: MemoryItem, activation_index: int) -> None:
                nonlocal memory_reacquisitions
                nonlocal correct_memory_reacquisitions
                if candidate.version <= 1:
                    return
                previous = store.get_memory(candidate.memory_id, version=candidate.version - 1)
                if previous is None or previous.status not in {
                    MemoryStatus.RETIRED,
                    MemoryStatus.SUPERSEDED,
                }:
                    return
                memory_reacquisitions += 1
                activation_sample = episodes[activation_index].sample
                activation_valid_tags = {
                    str(tag) for tag in activation_sample.metadata.get("valid_memory_tags", [])
                }
                activation_stale_tags = {
                    str(tag) for tag in activation_sample.metadata.get("stale_memory_tags", [])
                }
                correct = bool(activation_valid_tags.intersection(candidate.tags)) and not bool(
                    activation_stale_tags.intersection(candidate.tags)
                )
                if activation_stale_tags.intersection(candidate.tags):
                    activation_phase = int(activation_sample.metadata.get("phase_index", -1))
                    stale_pair_active_at_last_opportunity[
                        (candidate.memory_id, activation_phase)
                    ] = True
                correct_memory_reacquisitions += int(correct)
                store.record_event(
                    run_id,
                    "memory_reacquired",
                    f"{candidate.memory_id}@v{candidate.version}",
                    {
                        "activation_index": activation_index,
                        "oracle_correct_for_current_policy": correct,
                        "oracle_metrics_are_post_hoc_only": True,
                    },
                )

            def complete_future_audit(outcome: FutureAuditOutcome) -> None:
                nonlocal supersession_events
                nonlocal future_audit_completed
                nonlocal future_audit_confirmed
                nonlocal future_audit_rolled_back
                nonlocal future_audit_expired
                decision = outcome.decision
                decisions.append(decision)
                future_audit_outcomes.append(outcome)
                store.save_validation(run_id, decision)
                artifacts.append_decision(decision)
                current = store.get_memory(
                    outcome.audit.memory.memory_id,
                    version=outcome.audit.memory.version,
                )
                candidate = current or outcome.audit.memory
                evidence = audit_evidence.pop(candidate.memory_id, None)
                if outcome.completion_reason == "stream_end":
                    memory.reject(candidate)
                    future_audit_expired += 1
                    if evidence is not None:
                        candidate_pool.mark_rejected(evidence)
                elif decision.promote:
                    future_audit_completed += 1
                    activation = memory.activate(
                        candidate,
                        decision.result.mean_delta,
                        decision.result.ci_low,
                        decision.result.regression_rate,
                        apply_supersession=(self.config.evolution.conflict_supersession_enabled),
                    )
                    supersession_events += len(activation.superseded)
                    activation_index = decision.result.observation_end_index
                    if activation_index is not None:
                        note_reacquisition(activation.active, activation_index)
                    future_audit_confirmed += 1
                    if evidence is not None:
                        candidate_pool.mark_accepted(evidence)
                else:
                    future_audit_completed += 1
                    memory.reject(candidate)
                    future_audit_rolled_back += 1
                    if evidence is not None:
                        candidate_pool.mark_rejected(evidence)
                store.record_event(
                    run_id,
                    "future_counterfactual_audit",
                    decision.result.candidate_id,
                    {
                        "evidence_signature": outcome.audit.evidence_signature,
                        "decision": decision.model_dump(mode="json"),
                        "completion_reason": outcome.completion_reason,
                        "online_decision_uses": "learner_visible_feedback_only",
                        "oracle_metrics_are_post_hoc_only": True,
                    },
                )

            for index, sample in enumerate(samples):
                memory_promoted_this_episode = False
                memory_retired_this_episode = False
                active_before = memory.active()
                known_before = store.list_memories()
                stale_tags = {str(tag) for tag in sample.metadata.get("stale_memory_tags", [])}
                valid_tags = {str(tag) for tag in sample.metadata.get("valid_memory_tags", [])}
                phase_index = int(sample.metadata.get("phase_index", -1))
                for item in known_before:
                    if item.status not in {
                        MemoryStatus.ACTIVE,
                        MemoryStatus.SUPERSEDED,
                        MemoryStatus.RETIRED,
                    } or not stale_tags.intersection(item.tags):
                        continue
                    pair = (item.memory_id, phase_index)
                    stale_memory_opportunities += 1
                    stale_pair_opportunities[pair] = stale_pair_opportunities.get(pair, 0) + 1
                    stale_pair_active_at_last_opportunity[pair] = item.status == MemoryStatus.ACTIVE
                    if item.status == MemoryStatus.ACTIVE:
                        stale_active_memory_opportunities += 1
                        stale_pair_first_active_index.setdefault(pair, index)
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
                applied_ids = set(prediction.output.applied_memory_ids)
                applied_active = [
                    retrieved.item
                    for retrieved in prediction.retrieved
                    if retrieved.item.memory_id in applied_ids
                    and retrieved.item.status == MemoryStatus.ACTIVE
                ]
                active_memory_applications += len(applied_active)
                stale_applied = [
                    item for item in applied_active if stale_tags.intersection(item.tags)
                ]
                stale_active_memory_applications += len(stale_applied)
                if not score.success:
                    harmful_active_memory_exposure += len(stale_applied)
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
                    feedback_change_probability=assessment.change_probability,
                    feedback_source_regime=assessment.source_regime,
                    feedback_grace_observations=assessment.grace_observations,
                    selected_memory_ids=selected_ids,
                    policy_version=policy.version,
                    usage=prediction.usage,
                    shift=shift,
                )
                episodes.append(episode)
                store.save_episode(episode)
                artifacts.append_episode(episode)

                if (
                    future_auditor is not None
                    and assessment.trust >= self.config.evolution.min_feedback_trust_for_replay
                ):
                    for audit in future_auditor.pending_for(prediction.output.applied_memory_ids):
                        control_prediction = await agent.solve(
                            sample,
                            policy,
                            exclude_memory_ids=[audit.memory.memory_id],
                            use_memory=behavior.use_memory,
                        )
                        control_score = score_sample(
                            sample,
                            control_prediction.output.answer,
                        )
                        control_feedback = score_feedback_sample(
                            sample,
                            control_prediction.output.answer,
                        )
                        future_result = future_auditor.record(
                            audit,
                            index=index,
                            feedback_control=control_feedback.primary,
                            feedback_candidate=feedback_score.primary,
                            oracle_control=control_score.primary,
                            oracle_candidate=score.primary,
                            control_usage=control_prediction.usage,
                            candidate_usage=prediction.usage,
                            protected=bool(sample.metadata.get("protected")),
                        )
                        if future_result is not None:
                            complete_future_audit(future_result)

                if (
                    active_auditor is not None
                    and not self.frozen_audit
                    and not feedback_score.success
                    and assessment.trust
                    >= self.config.evolution.min_feedback_trust_for_active_audit
                ):
                    eligible_active = active_auditor.eligible(
                        applied_active,
                        prediction.output.applied_memory_ids,
                        episode_index=index,
                    )
                    active_audit_eligible += len(eligible_active)
                    for audited_memory in eligible_active[
                        : self.config.evolution.active_audit_max_per_episode
                    ]:
                        control_prediction = await agent.solve(
                            sample,
                            policy,
                            exclude_memory_versions=[
                                (audited_memory.memory_id, audited_memory.version)
                            ],
                            use_memory=behavior.use_memory,
                        )
                        control_score = score_sample(sample, control_prediction.output.answer)
                        control_feedback = score_feedback_sample(
                            sample,
                            control_prediction.output.answer,
                        )
                        audit_decision = active_auditor.observe(
                            audited_memory,
                            episode_index=index,
                            feedback_control=control_feedback.primary,
                            feedback_candidate=feedback_score.primary,
                        )
                        audit_outcome = memory.apply_active_audit(
                            audit_decision.memory_after,
                            retire=audit_decision.retire,
                            restore_predecessors=(
                                self.config.evolution.active_audit_restore_predecessors
                            ),
                        )
                        active_audit_observations += 1
                        active_audit_control_requests += 1
                        active_audit_control_input_tokens += control_prediction.usage.input_tokens
                        active_audit_control_output_tokens += control_prediction.usage.output_tokens
                        oracle_delta = score.primary - control_score.primary
                        active_audit_oracle_deltas.append(oracle_delta)
                        is_oracle_stale = bool(stale_tags.intersection(audited_memory.tags))
                        actually_retired = bool(audit_outcome.rolled_back)
                        store.record_event(
                            run_id,
                            "active_memory_causal_audit",
                            audit_decision.candidate_id,
                            {
                                "episode_index": index,
                                "feedback_control": audit_decision.feedback_control,
                                "feedback_candidate": audit_decision.feedback_candidate,
                                "learner_visible_delta": audit_decision.delta,
                                "causal_audit_count": (
                                    audit_decision.memory_after.causal_audit_count
                                ),
                                "causal_mean_delta": (
                                    audit_decision.memory_after.causal_mean_delta
                                ),
                                "retirement_decision": audit_decision.retire,
                                "retirement_applied": actually_retired,
                                "reason": audit_decision.reason,
                                "online_decision_uses": "learner_visible_feedback_only",
                                "oracle_metrics_are_post_hoc_only": True,
                                "oracle_delta": oracle_delta,
                                "oracle_stale_for_sample": is_oracle_stale,
                            },
                        )
                        if actually_retired:
                            retired = audit_outcome.rolled_back[0]
                            memory_retired_this_episode = True
                            rollbacks += 1
                            active_audit_retirements += 1
                            candidate_pool.mark_memory_retired(retired)
                            for predecessor_id in retired.supersedes_memory_ids:
                                predecessor = store.get_memory(predecessor_id)
                                if predecessor is not None:
                                    candidate_pool.mark_memory_retired(predecessor)
                            pair = (retired.memory_id, phase_index)
                            if is_oracle_stale:
                                active_audit_correct_retirements += 1
                                stale_pair_active_at_last_opportunity[pair] = False
                            else:
                                active_audit_false_retirements += 1
                            first_stale = stale_pair_first_active_index.get(pair)
                            if first_stale is not None:
                                active_audit_retirement_latencies.append(index - first_stale + 1)
                            store.record_event(
                                run_id,
                                "memory_rollback",
                                audit_decision.candidate_id,
                                {"reason": audit_decision.reason, "mechanism": "active_causal"},
                            )
                        for reactivated in audit_outcome.reactivated:
                            memory_reactivations += 1
                            active_audit_reactivations += 1
                            correct_reactivation = bool(
                                valid_tags.intersection(reactivated.tags)
                            ) and not bool(stale_tags.intersection(reactivated.tags))
                            if stale_tags.intersection(reactivated.tags):
                                stale_pair_active_at_last_opportunity[
                                    (reactivated.memory_id, phase_index)
                                ] = True
                            active_audit_correct_reactivations += int(correct_reactivation)
                            store.record_event(
                                run_id,
                                "memory_reactivated",
                                reactivated.memory_id,
                                {
                                    "reason": "causally retired successor",
                                    "oracle_correct_for_current_policy": correct_reactivation,
                                    "oracle_metrics_are_post_hoc_only": True,
                                },
                            )

                if (
                    not self.frozen_audit
                    and assessment.trust
                    >= self.config.evolution.min_feedback_trust_for_memory_update
                ):
                    credited_ids = prediction.output.applied_memory_ids or selected_ids
                    memory_outcome = memory.record_outcome(
                        credited_ids,
                        feedback_score.success,
                        policy,
                    )
                    for item in memory_outcome.rolled_back:
                        rollbacks += 1
                        payload = {"reason": "posterior utility below rollback threshold"}
                        store.record_event(
                            run_id,
                            "memory_rollback",
                            item.memory_id,
                            payload,
                        )
                    for item in memory_outcome.reactivated:
                        memory_reactivations += 1
                        store.record_event(
                            run_id,
                            "memory_reactivated",
                            item.memory_id,
                            {"reason": "superseding successor rolled back"},
                        )
                elif not self.frozen_audit:
                    feedback_quarantined += 1
                    store.record_event(
                        run_id,
                        "feedback_quarantined",
                        episode.episode_id,
                        {
                            "source": assessment.source,
                            "context": assessment.context,
                            "signal": assessment.signal,
                            "trust": assessment.trust,
                            "source_posterior_mean": assessment.source_posterior_mean,
                            "change_probability": assessment.change_probability,
                            "source_regime": assessment.source_regime,
                            "grace_observations": assessment.grace_observations,
                            "context_observations": assessment.context_observations,
                            "pending_observations": assessment.pending_observations,
                            "reason": assessment.reason,
                        },
                    )

                should_extract = (
                    feedback_eligible
                    and (
                        not feedback_score.success
                        or (
                            policy.learn_from_success_every > 0
                            and (index + 1) % policy.learn_from_success_every == 0
                        )
                    )
                    and not memory_retired_this_episode
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
                                activation = memory.activate(
                                    candidate,
                                    0.0,
                                    0.0,
                                    0.0,
                                    apply_supersession=(
                                        self.config.evolution.conflict_supersession_enabled
                                    ),
                                )
                                supersession_events += len(activation.superseded)
                                note_reacquisition(activation.active, index)
                                store.record_event(
                                    run_id,
                                    "memory_promoted_unverified",
                                    (f"{activation.active.memory_id}@v{activation.active.version}"),
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
                                if self.config.evolution.candidate_replay_since_first_evidence:
                                    regime_start = max(
                                        regime_start or 0,
                                        evidence.first_episode_index,
                                    )
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
                                            if future_auditor is not None:
                                                if future_auditor.is_pending(candidate.memory_id):
                                                    memory.reject(candidate)
                                                    candidate_pool.mark_rejected(evidence)
                                                    candidate_deferred += 1
                                                    store.record_event(
                                                        run_id,
                                                        "memory_probation_deferred",
                                                        (
                                                            f"{candidate.memory_id}"
                                                            f"@v{candidate.version}"
                                                        ),
                                                        {
                                                            "reason": (
                                                                "older_version_future_audit_pending"
                                                            ),
                                                            "candidate_signature": (
                                                                evidence.signature
                                                            ),
                                                        },
                                                    )
                                                else:
                                                    probationary = memory.probation(
                                                        candidate,
                                                        decision.result.mean_delta,
                                                        decision.result.ci_low,
                                                        decision.result.regression_rate,
                                                    )
                                                    future_auditor.register(
                                                        probationary,
                                                        evidence_signature=evidence.signature,
                                                        start_index=index,
                                                    )
                                                    audit_evidence[probationary.memory_id] = (
                                                        evidence
                                                    )
                                                    candidate_pool.mark_probation(evidence)
                                                    future_audit_registered += 1
                                                    memory_promoted_this_episode = True
                                                    store.record_event(
                                                        run_id,
                                                        "memory_probation_started",
                                                        (
                                                            f"{probationary.memory_id}"
                                                            f"@v{probationary.version}"
                                                        ),
                                                        {
                                                            "candidate_signature": (
                                                                evidence.signature
                                                            ),
                                                            "future_audit_min_observations": (
                                                                self.config.evolution.future_audit_min_observations
                                                            ),
                                                        },
                                                    )
                                            else:
                                                activation = memory.activate(
                                                    candidate,
                                                    decision.result.mean_delta,
                                                    decision.result.ci_low,
                                                    decision.result.regression_rate,
                                                    apply_supersession=(
                                                        self.config.evolution.conflict_supersession_enabled
                                                    ),
                                                )
                                                supersession_events += len(activation.superseded)
                                                note_reacquisition(activation.active, index)
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

            if future_auditor is not None:
                for outcome in future_auditor.finalize():
                    complete_future_audit(outcome)

            final_memories = tuple(memory.active())
            final_state_hash = state_fingerprint(policy, final_memories)
            if (
                self.frozen_audit
                and self.source_state_hash
                and final_state_hash != self.source_state_hash
            ):
                raise RuntimeError("frozen audit state changed during evaluation")

            replay_memory_decisions = [
                decision for decision in decisions if decision.result.candidate_type == "memory"
            ]
            future_memory_decisions = [
                decision
                for decision in decisions
                if decision.result.candidate_type == "memory_future_audit"
            ]
            policy_decisions = [
                decision for decision in decisions if decision.result.candidate_type == "policy"
            ]
            completed_future_audits = [
                outcome
                for outcome in future_audit_outcomes
                if outcome.completion_reason == "evidence"
            ]
            realized_gain_by_candidate = {
                outcome.decision.result.candidate_id: (
                    outcome.decision.result.oracle_mean_delta or 0.0
                )
                for outcome in completed_future_audits
            }
            replay_precision_inputs = None if self.frozen_audit else replay_memory_decisions
            realized_precision_inputs: List[PromotionDecision] = []
            realized_gains: List[float] = []
            if future_auditor is not None and replay_precision_inputs is not None:
                realized_precision_inputs = [
                    decision
                    for decision in replay_precision_inputs
                    if decision.result.candidate_id in realized_gain_by_candidate
                ]
                realized_gains = [
                    realized_gain_by_candidate[decision.result.candidate_id]
                    for decision in realized_precision_inputs
                ]
            metrics = compute_stream_metrics(
                episodes,
                recovery_fraction=self.config.evaluation.recovery_fraction,
                promotions=replay_precision_inputs,
            )
            replay_estimated_precision = (
                metrics.get("promotion_precision") if replay_memory_decisions else None
            )
            realized_precision = (
                promotion_precision(realized_precision_inputs, realized_gains)
                if realized_precision_inputs
                else None
            )
            if self.frozen_audit:
                metrics["promotion_precision"] = None
                metrics["promotion_precision_basis"] = "not_applicable"
                replay_estimated_precision = None
            elif future_auditor is not None:
                metrics["promotion_precision"] = realized_precision
                metrics["promotion_precision_basis"] = "future_counterfactual"
            elif replay_memory_decisions:
                metrics["promotion_precision"] = replay_estimated_precision
                metrics["promotion_precision_basis"] = "replay_estimated"
            else:
                metrics["promotion_precision"] = None
                metrics["promotion_precision_basis"] = "not_applicable"
            metrics["replay_estimated_promotion_precision"] = replay_estimated_precision
            metrics["realized_promotion_precision"] = realized_precision
            promoted_memories = sum(
                int(decision.promote)
                for decision in (
                    future_memory_decisions
                    if future_auditor is not None
                    else replay_memory_decisions
                )
            )
            promoted_policies = sum(int(decision.promote) for decision in policy_decisions)
            promoted = promoted_memories + promoted_policies
            rejected_memories = len(replay_memory_decisions) - promoted_memories
            rejected_policies = len(policy_decisions) - promoted_policies
            replay_promotions = [
                decision for decision in replay_memory_decisions if decision.promote
            ]
            realized_replay_promotions = [
                decision
                for decision in replay_promotions
                if decision.result.candidate_id in realized_gain_by_candidate
            ]
            audit_oracle_gains = [
                outcome.decision.result.oracle_mean_delta
                for outcome in completed_future_audits
                if outcome.decision.result.oracle_mean_delta is not None
            ]
            audit_latencies = [
                outcome.decision.result.observation_end_index - outcome.audit.start_index
                for outcome in completed_future_audits
                if outcome.decision.result.observation_end_index is not None
            ]
            confirmation_latencies = [
                outcome.decision.result.observation_end_index - outcome.audit.start_index
                for outcome in completed_future_audits
                if outcome.decision.promote
                and outcome.decision.result.observation_end_index is not None
            ]
            rollback_latencies = [
                outcome.decision.result.observation_end_index - outcome.audit.start_index
                for outcome in completed_future_audits
                if not outcome.decision.promote
                and outcome.decision.result.observation_end_index is not None
            ]
            harmful_exposure_latencies = [
                outcome.decision.result.observation_end_index - outcome.audit.start_index
                for outcome in completed_future_audits
                if outcome.decision.result.oracle_mean_delta is not None
                and outcome.decision.result.oracle_mean_delta < 0.0
                and outcome.decision.result.observation_end_index is not None
            ]
            rollback_oracle_gains = [
                outcome.decision.result.oracle_mean_delta
                for outcome in completed_future_audits
                if not outcome.decision.promote
                and outcome.decision.result.oracle_mean_delta is not None
            ]
            confirmation_observations = [
                outcome.decision.result.n
                for outcome in completed_future_audits
                if outcome.decision.promote
            ]
            rollback_observations = [
                outcome.decision.result.n
                for outcome in completed_future_audits
                if not outcome.decision.promote
            ]
            harmful_exposure_observations = [
                outcome.decision.result.n
                for outcome in completed_future_audits
                if outcome.decision.result.oracle_mean_delta is not None
                and outcome.decision.result.oracle_mean_delta < 0.0
            ]
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
                        "candidates_evaluated": (
                            len(replay_memory_decisions) + len(policy_decisions)
                        ),
                        "candidates_promoted": promoted,
                        "candidates_rejected": rejected_memories + rejected_policies,
                        "memory_candidates_evaluated": len(replay_memory_decisions),
                        "memory_candidates_promoted": promoted_memories,
                        "memory_candidates_rejected": rejected_memories,
                        "memory_replay_gates_passed": sum(
                            int(decision.promote) for decision in replay_memory_decisions
                        ),
                        "policy_candidates_evaluated": len(policy_decisions),
                        "policy_candidates_promoted": promoted_policies,
                        "policy_candidates_rejected": rejected_policies,
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
                        "memory_supersessions": supersession_events,
                        "memory_reactivations": memory_reactivations,
                        "final_active_memories": len(final_memories),
                        "final_policy_version": policy.version,
                    },
                    "feedback_trust_model": trust_model.snapshot(),
                    "future_audit": {
                        "enabled": future_auditor is not None,
                        "registered": future_audit_registered,
                        "completed": future_audit_completed,
                        "confirmed": future_audit_confirmed,
                        "rolled_back": future_audit_rolled_back,
                        "expired": future_audit_expired,
                        "realized_promotion_coverage": (
                            len(realized_replay_promotions) / len(replay_promotions)
                            if replay_promotions
                            else 0.0
                        ),
                        "mean_oracle_delta": (
                            statistics.fmean(audit_oracle_gains) if audit_oracle_gains else None
                        ),
                        "harmful_promotion_rate": (
                            statistics.fmean(float(value < 0.0) for value in audit_oracle_gains)
                            if audit_oracle_gains
                            else 0.0
                        ),
                        "false_rollback_rate": (
                            statistics.fmean(float(value >= 0.0) for value in rollback_oracle_gains)
                            if rollback_oracle_gains
                            else 0.0
                        ),
                        "mean_audit_latency": (
                            statistics.fmean(audit_latencies) if audit_latencies else None
                        ),
                        "mean_confirmation_latency": (
                            statistics.fmean(confirmation_latencies)
                            if confirmation_latencies
                            else None
                        ),
                        "mean_rollback_latency": (
                            statistics.fmean(rollback_latencies) if rollback_latencies else None
                        ),
                        "mean_harmful_exposure_latency": (
                            statistics.fmean(harmful_exposure_latencies)
                            if harmful_exposure_latencies
                            else None
                        ),
                        "mean_confirmation_observations": (
                            statistics.fmean(confirmation_observations)
                            if confirmation_observations
                            else None
                        ),
                        "mean_rollback_observations": (
                            statistics.fmean(rollback_observations)
                            if rollback_observations
                            else None
                        ),
                        "mean_harmful_exposure_observations": (
                            statistics.fmean(harmful_exposure_observations)
                            if harmful_exposure_observations
                            else None
                        ),
                    },
                    "active_memory_governance": {
                        "enabled": active_auditor is not None,
                        "eligible_active_memory_failures": active_audit_eligible,
                        "counterfactual_audit_observations": active_audit_observations,
                        "counterfactual_audit_coverage": (
                            active_audit_observations / active_audit_eligible
                            if active_audit_eligible
                            else 0.0
                        ),
                        "audit_budget_utilization": (
                            active_audit_observations
                            / (len(episodes) * self.config.evolution.active_audit_max_per_episode)
                            if episodes and active_auditor is not None
                            else 0.0
                        ),
                        "harmful_active_memory_exposure_n": (harmful_active_memory_exposure),
                        "harmful_active_memory_exposure_rate": (
                            harmful_active_memory_exposure / active_memory_applications
                            if active_memory_applications
                            else 0.0
                        ),
                        "stale_active_memory_application_n": (stale_active_memory_applications),
                        "stale_application_harm_rate": (
                            harmful_active_memory_exposure / stale_active_memory_applications
                            if stale_active_memory_applications
                            else 0.0
                        ),
                        "stale_memory_opportunities": stale_memory_opportunities,
                        "stale_active_memory_opportunities": (stale_active_memory_opportunities),
                        "stale_memory_retention_rate": (
                            stale_active_memory_opportunities / stale_memory_opportunities
                            if stale_memory_opportunities
                            else 0.0
                        ),
                        "causal_retirements": active_audit_retirements,
                        "selective_forgetting_precision": (
                            active_audit_correct_retirements / active_audit_retirements
                            if active_audit_retirements
                            else 0.0
                        ),
                        "selective_forgetting_recall": (
                            len(
                                {
                                    pair
                                    for pair, active in (
                                        stale_pair_active_at_last_opportunity.items()
                                    )
                                    if not active
                                }
                                & {
                                    pair
                                    for pair, count in stale_pair_opportunities.items()
                                    if count >= self.config.evolution.active_audit_min_observations
                                }
                            )
                            / len(
                                {
                                    pair
                                    for pair, count in stale_pair_opportunities.items()
                                    if count >= self.config.evolution.active_audit_min_observations
                                }
                            )
                            if any(
                                count >= self.config.evolution.active_audit_min_observations
                                for count in stale_pair_opportunities.values()
                            )
                            else 0.0
                        ),
                        "false_retirement_rate": (
                            active_audit_false_retirements / active_audit_retirements
                            if active_audit_retirements
                            else 0.0
                        ),
                        "mean_retirement_latency": (
                            statistics.fmean(active_audit_retirement_latencies)
                            if active_audit_retirement_latencies
                            else None
                        ),
                        "reactivations": active_audit_reactivations,
                        "correct_reactivation_rate": (
                            active_audit_correct_reactivations / active_audit_reactivations
                            if active_audit_reactivations
                            else None
                        ),
                        "reacquisitions": memory_reacquisitions,
                        "correct_reacquisition_rate": (
                            correct_memory_reacquisitions / memory_reacquisitions
                            if memory_reacquisitions
                            else None
                        ),
                        "mean_post_hoc_oracle_delta": (
                            statistics.fmean(active_audit_oracle_deltas)
                            if active_audit_oracle_deltas
                            else None
                        ),
                        "control_requests": active_audit_control_requests,
                        "control_input_tokens": active_audit_control_input_tokens,
                        "control_output_tokens": active_audit_control_output_tokens,
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
                            else (
                                "future counterfactual oracle attribution; online decisions use "
                                "learner-visible feedback only; precision covers completed "
                                "evidence audits and excludes stream-end expirations"
                                if future_auditor is not None
                                else (
                                    "replay-estimated; held-out audit required for claims"
                                    if replay_memory_decisions
                                    else "not applicable: no replay validation decisions"
                                )
                            )
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
