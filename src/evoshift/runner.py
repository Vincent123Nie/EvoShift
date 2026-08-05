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
    ActiveAuditDecision,
    ActiveMemoryAuditor,
    CandidateEvidencePool,
    CausalCircuitBreaker,
    DormantMemoryRevival,
    ExperienceCritic,
    FeedbackTrustModel,
    FutureAuditOutcome,
    FutureCounterfactualAuditor,
    PageHinkleyShiftDetector,
    PendingCausalCanary,
    PendingRetirement,
    RetirementProbation,
    dormant_candidate_keys,
    order_semantic_dormant_candidates,
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


@dataclass(frozen=True)
class ActiveAuditApplication:
    retired: bool
    pending_retirement: PendingRetirement | None = None


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
        expected_state_hash: str = "",
        audit_variant: str = "",
        excluded_memory_id: str = "",
        excluded_memory_version: Optional[int] = None,
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
        self.expected_state_hash = expected_state_hash
        self.audit_variant = audit_variant
        self.excluded_memory_id = excluded_memory_id
        self.excluded_memory_version = excluded_memory_version
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
            audit_variant=self.audit_variant,
            excluded_memory_id=self.excluded_memory_id,
            excluded_memory_version=self.excluded_memory_version,
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
            memory = MemoryManager(
                store,
                status_indexed_revival=(self.config.evolution.dormant_revival_status_index_enabled),
            )
            behavior = behavior_for(self.config.algorithm)
            agent = MemoryAgent(client, self.config.provider, memory)
            critic = ExperienceCritic(client, self.config.provider, self.config.evolution)
            trust_model = FeedbackTrustModel(
                self.config.evolution,
                dynamic_enabled=(
                    self.config.evolution.dynamic_feedback_trust_enabled
                    and behavior.dynamic_feedback_trust
                    and not self.frozen_audit
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
                min_trusted_observations=(self.config.evolution.candidate_min_trusted_observations),
                min_new_observations=(self.config.evolution.candidate_min_new_observations),
                cooldown_episodes=self.config.evolution.candidate_cooldown_episodes,
                shadow_eprocess_enabled=self.config.evolution.shadow_eprocess_enabled,
                shadow_eprocess_null_match_probability=(
                    self.config.evolution.shadow_eprocess_null_match_probability
                ),
                shadow_eprocess_alternative_match_probability=(
                    self.config.evolution.shadow_eprocess_alternative_match_probability
                ),
                shadow_eprocess_alpha=self.config.evolution.shadow_eprocess_alpha,
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
            circuit_breaker = (
                CausalCircuitBreaker(self.config.evolution)
                if active_auditor is not None
                and self.config.evolution.active_audit_circuit_breaker_enabled
                and not self.frozen_audit
                else None
            )
            dormant_revival = (
                DormantMemoryRevival(self.config.evolution)
                if self.config.evolution.dormant_revival_enabled
                and behavior.verify_before_promotion
                and behavior.use_memory
                and not self.frozen_audit
                else None
            )
            retirement_probation = (
                RetirementProbation(self.config.evolution)
                if self.config.evolution.active_audit_retirement_probation_enabled
                and active_auditor is not None
                and not self.frozen_audit
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
            shadow_failure_extractions = 0
            shadow_candidate_observations = 0
            shadow_candidate_replay_attempts = 0
            shadow_only_candidate_replay_attempts = 0
            shadow_candidate_probations = 0
            shadow_candidate_activations = 0
            shadow_candidate_rejections = 0
            shadow_candidate_expirations = 0
            trusted_candidate_shadow_cooldown_bypasses = 0
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
            active_audit_early_retirements = 0
            active_audit_early_correct_retirements = 0
            active_audit_early_false_retirements = 0
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
            circuit_control_requests = 0
            circuit_control_input_tokens = 0
            circuit_control_output_tokens = 0
            circuit_correct_confirmations = 0
            circuit_false_confirmations = 0
            circuit_unconfirmed_persistent_transitions = 0
            circuit_oracle_intervention_deltas: List[float] = []
            revival_control_requests = 0
            revival_control_input_tokens = 0
            revival_control_output_tokens = 0
            revival_correct_confirmations = 0
            revival_false_confirmations = 0
            revival_unconfirmed_persistent_transitions = 0
            revival_oracle_intervention_deltas: List[float] = []
            retirement_probation_correct_confirmations = 0
            retirement_probation_false_confirmations = 0
            retirement_probation_unconfirmed_persistent_transitions = 0
            retirement_probation_oracle_deltas: List[float] = []
            retirement_probation_control_requests = 0
            retirement_probation_control_input_tokens = 0
            retirement_probation_control_output_tokens = 0
            retirement_circuit_canaries: Dict[tuple[str, str, str, int], PendingCausalCanary] = {}
            retired_memory_indices: Dict[tuple[str, int], int] = {}
            reactivated_memory_indices: Dict[tuple[str, int], int] = {}
            reactivation_grace_audit_suppressions = 0
            reactivation_grace_protected_rollbacks = 0

            def note_reacquisition(candidate: MemoryItem, activation_index: int) -> None:
                nonlocal memory_reacquisitions
                nonlocal correct_memory_reacquisitions
                retired_memory_indices.pop((candidate.memory_id, candidate.version), None)
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

            def note_reactivation(item: MemoryItem, episode_index: int) -> None:
                if self.config.evolution.active_audit_reactivation_grace_episodes <= 0:
                    return
                key = (item.memory_id, item.version)
                reactivated_memory_indices[key] = episode_index
                if circuit_breaker is not None:
                    circuit_breaker.invalidate_memory_versions(
                        [key],
                        reason="memory version was reactivated into a new lifecycle",
                    )

            def reactivation_grace(
                item: MemoryItem,
                episode_index: int,
            ) -> tuple[int, int] | None:
                grace_episodes = self.config.evolution.active_audit_reactivation_grace_episodes
                reactivated_index = reactivated_memory_indices.get((item.memory_id, item.version))
                if grace_episodes <= 0 or reactivated_index is None:
                    return None
                age = episode_index - reactivated_index
                if age < 0 or age > grace_episodes:
                    return None
                return reactivated_index, age

            def complete_future_audit(outcome: FutureAuditOutcome) -> None:
                nonlocal supersession_events
                nonlocal future_audit_completed
                nonlocal future_audit_confirmed
                nonlocal future_audit_rolled_back
                nonlocal future_audit_expired
                nonlocal shadow_candidate_activations
                nonlocal shadow_candidate_rejections
                nonlocal shadow_candidate_expirations
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
                shadow_derived = bool(evidence and evidence.has_shadow_evidence)
                if outcome.completion_reason == "stream_end":
                    memory.reject(candidate)
                    future_audit_expired += 1
                    shadow_candidate_expirations += int(shadow_derived)
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
                    shadow_candidate_activations += int(shadow_derived)
                    if evidence is not None:
                        candidate_pool.mark_accepted(evidence)
                else:
                    future_audit_completed += 1
                    memory.reject(candidate)
                    future_audit_rolled_back += 1
                    shadow_candidate_rejections += int(shadow_derived)
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
                        "shadow_derived": shadow_derived,
                        "candidate_trust": (
                            {
                                "trusted_observations": evidence.trusted_observation_count,
                                "shadow_observations": evidence.shadow_observation_count,
                                "mean": evidence.mean_trust,
                                "min": evidence.min_trust,
                                "max": evidence.max_trust,
                            }
                            if evidence is not None
                            else None
                        ),
                        "online_decision_uses": "learner_visible_feedback_only",
                        "oracle_metrics_are_post_hoc_only": True,
                    },
                )

            def record_extra_audit_request(
                prediction: Any,
                *,
                circuit_request: bool,
            ) -> None:
                nonlocal active_audit_control_requests
                nonlocal active_audit_control_input_tokens
                nonlocal active_audit_control_output_tokens
                nonlocal circuit_control_requests
                nonlocal circuit_control_input_tokens
                nonlocal circuit_control_output_tokens
                active_audit_control_requests += 1
                active_audit_control_input_tokens += prediction.usage.input_tokens
                active_audit_control_output_tokens += prediction.usage.output_tokens
                if circuit_request:
                    circuit_control_requests += 1
                    circuit_control_input_tokens += prediction.usage.input_tokens
                    circuit_control_output_tokens += prediction.usage.output_tokens

            def record_revival_request(prediction: Any) -> None:
                nonlocal revival_control_requests
                nonlocal revival_control_input_tokens
                nonlocal revival_control_output_tokens
                revival_control_requests += 1
                revival_control_input_tokens += prediction.usage.input_tokens
                revival_control_output_tokens += prediction.usage.output_tokens

            def record_retirement_probation_request(prediction: Any) -> None:
                nonlocal retirement_probation_control_requests
                nonlocal retirement_probation_control_input_tokens
                nonlocal retirement_probation_control_output_tokens
                retirement_probation_control_requests += 1
                retirement_probation_control_input_tokens += prediction.usage.input_tokens
                retirement_probation_control_output_tokens += prediction.usage.output_tokens
                record_extra_audit_request(prediction, circuit_request=False)

            def finalize_retirement(
                retired: MemoryItem,
                *,
                episode_index: int,
                phase_index: int,
                is_oracle_stale: bool,
                early_retirement: bool,
                reason: str,
                mechanism: str,
                entity_id: str,
                count_active_audit: bool = True,
            ) -> None:
                nonlocal rollbacks
                nonlocal active_audit_retirements
                nonlocal active_audit_correct_retirements
                nonlocal active_audit_false_retirements
                nonlocal active_audit_early_retirements
                nonlocal active_audit_early_correct_retirements
                nonlocal active_audit_early_false_retirements

                retired_memory_indices[(retired.memory_id, retired.version)] = episode_index
                rollbacks += 1
                candidate_pool.mark_memory_retired(retired)
                for predecessor_id in retired.supersedes_memory_ids:
                    predecessor = store.get_memory(predecessor_id)
                    if predecessor is not None:
                        candidate_pool.mark_memory_retired(predecessor)
                if count_active_audit:
                    active_audit_retirements += 1
                    active_audit_early_retirements += int(early_retirement)
                    pair = (retired.memory_id, phase_index)
                    if is_oracle_stale:
                        active_audit_correct_retirements += 1
                        stale_pair_active_at_last_opportunity[pair] = False
                    else:
                        active_audit_false_retirements += 1
                    if early_retirement:
                        if is_oracle_stale:
                            active_audit_early_correct_retirements += 1
                        else:
                            active_audit_early_false_retirements += 1
                    first_stale = stale_pair_first_active_index.get(pair)
                    if first_stale is not None:
                        active_audit_retirement_latencies.append(episode_index - first_stale + 1)
                store.record_event(
                    run_id,
                    "memory_rollback",
                    entity_id,
                    {"reason": reason, "mechanism": mechanism},
                )

            def record_causal_audit(
                decision: ActiveAuditDecision,
                *,
                episode_index: int,
                phase_index: int,
                stale_tags: set[str],
                valid_tags: set[str],
                oracle_control: float,
                oracle_candidate: float,
                additional_prediction: Any,
                mechanism: str,
                source: str,
                context: str,
                persist_non_retirement: bool = True,
                circuit_request: bool = False,
            ) -> ActiveAuditApplication:
                nonlocal rollbacks
                nonlocal memory_reactivations
                nonlocal active_audit_observations
                nonlocal active_audit_retirements
                nonlocal active_audit_correct_retirements
                nonlocal active_audit_false_retirements
                nonlocal active_audit_early_retirements
                nonlocal active_audit_early_correct_retirements
                nonlocal active_audit_early_false_retirements
                nonlocal active_audit_reactivations
                nonlocal active_audit_correct_reactivations

                audit_outcome = None
                if decision.retire or persist_non_retirement:
                    audit_outcome = memory.apply_active_audit(
                        decision.memory_after,
                        retire=decision.retire,
                        restore_predecessors=(
                            self.config.evolution.active_audit_restore_predecessors
                        ),
                    )
                active_audit_observations += 1
                record_extra_audit_request(
                    additional_prediction,
                    circuit_request=circuit_request,
                )
                oracle_delta = float(oracle_candidate) - float(oracle_control)
                active_audit_oracle_deltas.append(oracle_delta)
                is_oracle_stale = bool(stale_tags.intersection(decision.memory_before.tags))
                actually_retired = bool(audit_outcome and audit_outcome.rolled_back)
                pending_retirement: PendingRetirement | None = None
                early_retirement = decision.reason.startswith("early retire:")
                if actually_retired and audit_outcome is not None:
                    retired = audit_outcome.rolled_back[0]
                    retired_memory_indices[(retired.memory_id, retired.version)] = episode_index
                    if retirement_probation is not None:
                        pending_retirement = retirement_probation.register(
                            source=source,
                            context=context,
                            memory=retired,
                            active_snapshot=decision.memory_before,
                            restored_predecessors=audit_outcome.reactivated,
                            episode_index=episode_index,
                            mechanism=mechanism,
                            phase_index=phase_index,
                            oracle_stale=is_oracle_stale,
                            early_retirement=early_retirement,
                        )
                    if pending_retirement is None:
                        finalize_retirement(
                            retired,
                            episode_index=episode_index,
                            phase_index=phase_index,
                            is_oracle_stale=is_oracle_stale,
                            early_retirement=early_retirement,
                            reason=decision.reason,
                            mechanism=mechanism,
                            entity_id=decision.candidate_id,
                        )
                    else:
                        candidate_pool.mark_memory_retired(retired)
                        store.record_event(
                            run_id,
                            "memory_retirement_probation_started",
                            f"{retired.memory_id}@v{retired.version}",
                            {
                                "episode_index": episode_index,
                                "source": source,
                                "context": context,
                                "expires_after_index": (pending_retirement.expires_after_index),
                                "mechanism": mechanism,
                                "reason": decision.reason,
                                "oracle_stale_for_sample": is_oracle_stale,
                                "early_retirement": early_retirement,
                                "persistent_state_changed": True,
                                "online_decision_uses": ("learner_visible_feedback_only"),
                            },
                        )
                store.record_event(
                    run_id,
                    "active_memory_causal_audit",
                    decision.candidate_id,
                    {
                        "episode_index": episode_index,
                        "feedback_control": decision.feedback_control,
                        "feedback_candidate": decision.feedback_candidate,
                        "learner_visible_delta": decision.delta,
                        "causal_audit_count": decision.memory_after.causal_audit_count,
                        "causal_mean_delta": decision.memory_after.causal_mean_delta,
                        "retirement_decision": decision.retire,
                        "early_retirement": decision.reason.startswith("early retire:"),
                        "retirement_applied": actually_retired,
                        "retirement_probation_pending": pending_retirement is not None,
                        "persist_non_retirement": persist_non_retirement,
                        "mechanism": mechanism,
                        "reason": decision.reason,
                        "online_decision_uses": "learner_visible_feedback_only",
                        "oracle_metrics_are_post_hoc_only": True,
                        "oracle_delta": oracle_delta,
                        "oracle_stale_for_sample": is_oracle_stale,
                    },
                )
                if audit_outcome is not None:
                    for reactivated in audit_outcome.reactivated:
                        note_reactivation(reactivated, episode_index)
                        retired_memory_indices.pop(
                            (reactivated.memory_id, reactivated.version),
                            None,
                        )
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
                                "mechanism": mechanism,
                                "oracle_correct_for_current_policy": correct_reactivation,
                                "oracle_metrics_are_post_hoc_only": True,
                            },
                        )
                return ActiveAuditApplication(actually_retired, pending_retirement)

            def revert_provisional_canary(
                pending: PendingCausalCanary,
                *,
                episode_index: int,
                reason: str,
            ) -> bool:
                if active_auditor is None:
                    return False
                memory_id, version = pending.memory_key
                current = store.get_memory(memory_id, version=version)
                reverted = False
                if (
                    current is not None
                    and current.status == MemoryStatus.ACTIVE
                    and active_auditor.observation_is_present(current, pending.observation)
                ):
                    restored = active_auditor.revert_observation(current, pending.observation)
                    memory.apply_active_audit(restored, retire=False)
                    reverted = True
                store.record_event(
                    run_id,
                    "causal_circuit_breaker_reverted",
                    f"{memory_id}@v{version}",
                    {
                        "episode_index": episode_index,
                        "registered_index": pending.registered_index,
                        "source": pending.source,
                        "context": pending.context,
                        "reason": reason,
                        "provisional_ledger_reverted": reverted,
                        "online_decision_uses": "learner_visible_feedback_only",
                    },
                )
                return reverted

            def drain_circuit_invalidations(episode_index: int) -> None:
                if circuit_breaker is None:
                    return
                for invalidated_canary, invalidation_reason in circuit_breaker.drain_invalidated():
                    reverted = revert_provisional_canary(
                        invalidated_canary,
                        episode_index=episode_index,
                        reason=invalidation_reason,
                    )
                    store.record_event(
                        run_id,
                        "causal_circuit_breaker_invalidated",
                        (f"{invalidated_canary.memory_key[0]}@v{invalidated_canary.memory_key[1]}"),
                        {
                            "episode_index": episode_index,
                            "registered_index": invalidated_canary.registered_index,
                            "source": invalidated_canary.source,
                            "context": invalidated_canary.context,
                            "control_memory_key": invalidated_canary.control_memory_key,
                            "reason": invalidation_reason,
                            "provisional_ledger_reverted": reverted,
                            "persistent_state_changed": False,
                            "online_decision_uses": "learner_visible_feedback_only",
                        },
                    )

            def rollback_retirement_probation(
                pending: PendingRetirement,
                *,
                episode_index: int,
                reason: str,
            ) -> MemoryItem | None:
                nonlocal retirement_probation_unconfirmed_persistent_transitions
                restored = memory.rollback_retirement(
                    pending.memory,
                    pending.restored_predecessors,
                    pending.active_snapshot,
                )
                if restored is not None:
                    retired_memory_indices.pop(pending.memory_key, None)
                    for predecessor in pending.restored_predecessors:
                        reactivated_memory_indices.pop(
                            (predecessor.memory_id, predecessor.version),
                            None,
                        )
                    note_reactivation(restored, episode_index)
                    candidate_pool.seed_accepted([restored])
                    store.record_event(
                        run_id,
                        "memory_retirement_probation_rollback",
                        f"{pending.memory.memory_id}@v{pending.memory.version}",
                        {
                            "episode_index": episode_index,
                            "registered_index": pending.registered_index,
                            "source": pending.source,
                            "context": pending.context,
                            "reason": reason,
                            "restored_predecessors": [
                                [item.memory_id, item.version]
                                for item in pending.restored_predecessors
                            ],
                            "persistent_state_changed": True,
                            "online_decision_uses": ("learner_visible_feedback_only"),
                        },
                    )
                else:
                    retirement_probation_unconfirmed_persistent_transitions += 1
                    store.record_event(
                        run_id,
                        "memory_retirement_probation_invalidated",
                        f"{pending.memory.memory_id}@v{pending.memory.version}",
                        {
                            "episode_index": episode_index,
                            "registered_index": pending.registered_index,
                            "reason": "exact retired version was no longer available",
                            "requested_reason": reason,
                            "persistent_state_changed": False,
                            "online_decision_uses": ("learner_visible_feedback_only"),
                        },
                    )
                return restored

            def resolve_retirement_circuit(
                pending: PendingRetirement,
                *,
                committed: bool,
                episode_index: int,
                oracle_stale: bool,
                reason: str,
            ) -> None:
                nonlocal circuit_correct_confirmations
                nonlocal circuit_false_confirmations
                canary = retirement_circuit_canaries.pop(pending.transaction_key, None)
                if canary is None or circuit_breaker is None:
                    return
                reverted = False
                if committed:
                    circuit_breaker.resolve(canary, confirmed=True)
                    circuit_correct_confirmations += int(oracle_stale)
                    circuit_false_confirmations += int(not oracle_stale)
                else:
                    reverted = revert_provisional_canary(
                        canary,
                        episode_index=episode_index,
                        reason=reason,
                    )
                    circuit_breaker.resolve(canary, confirmed=False)
                store.record_event(
                    run_id,
                    "causal_circuit_breaker_retirement_resolved",
                    f"{canary.memory_key[0]}@v{canary.memory_key[1]}",
                    {
                        "episode_index": episode_index,
                        "retirement_transaction_key": pending.transaction_key,
                        "committed": committed,
                        "reason": reason,
                        "provisional_ledger_reverted": reverted,
                        "online_decision_uses": "learner_visible_feedback_only",
                        "oracle_metrics_are_post_hoc_only": True,
                        "oracle_stale_at_resolution": oracle_stale,
                    },
                )

            def register_posterior_retirement(
                retired: MemoryItem,
                active_snapshot: MemoryItem,
                predecessors: Sequence[MemoryItem],
                *,
                episode_index: int,
                phase_index: int,
                oracle_stale: bool,
            ) -> PendingRetirement | None:
                """Put a posterior-utility retirement through the same reversible lane."""

                retired_memory_indices[(retired.memory_id, retired.version)] = episode_index
                pending: PendingRetirement | None = None
                if retirement_probation is not None:
                    pending = retirement_probation.register(
                        source=assessment.source,
                        context=assessment.context,
                        memory=retired,
                        active_snapshot=active_snapshot,
                        restored_predecessors=predecessors,
                        episode_index=episode_index,
                        mechanism="posterior_utility",
                        phase_index=phase_index,
                        oracle_stale=oracle_stale,
                        early_retirement=False,
                    )
                candidate_pool.mark_memory_retired(retired)
                if pending is not None:
                    store.record_event(
                        run_id,
                        "memory_retirement_probation_started",
                        f"{retired.memory_id}@v{retired.version}",
                        {
                            "episode_index": episode_index,
                            "source": assessment.source,
                            "context": assessment.context,
                            "expires_after_index": pending.expires_after_index,
                            "mechanism": "posterior_utility",
                            "reason": "posterior utility below rollback threshold",
                            "oracle_stale_for_sample": oracle_stale,
                            "early_retirement": False,
                            "persistent_state_changed": True,
                            "online_decision_uses": "learner_visible_feedback_only",
                        },
                    )
                return pending

            for index, sample in enumerate(samples):
                memory_promoted_this_episode = False
                memory_retired_this_episode = False
                if circuit_breaker is not None:
                    for expired_canary in circuit_breaker.expire(index):
                        revert_provisional_canary(
                            expired_canary,
                            episode_index=index,
                            reason="ttl_expired",
                        )
                if dormant_revival is not None:
                    for expired_revival in dormant_revival.expire(index):
                        store.record_event(
                            run_id,
                            "dormant_memory_revival_expired",
                            (f"{expired_revival.memory_key[0]}@v{expired_revival.memory_key[1]}"),
                            {
                                "episode_index": index,
                                "registered_index": expired_revival.registered_index,
                                "source": expired_revival.source,
                                "context": expired_revival.context,
                                "persistent_state_changed": False,
                            },
                        )
                if retirement_probation is not None:
                    for expired_retirement in retirement_probation.expire(index):
                        rollback_retirement_probation(
                            expired_retirement,
                            episode_index=index,
                            reason="retirement probation TTL expired",
                        )
                        resolve_retirement_circuit(
                            expired_retirement,
                            committed=False,
                            episode_index=index,
                            oracle_stale=expired_retirement.oracle_stale,
                            reason="retirement probation TTL expired",
                        )
                active_before = memory.active()
                active_before_keys = {(item.memory_id, item.version) for item in active_before}
                grace_episodes = self.config.evolution.active_audit_reactivation_grace_episodes
                for memory_key, reactivated_index in tuple(reactivated_memory_indices.items()):
                    if (
                        memory_key not in active_before_keys
                        or index - reactivated_index > grace_episodes
                    ):
                        reactivated_memory_indices.pop(memory_key, None)
                known_before = store.list_memories()
                semantic_dormant_index = bool(
                    dormant_revival is not None and dormant_revival.semantic_indexed
                )
                retired_before = (
                    store.list_memories([MemoryStatus.RETIRED])
                    if (
                        self.config.evolution.dormant_revival_status_index_enabled
                        or semantic_dormant_index
                    )
                    else known_before
                )
                superseded_before = store.list_memories([MemoryStatus.SUPERSEDED])
                occupied_memory_scopes = {item.scope for item in active_before}
                eligible_dormant_keys = dormant_candidate_keys(
                    retired_before,
                    retired_memory_indices,
                    semantic_indexed=semantic_dormant_index,
                )
                if retirement_probation is not None:
                    eligible_dormant_keys.difference_update(
                        retirement_probation.pending_memory_keys()
                    )
                observable_source, observable_context = trust_model.observable_key(sample)
                circuit_intervention = (
                    circuit_breaker.match(
                        source=observable_source,
                        context=observable_context,
                        episode_index=index,
                        active_memories=active_before,
                        lineage_control_versions={
                            (item.memory_id, item.version) for item in superseded_before
                        },
                    )
                    if circuit_breaker is not None
                    else None
                )
                drain_circuit_invalidations(index)
                retirement_intervention = (
                    retirement_probation.match(
                        source=observable_source,
                        context=observable_context,
                        episode_index=index,
                        retired_memory_versions={
                            (item.memory_id, item.version)
                            for item in retired_before
                            if item.status == MemoryStatus.RETIRED
                            and (item.memory_id, item.version) in retired_memory_indices
                        },
                    )
                    if retirement_probation is not None and circuit_intervention is None
                    else None
                )
                if retirement_probation is not None:
                    for (
                        invalidated_retirement,
                        invalidation_reason,
                    ) in retirement_probation.drain_invalidated():
                        resolve_retirement_circuit(
                            invalidated_retirement,
                            committed=False,
                            episode_index=index,
                            oracle_stale=invalidated_retirement.oracle_stale,
                            reason=invalidation_reason,
                        )
                        store.record_event(
                            run_id,
                            "memory_retirement_probation_invalidated",
                            (
                                f"{invalidated_retirement.memory.memory_id}"
                                f"@v{invalidated_retirement.memory.version}"
                            ),
                            {
                                "episode_index": index,
                                "registered_index": invalidated_retirement.registered_index,
                                "reason": invalidation_reason,
                                "persistent_state_changed": False,
                                "online_decision_uses": "learner_visible_memory_lifecycle_only",
                            },
                        )
                revival_intervention = (
                    dormant_revival.match(
                        source=observable_source,
                        context=observable_context,
                        episode_index=index,
                        retired_latest_versions={
                            (item.memory_id, item.version)
                            for item in retired_before
                            if item.status == MemoryStatus.RETIRED
                            and item.scope not in occupied_memory_scopes
                            and (item.memory_id, item.version) in eligible_dormant_keys
                        },
                    )
                    if (
                        dormant_revival is not None
                        and circuit_intervention is None
                        and retirement_intervention is None
                    )
                    else None
                )
                suppress_failure_extraction_this_episode = (
                    circuit_intervention is not None
                    or retirement_intervention is not None
                    or revival_intervention is not None
                )
                if circuit_intervention is not None:
                    store.record_event(
                        run_id,
                        "causal_circuit_breaker_intervention",
                        (
                            f"{circuit_intervention.memory_key[0]}"
                            f"@v{circuit_intervention.memory_key[1]}"
                        ),
                        {
                            "episode_index": index,
                            "registered_index": circuit_intervention.registered_index,
                            "source": circuit_intervention.source,
                            "context": circuit_intervention.context,
                            "action": (
                                "temporarily_replace_with_exact_direct_predecessor"
                                if circuit_intervention.control_memory is not None
                                else "temporarily_exclude_exact_memory_version"
                            ),
                            "control_memory_key": (circuit_intervention.control_memory_key),
                            "online_decision_uses": "learner_visible_feedback_only",
                        },
                    )
                if revival_intervention is not None:
                    store.record_event(
                        run_id,
                        "dormant_memory_revival_intervention",
                        (
                            f"{revival_intervention.memory_key[0]}"
                            f"@v{revival_intervention.memory_key[1]}"
                        ),
                        {
                            "episode_index": index,
                            "registered_index": revival_intervention.registered_index,
                            "source": revival_intervention.source,
                            "context": revival_intervention.context,
                            "action": "temporarily_force_exact_retired_memory_version",
                            "online_decision_uses": "learner_visible_feedback_only",
                        },
                    )
                if retirement_intervention is not None:
                    store.record_event(
                        run_id,
                        "memory_retirement_probation_intervention",
                        (
                            f"{retirement_intervention.memory_key[0]}"
                            f"@v{retirement_intervention.memory_key[1]}"
                        ),
                        {
                            "episode_index": index,
                            "registered_index": retirement_intervention.registered_index,
                            "source": retirement_intervention.source,
                            "context": retirement_intervention.context,
                            "action": "evaluate_exact_old_memory_counterfactual",
                            "online_decision_uses": "learner_visible_feedback_only",
                        },
                    )
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
                    exclude_memory_versions=(
                        [circuit_intervention.memory_key]
                        if circuit_intervention is not None
                        else None
                    ),
                    extra_memories=(
                        [circuit_intervention.control_memory]
                        if circuit_intervention is not None
                        and circuit_intervention.control_memory is not None
                        else (
                            [revival_intervention.memory]
                            if revival_intervention is not None
                            else None
                        )
                    ),
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
                assessment = trust_model.assess(sample, episode_index=index)
                feedback_eligible = (
                    assessment.trust >= self.config.evolution.min_feedback_trust_for_candidate
                )
                shadow_candidate_eligible = (
                    self.config.evolution.shadow_candidate_enabled
                    and assessment.trust
                    >= self.config.evolution.min_feedback_trust_for_shadow_candidate
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

                if retirement_intervention is not None:
                    assert retirement_probation is not None
                    retirement_memory_id, retirement_memory_version = (
                        retirement_intervention.memory_key
                    )
                    old_memory_prediction = await agent.solve(
                        sample,
                        policy,
                        exclude_memory_versions=[
                            (item.memory_id, item.version)
                            for item in retirement_intervention.restored_predecessors
                        ],
                        extra_memories=[retirement_intervention.memory],
                        use_memory=behavior.use_memory,
                    )
                    record_retirement_probation_request(old_memory_prediction)
                    old_memory_score = score_sample(
                        sample,
                        old_memory_prediction.output.answer,
                    )
                    old_memory_feedback = score_feedback_sample(
                        sample,
                        old_memory_prediction.output.answer,
                    )
                    retirement_probation_oracle_deltas.append(
                        score.primary - old_memory_score.primary
                    )
                    old_memory_applied = (
                        retirement_memory_id in old_memory_prediction.output.applied_memory_ids
                    )
                    feedback_delta = feedback_score.primary - old_memory_feedback.primary
                    evidence_decision = retirement_probation.observe(
                        retirement_intervention,
                        episode_index=index,
                        trust=assessment.trust,
                        old_memory_applied=old_memory_applied,
                        delta=feedback_delta,
                    )
                    resolved_retirement = evidence_decision.pending
                    confirmed = evidence_decision.outcome == "confirm"
                    vetoed = evidence_decision.outcome == "veto"
                    deferred = evidence_decision.outcome == "defer"
                    restored_retirement: MemoryItem | None = None
                    confirmation_oracle_stale = bool(
                        stale_tags.intersection(resolved_retirement.memory.tags)
                    )
                    if confirmed:
                        finalize_retirement(
                            resolved_retirement.memory,
                            episode_index=index,
                            phase_index=phase_index,
                            is_oracle_stale=confirmation_oracle_stale,
                            early_retirement=resolved_retirement.early_retirement,
                            reason="confirmed retirement probation",
                            mechanism=resolved_retirement.mechanism,
                            entity_id=(f"{retirement_memory_id}@v{retirement_memory_version}"),
                            count_active_audit=(
                                resolved_retirement.mechanism != "posterior_utility"
                            ),
                        )
                        retirement_probation_correct_confirmations += int(confirmation_oracle_stale)
                        retirement_probation_false_confirmations += int(
                            not confirmation_oracle_stale
                        )
                        resolve_retirement_circuit(
                            resolved_retirement,
                            committed=True,
                            episode_index=index,
                            oracle_stale=confirmation_oracle_stale,
                            reason="sequential retirement evidence confirmed",
                        )
                    elif vetoed:
                        restored_retirement = rollback_retirement_probation(
                            resolved_retirement,
                            episode_index=index,
                            reason="sequential old-memory evidence vetoed retirement",
                        )
                        memory_reactivations += int(restored_retirement is not None)
                        resolve_retirement_circuit(
                            resolved_retirement,
                            committed=False,
                            episode_index=index,
                            oracle_stale=confirmation_oracle_stale,
                            reason="sequential retirement evidence vetoed",
                        )
                    if confirmed:
                        resolution_reason = (
                            "confirm: sequential current-over-old evidence reached threshold"
                        )
                    elif vetoed:
                        resolution_reason = (
                            "cancel: sequential old-over-current evidence reached threshold"
                        )
                    else:
                        resolution_reason = f"defer: {evidence_decision.reason}"
                    store.record_event(
                        run_id,
                        (
                            "memory_retirement_probation_deferred"
                            if deferred
                            else "memory_retirement_probation_resolved"
                        ),
                        f"{retirement_memory_id}@v{retirement_memory_version}",
                        {
                            "episode_index": index,
                            "registered_index": retirement_intervention.registered_index,
                            "source": retirement_intervention.source,
                            "context": retirement_intervention.context,
                            "feedback_current": feedback_score.primary,
                            "feedback_old_memory": old_memory_feedback.primary,
                            "feedback_delta": feedback_delta,
                            "old_memory_applied": old_memory_applied,
                            "feedback_trust": assessment.trust,
                            "evidence_recorded": evidence_decision.evidence_recorded,
                            "confirmation_indices": list(resolved_retirement.confirmation_indices),
                            "veto_indices": list(resolved_retirement.veto_indices),
                            "confirmed": confirmed,
                            "vetoed": vetoed,
                            "deferred": deferred,
                            "persistent_state_changed": confirmed
                            or restored_retirement is not None,
                            "reason": resolution_reason,
                            "online_decision_uses": "learner_visible_feedback_only",
                            "oracle_metrics_are_post_hoc_only": True,
                            "oracle_intervention_delta": (score.primary - old_memory_score.primary),
                            "oracle_stale_at_registration": (retirement_intervention.oracle_stale),
                            "oracle_stale_at_resolution": confirmation_oracle_stale,
                        },
                    )

                if revival_intervention is not None:
                    assert dormant_revival is not None
                    revival_memory_id, revival_memory_version = revival_intervention.memory_key
                    memory_off_prediction = await agent.solve(
                        sample,
                        policy,
                        exclude_memory_versions=[(revival_memory_id, revival_memory_version)],
                        use_memory=behavior.use_memory,
                    )
                    record_revival_request(memory_off_prediction)
                    memory_off_score = score_sample(
                        sample,
                        memory_off_prediction.output.answer,
                    )
                    memory_off_feedback = score_feedback_sample(
                        sample,
                        memory_off_prediction.output.answer,
                    )
                    revival_oracle_intervention_deltas.append(
                        score.primary - memory_off_score.primary
                    )
                    treatment_applied = revival_memory_id in prediction.output.applied_memory_ids
                    confirmation_eligible = (
                        treatment_applied
                        and assessment.trust
                        >= self.config.evolution.min_feedback_trust_for_active_audit
                        and dormant_revival.qualifies(
                            feedback_score.primary - memory_off_feedback.primary
                        )
                    )
                    reactivated = (
                        memory.reactivate_retired(revival_intervention.memory)
                        if confirmation_eligible
                        else None
                    )
                    confirmed = reactivated is not None
                    dormant_revival.resolve(
                        revival_intervention,
                        confirmed=confirmed,
                    )
                    resolution_reason = (
                        "confirm: two strong learner-visible memory-on gains"
                        if confirmed
                        else (
                            "cancel: forced retired memory was not explicitly applied"
                            if not treatment_applied
                            else (
                                "cancel: feedback trust remained below audit threshold"
                                if assessment.trust
                                < self.config.evolution.min_feedback_trust_for_active_audit
                                else "cancel: second paired gain below revival threshold"
                            )
                        )
                    )
                    if confirmed and reactivated is not None:
                        note_reactivation(reactivated, index)
                        retired_memory_indices.pop(
                            (reactivated.memory_id, reactivated.version),
                            None,
                        )
                        memory_reactivations += 1
                        memory_promoted_this_episode = True
                        candidate_pool.seed_accepted([reactivated])
                        correct_revival = bool(
                            valid_tags.intersection(reactivated.tags)
                        ) and not bool(stale_tags.intersection(reactivated.tags))
                        revival_correct_confirmations += int(correct_revival)
                        revival_false_confirmations += int(not correct_revival)
                        store.record_event(
                            run_id,
                            "memory_reactivated",
                            reactivated.memory_id,
                            {
                                "reason": "confirmed dormant-memory recurrence",
                                "mechanism": "cooldown_dormant_revival",
                                "oracle_correct_for_current_policy": correct_revival,
                                "oracle_metrics_are_post_hoc_only": True,
                            },
                        )
                    store.record_event(
                        run_id,
                        "dormant_memory_revival_resolved",
                        f"{revival_memory_id}@v{revival_memory_version}",
                        {
                            "episode_index": index,
                            "registered_index": revival_intervention.registered_index,
                            "source": revival_intervention.source,
                            "context": revival_intervention.context,
                            "feedback_memory_off": memory_off_feedback.primary,
                            "feedback_memory_on": feedback_score.primary,
                            "first_learner_visible_delta": revival_intervention.delta,
                            "second_learner_visible_delta": (
                                feedback_score.primary - memory_off_feedback.primary
                            ),
                            "forced_memory_applied": treatment_applied,
                            "feedback_trust": assessment.trust,
                            "confirmed": confirmed,
                            "reason": resolution_reason,
                            "persistent_state_changed": confirmed,
                            "online_decision_uses": "learner_visible_feedback_only",
                            "oracle_metrics_are_post_hoc_only": True,
                            "oracle_intervention_delta": (score.primary - memory_off_score.primary),
                        },
                    )

                if (
                    active_auditor is not None
                    and not self.frozen_audit
                    and retirement_intervention is None
                ):
                    if circuit_intervention is not None:
                        assert circuit_breaker is not None
                        active_audit_eligible += 1
                        memory_id, memory_version = circuit_intervention.memory_key
                        audited_memory = store.get_memory(memory_id, version=memory_version)
                        if (
                            audited_memory is not None
                            and audited_memory.status == MemoryStatus.ACTIVE
                        ):
                            forced_on_prediction = await agent.solve(
                                sample,
                                policy,
                                extra_memories=[audited_memory],
                                use_memory=behavior.use_memory,
                            )
                            forced_on_score = score_sample(
                                sample,
                                forced_on_prediction.output.answer,
                            )
                            forced_on_feedback = score_feedback_sample(
                                sample,
                                forced_on_prediction.output.answer,
                            )
                            circuit_oracle_intervention_deltas.append(
                                score.primary - forced_on_score.primary
                            )
                            treatment_applied = (
                                memory_id in forced_on_prediction.output.applied_memory_ids
                            )
                            confirmation_eligible = (
                                treatment_applied
                                and assessment.trust
                                >= self.config.evolution.min_feedback_trust_for_active_audit
                            )
                            confirmed = False
                            circuit_retirement_pending = False
                            resolution_reason = ""
                            if confirmation_eligible:
                                confirmation_decision = active_auditor.observe(
                                    audited_memory,
                                    episode_index=index,
                                    feedback_control=feedback_score.primary,
                                    feedback_candidate=forced_on_feedback.primary,
                                    shift_detected=shift.detected,
                                )
                                circuit_application = record_causal_audit(
                                    confirmation_decision,
                                    episode_index=index,
                                    phase_index=phase_index,
                                    stale_tags=stale_tags,
                                    valid_tags=valid_tags,
                                    oracle_control=score.primary,
                                    oracle_candidate=forced_on_score.primary,
                                    additional_prediction=forced_on_prediction,
                                    mechanism="recurrence_circuit_confirmation",
                                    source=assessment.source,
                                    context=assessment.context,
                                    persist_non_retirement=False,
                                    circuit_request=True,
                                )
                                confirmed = bool(
                                    circuit_application.retired
                                    and circuit_application.pending_retirement is None
                                )
                                if circuit_application.pending_retirement is not None:
                                    circuit_retirement_pending = True
                                    retirement_circuit_canaries[
                                        circuit_application.pending_retirement.transaction_key
                                    ] = circuit_intervention
                                    resolution_reason = (
                                        "pending: destructive retirement awaits sequential evidence"
                                    )
                                else:
                                    resolution_reason = confirmation_decision.reason
                            else:
                                record_extra_audit_request(
                                    forced_on_prediction,
                                    circuit_request=True,
                                )
                                resolution_reason = (
                                    "cancel: forced memory was not explicitly applied"
                                    if not treatment_applied
                                    else "cancel: feedback trust remained below audit threshold"
                                )
                            if confirmed:
                                memory_retired_this_episode = True
                                circuit_breaker.resolve(
                                    circuit_intervention,
                                    confirmed=True,
                                )
                                is_oracle_stale = bool(stale_tags.intersection(audited_memory.tags))
                                circuit_correct_confirmations += int(is_oracle_stale)
                                circuit_false_confirmations += int(not is_oracle_stale)
                            elif circuit_retirement_pending:
                                memory_retired_this_episode = True
                            else:
                                revert_provisional_canary(
                                    circuit_intervention,
                                    episode_index=index,
                                    reason=resolution_reason,
                                )
                                circuit_breaker.resolve(
                                    circuit_intervention,
                                    confirmed=False,
                                )
                            store.record_event(
                                run_id,
                                (
                                    "causal_circuit_breaker_resolution_pending"
                                    if circuit_retirement_pending
                                    else "causal_circuit_breaker_resolved"
                                ),
                                f"{memory_id}@v{memory_version}",
                                {
                                    "episode_index": index,
                                    "registered_index": (circuit_intervention.registered_index),
                                    "source": circuit_intervention.source,
                                    "context": circuit_intervention.context,
                                    "control_kind": (
                                        "direct_predecessor"
                                        if circuit_intervention.control_memory is not None
                                        else "memory_off"
                                    ),
                                    "control_memory_key": (circuit_intervention.control_memory_key),
                                    "feedback_control": feedback_score.primary,
                                    "feedback_forced_memory_on": (forced_on_feedback.primary),
                                    "forced_memory_applied": treatment_applied,
                                    "feedback_trust": assessment.trust,
                                    "confirmed": confirmed,
                                    "retirement_probation_pending": (circuit_retirement_pending),
                                    "reason": resolution_reason,
                                    "online_decision_uses": ("learner_visible_feedback_only"),
                                    "oracle_metrics_are_post_hoc_only": True,
                                    "oracle_intervention_delta": (
                                        score.primary - forced_on_score.primary
                                    ),
                                },
                            )
                        else:
                            circuit_breaker.resolve(circuit_intervention, confirmed=False)
                    elif not feedback_score.success:
                        ordinary_audit_eligible = (
                            assessment.trust
                            >= self.config.evolution.min_feedback_trust_for_active_audit
                        )
                        circuit_probe_eligible = (
                            circuit_breaker is not None
                            and circuit_breaker.trust_is_probe_eligible(assessment.trust)
                        )
                        eligible_active = (
                            active_auditor.eligible(
                                applied_active,
                                prediction.output.applied_memory_ids,
                                episode_index=index,
                            )
                            if ordinary_audit_eligible or circuit_probe_eligible
                            else []
                        )
                        grace_suppressed = [
                            item
                            for item in eligible_active
                            if reactivation_grace(item, index) is not None
                        ]
                        for suppressed_memory in grace_suppressed:
                            grace_state = reactivation_grace(suppressed_memory, index)
                            assert grace_state is not None
                            reactivated_index, reactivation_age = grace_state
                            reactivation_grace_audit_suppressions += 1
                            store.record_event(
                                run_id,
                                "active_memory_audit_suppressed",
                                (f"{suppressed_memory.memory_id}@v{suppressed_memory.version}"),
                                {
                                    "episode_index": index,
                                    "reactivated_index": reactivated_index,
                                    "reactivation_age": reactivation_age,
                                    "grace_episodes": (
                                        self.config.evolution.active_audit_reactivation_grace_episodes
                                    ),
                                    "audit_lane": (
                                        "ordinary" if ordinary_audit_eligible else "causal_circuit"
                                    ),
                                    "feedback_trust": assessment.trust,
                                    "feedback_trust_reason": assessment.reason,
                                    "reason": "exact reactivated version is inside grace",
                                    "persistent_state_changed": False,
                                    "online_decision_uses": (
                                        "learner_visible_memory_lifecycle_only"
                                    ),
                                },
                            )
                        grace_suppressed_keys = {
                            (item.memory_id, item.version) for item in grace_suppressed
                        }
                        eligible_active = [
                            item
                            for item in eligible_active
                            if (item.memory_id, item.version) not in grace_suppressed_keys
                        ]
                        active_audit_eligible += len(eligible_active)
                        selected_active = eligible_active[
                            : self.config.evolution.active_audit_max_per_episode
                        ]
                        if ordinary_audit_eligible:
                            for audited_memory in selected_active:
                                control_prediction = await agent.solve(
                                    sample,
                                    policy,
                                    exclude_memory_versions=[
                                        (audited_memory.memory_id, audited_memory.version)
                                    ],
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
                                audit_decision = active_auditor.observe(
                                    audited_memory,
                                    episode_index=index,
                                    feedback_control=control_feedback.primary,
                                    feedback_candidate=feedback_score.primary,
                                    shift_detected=shift.detected,
                                )
                                audit_application = record_causal_audit(
                                    audit_decision,
                                    episode_index=index,
                                    phase_index=phase_index,
                                    stale_tags=stale_tags,
                                    valid_tags=valid_tags,
                                    oracle_control=control_score.primary,
                                    oracle_candidate=score.primary,
                                    additional_prediction=control_prediction,
                                    mechanism="active_causal",
                                    source=assessment.source,
                                    context=assessment.context,
                                )
                                memory_retired_this_episode = (
                                    audit_application.retired or memory_retired_this_episode
                                )
                        elif circuit_probe_eligible and circuit_breaker is not None:
                            for audited_memory in selected_active:
                                lineage_control = None
                                if self.config.evolution.active_audit_lineage_control_enabled:
                                    ranked_predecessors = memory.rank_memories(
                                        sample.prompt,
                                        memory.direct_superseded_predecessors(audited_memory),
                                        policy,
                                        domain=sample.domain,
                                    )
                                    if ranked_predecessors:
                                        lineage_control = ranked_predecessors[0].item
                                control_prediction = await agent.solve(
                                    sample,
                                    policy,
                                    exclude_memory_versions=[
                                        (audited_memory.memory_id, audited_memory.version)
                                    ],
                                    extra_memories=(
                                        [lineage_control] if lineage_control is not None else None
                                    ),
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
                                circuit_breaker.note_probe(
                                    used_lineage_control=lineage_control is not None
                                )
                                provisional_decision = active_auditor.observe(
                                    audited_memory,
                                    episode_index=index,
                                    feedback_control=control_feedback.primary,
                                    feedback_candidate=feedback_score.primary,
                                    shift_detected=shift.detected,
                                    retirement_enabled=False,
                                )
                                pending = circuit_breaker.register(
                                    source=assessment.source,
                                    context=assessment.context,
                                    observation=provisional_decision,
                                    control_memory=lineage_control,
                                )
                                if pending is not None:
                                    provisional_application = record_causal_audit(
                                        provisional_decision,
                                        episode_index=index,
                                        phase_index=phase_index,
                                        stale_tags=stale_tags,
                                        valid_tags=valid_tags,
                                        oracle_control=control_score.primary,
                                        oracle_candidate=score.primary,
                                        additional_prediction=control_prediction,
                                        mechanism="recurrence_circuit_registration",
                                        source=assessment.source,
                                        context=assessment.context,
                                        circuit_request=True,
                                    )
                                    circuit_unconfirmed_persistent_transitions += int(
                                        provisional_application.retired
                                    )
                                    store.record_event(
                                        run_id,
                                        "causal_circuit_breaker_registered",
                                        provisional_decision.candidate_id,
                                        {
                                            "episode_index": index,
                                            "expires_after_index": (pending.expires_after_index),
                                            "source": pending.source,
                                            "context": pending.context,
                                            "control_kind": (
                                                "direct_predecessor"
                                                if pending.control_memory is not None
                                                else "memory_off"
                                            ),
                                            "control_memory_key": pending.control_memory_key,
                                            "learner_visible_delta": (provisional_decision.delta),
                                            "retirement_disabled": True,
                                            "online_decision_uses": (
                                                "learner_visible_feedback_only"
                                            ),
                                        },
                                    )
                                else:
                                    record_extra_audit_request(
                                        control_prediction,
                                        circuit_request=True,
                                    )
                                    store.record_event(
                                        run_id,
                                        "causal_circuit_breaker_probe",
                                        provisional_decision.candidate_id,
                                        {
                                            "episode_index": index,
                                            "source": assessment.source,
                                            "context": assessment.context,
                                            "control_kind": (
                                                "direct_predecessor"
                                                if lineage_control is not None
                                                else "memory_off"
                                            ),
                                            "control_memory_key": (
                                                (
                                                    lineage_control.memory_id,
                                                    lineage_control.version,
                                                )
                                                if lineage_control is not None
                                                else None
                                            ),
                                            "learner_visible_delta": (provisional_decision.delta),
                                            "registered": False,
                                            "reason": "causal delta above circuit threshold",
                                            "online_decision_uses": (
                                                "learner_visible_feedback_only"
                                            ),
                                        },
                                    )

                if (
                    dormant_revival is not None
                    and circuit_intervention is None
                    and revival_intervention is None
                    and not feedback_score.success
                    and not applied_active
                    and dormant_revival.trust_is_probe_eligible(assessment.trust)
                ):
                    eligible_dormant = [
                        item
                        for item in retired_before
                        if item.status == MemoryStatus.RETIRED
                        and item.scope not in occupied_memory_scopes
                        and (item.memory_id, item.version) in eligible_dormant_keys
                        and (item.memory_id, item.version) in retired_memory_indices
                        and dormant_revival.retired_long_enough(
                            retired_index=retired_memory_indices[(item.memory_id, item.version)],
                            episode_index=index,
                        )
                    ]
                    ranked_dormant = memory.rank_memories(
                        sample.prompt,
                        eligible_dormant,
                        policy,
                        domain=sample.domain,
                    )
                    if semantic_dormant_index:
                        # Retrieval relevance/utility remains the primary
                        # ordering. Retirement recency is only a deterministic
                        # tie-break, so an older but more applicable rule cannot
                        # be hidden by a newer card in the same scope.
                        ranked_dormant = order_semantic_dormant_candidates(
                            ranked_dormant,
                            retired_memory_indices,
                        )
                    for retrieved_dormant in ranked_dormant[:1]:
                        dormant_memory = retrieved_dormant.item
                        forced_on_prediction = await agent.solve(
                            sample,
                            policy,
                            extra_memories=[dormant_memory],
                            use_memory=behavior.use_memory,
                        )
                        record_revival_request(forced_on_prediction)
                        forced_on_score = score_sample(
                            sample,
                            forced_on_prediction.output.answer,
                        )
                        forced_on_feedback = score_feedback_sample(
                            sample,
                            forced_on_prediction.output.answer,
                        )
                        dormant_revival.note_probe()
                        treatment_applied = (
                            dormant_memory.memory_id
                            in forced_on_prediction.output.applied_memory_ids
                        )
                        pending_revival = (
                            dormant_revival.register(
                                source=assessment.source,
                                context=assessment.context,
                                memory=dormant_memory,
                                episode_index=index,
                                feedback_off=feedback_score.primary,
                                feedback_on=forced_on_feedback.primary,
                            )
                            if treatment_applied
                            else None
                        )
                        store.record_event(
                            run_id,
                            (
                                "dormant_memory_revival_registered"
                                if pending_revival is not None
                                else "dormant_memory_revival_probe"
                            ),
                            f"{dormant_memory.memory_id}@v{dormant_memory.version}",
                            {
                                "episode_index": index,
                                "source": assessment.source,
                                "context": assessment.context,
                                "candidate_status": dormant_memory.status.value,
                                "status_indexed_candidate_view": (
                                    self.config.evolution.dormant_revival_status_index_enabled
                                ),
                                "semantic_candidate_view": semantic_dormant_index,
                                "selected_retired_key": [
                                    dormant_memory.memory_id,
                                    dormant_memory.version,
                                ],
                                "retired_index": retired_memory_indices[
                                    (dormant_memory.memory_id, dormant_memory.version)
                                ],
                                "retired_age": (
                                    index
                                    - retired_memory_indices[
                                        (dormant_memory.memory_id, dormant_memory.version)
                                    ]
                                ),
                                "feedback_memory_off": feedback_score.primary,
                                "feedback_memory_on": forced_on_feedback.primary,
                                "learner_visible_delta": (
                                    forced_on_feedback.primary - feedback_score.primary
                                ),
                                "forced_memory_applied": treatment_applied,
                                "registered": pending_revival is not None,
                                "expires_after_index": (
                                    pending_revival.expires_after_index
                                    if pending_revival is not None
                                    else None
                                ),
                                "persistent_state_changed": False,
                                "online_decision_uses": ("learner_visible_feedback_only"),
                                "oracle_metrics_are_post_hoc_only": True,
                                "oracle_probe_delta": (forced_on_score.primary - score.primary),
                            },
                        )

                if (
                    not self.frozen_audit
                    and assessment.trust
                    >= self.config.evolution.min_feedback_trust_for_memory_update
                ):
                    credited_ids = prediction.output.applied_memory_ids or selected_ids
                    outcome_protected_versions = {
                        (item.memory_id, item.version)
                        for item in memory.active()
                        if reactivation_grace(item, index) is not None
                    }
                    memory_outcome = memory.record_outcome(
                        credited_ids,
                        feedback_score.success,
                        policy,
                        retirement_protected_versions=outcome_protected_versions,
                    )
                    for item, predecessors, active_snapshot in zip(
                        memory_outcome.rolled_back,
                        memory_outcome.retirement_predecessors,
                        memory_outcome.retirement_snapshots,
                    ):
                        memory_retired_this_episode = True
                        pending_posterior = register_posterior_retirement(
                            item,
                            active_snapshot,
                            predecessors,
                            episode_index=index,
                            phase_index=phase_index,
                            oracle_stale=bool(stale_tags.intersection(item.tags)),
                        )
                        for reactivated in predecessors:
                            note_reactivation(reactivated, index)
                            retired_memory_indices.pop(
                                (reactivated.memory_id, reactivated.version),
                                None,
                            )
                            memory_reactivations += 1
                            candidate_pool.seed_accepted([reactivated])
                            store.record_event(
                                run_id,
                                "memory_reactivated",
                                reactivated.memory_id,
                                {
                                    "reason": "posterior-retired successor rolled back",
                                    "mechanism": "posterior_utility",
                                    "provisional_retirement": pending_posterior is not None,
                                },
                            )
                        if pending_posterior is not None:
                            continue
                        rollbacks += 1
                        store.record_event(
                            run_id,
                            "memory_rollback",
                            item.memory_id,
                            {"reason": "posterior utility below rollback threshold"},
                        )
                    for item in memory_outcome.retirement_protected:
                        grace_state = reactivation_grace(item, index)
                        assert grace_state is not None
                        reactivated_index, reactivation_age = grace_state
                        reactivation_grace_protected_rollbacks += 1
                        store.record_event(
                            run_id,
                            "memory_rollback_suppressed",
                            f"{item.memory_id}@v{item.version}",
                            {
                                "episode_index": index,
                                "reactivated_index": reactivated_index,
                                "reactivation_age": reactivation_age,
                                "grace_episodes": (
                                    self.config.evolution.active_audit_reactivation_grace_episodes
                                ),
                                "reason": ("posterior rollback blocked inside reactivation grace"),
                                "persistent_state_changed": False,
                                "online_decision_uses": ("learner_visible_memory_lifecycle_only"),
                            },
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
                            "context_observations": assessment.context_observations,
                            "pending_observations": assessment.pending_observations,
                            "reason": assessment.reason,
                        },
                    )

                should_extract = (
                    (feedback_eligible or shadow_candidate_eligible)
                    and (
                        not feedback_score.success
                        or (
                            policy.learn_from_success_every > 0
                            and (index + 1) % policy.learn_from_success_every == 0
                        )
                    )
                    and not memory_retired_this_episode
                    and not suppress_failure_extraction_this_episode
                )
                if (
                    self.config.evolution.enabled
                    and not self.frozen_audit
                    and behavior.generate_experience
                    and should_extract
                ):
                    evidence_lane = "trusted" if feedback_eligible else "shadow"
                    shadow_failure_extractions += int(evidence_lane == "shadow")
                    failure = await critic.analyze(episode, memory.active(), shift)
                    failures.append(failure)
                    artifacts.append_failure(failure)
                    store.record_event(
                        run_id,
                        "failure_attribution",
                        failure.failure_id,
                        {
                            **failure.model_dump(mode="json"),
                            "evidence_lane": evidence_lane,
                            "feedback_trust": assessment.trust,
                            "feedback_trust_reason": assessment.reason,
                        },
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
                                trust=assessment.trust,
                                trusted=feedback_eligible,
                            )
                            shadow_candidate_observations += int(not feedback_eligible)
                            shadow_cooldown_active = (
                                feedback_eligible
                                and candidate_pool.shadow_cooldown_would_block(evidence, index)
                            )
                            ready, readiness_reason = candidate_pool.readiness(
                                evidence,
                                index,
                                trusted=feedback_eligible,
                            )
                            if ready and shadow_cooldown_active:
                                trusted_candidate_shadow_cooldown_bypasses += 1
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
                                        "last_trusted_validation_observation_count": (
                                            evidence.last_trusted_validation_observation_count
                                        ),
                                        "last_shadow_validation_observation_count": (
                                            evidence.last_shadow_validation_observation_count
                                        ),
                                        "trusted_observation_count": (
                                            evidence.trusted_observation_count
                                        ),
                                        "shadow_observation_count": (
                                            evidence.shadow_observation_count
                                        ),
                                        "mean_evidence_trust": evidence.mean_trust,
                                        "evidence_lane": evidence_lane,
                                        "shadow_e_value": evidence.shadow_e_value,
                                        "shadow_eprocess_ready": (evidence.shadow_eprocess_ready),
                                        "shadow_eprocess_threshold": (
                                            candidate_pool.shadow_eprocess_threshold
                                        ),
                                    },
                                )
                            else:
                                regime_start = regime_starts.get(sample.domain)
                                if self.config.evolution.candidate_replay_since_first_evidence:
                                    lane_first_evidence = (
                                        evidence.first_trusted_episode_index
                                        if feedback_eligible
                                        else evidence.first_shadow_episode_index
                                    )
                                    regime_start = max(
                                        regime_start or 0,
                                        lane_first_evidence
                                        if lane_first_evidence is not None
                                        else evidence.first_episode_index,
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
                                            "evidence_lane": evidence_lane,
                                            "shadow_e_value": evidence.shadow_e_value,
                                            "shadow_eprocess_ready": (
                                                evidence.shadow_eprocess_ready
                                            ),
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
                                            "evidence_lane": evidence_lane,
                                            "shadow_e_value": evidence.shadow_e_value,
                                            "shadow_eprocess_ready": (
                                                evidence.shadow_eprocess_ready
                                            ),
                                        },
                                    )
                                    candidate_pool.mark_validated(
                                        evidence,
                                        index,
                                        trusted=feedback_eligible,
                                    )
                                    if candidate.status != MemoryStatus.REJECTED:
                                        candidate_replay_attempts += 1
                                        shadow_candidate_replay_attempts += int(
                                            evidence.has_shadow_evidence
                                        )
                                        shadow_only_candidate_replay_attempts += int(
                                            not feedback_eligible
                                        )
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
                                                    shadow_candidate_rejections += int(
                                                        evidence.has_shadow_evidence
                                                    )
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
                                                    shadow_candidate_probations += int(
                                                        evidence.has_shadow_evidence
                                                    )
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
                                                            "shadow_derived": (
                                                                evidence.has_shadow_evidence
                                                            ),
                                                            "trusted_observation_count": (
                                                                evidence.trusted_observation_count
                                                            ),
                                                            "shadow_observation_count": (
                                                                evidence.shadow_observation_count
                                                            ),
                                                            "mean_evidence_trust": (
                                                                evidence.mean_trust
                                                            ),
                                                            "evidence_lane": evidence_lane,
                                                            "shadow_e_value": (
                                                                evidence.shadow_e_value
                                                            ),
                                                            "shadow_eprocess_ready": (
                                                                evidence.shadow_eprocess_ready
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
                                            shadow_candidate_rejections += int(
                                                evidence.has_shadow_evidence
                                            )
                                    else:
                                        shadow_candidate_rejections += int(
                                            evidence.has_shadow_evidence
                                        )

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

            if retirement_probation is not None:
                for pending_retirement in retirement_probation.expire_all():
                    rollback_retirement_probation(
                        pending_retirement,
                        episode_index=len(samples),
                        reason="stream_end",
                    )
                    resolve_retirement_circuit(
                        pending_retirement,
                        committed=False,
                        episode_index=len(samples),
                        oracle_stale=pending_retirement.oracle_stale,
                        reason="retirement probation stream end",
                    )

            if circuit_breaker is not None:
                drain_circuit_invalidations(len(samples))
                for expired_canary in circuit_breaker.expire_all():
                    revert_provisional_canary(
                        expired_canary,
                        episode_index=len(samples),
                        reason="stream_end",
                    )

            if dormant_revival is not None:
                for expired_revival in dormant_revival.expire_all():
                    store.record_event(
                        run_id,
                        "dormant_memory_revival_expired",
                        (f"{expired_revival.memory_key[0]}@v{expired_revival.memory_key[1]}"),
                        {
                            "episode_index": len(samples),
                            "registered_index": expired_revival.registered_index,
                            "source": expired_revival.source,
                            "context": expired_revival.context,
                            "reason": "stream_end",
                            "persistent_state_changed": False,
                        },
                    )

            if future_auditor is not None:
                for outcome in future_auditor.finalize():
                    complete_future_audit(outcome)

            final_memories = tuple(memory.active())
            final_state_hash = state_fingerprint(policy, final_memories)
            expected_state_hash = self.expected_state_hash or self.source_state_hash
            if (
                self.frozen_audit
                and expected_state_hash
                and final_state_hash != expected_state_hash
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
                        "shadow_failure_extractions": shadow_failure_extractions,
                        "shadow_candidate_observations": shadow_candidate_observations,
                        "shadow_candidate_replay_attempts": (shadow_candidate_replay_attempts),
                        "shadow_only_candidate_replay_attempts": (
                            shadow_only_candidate_replay_attempts
                        ),
                        "shadow_candidate_probations": shadow_candidate_probations,
                        "shadow_candidate_activations": shadow_candidate_activations,
                        "shadow_candidate_rejections": shadow_candidate_rejections,
                        "shadow_candidate_expirations": shadow_candidate_expirations,
                        "shadow_eprocess_opportunities": (
                            candidate_pool.shadow_eprocess_opportunities
                        ),
                        "shadow_eprocess_crossings": (candidate_pool.shadow_eprocess_crossings),
                        "trusted_candidate_shadow_cooldown_bypasses": (
                            trusted_candidate_shadow_cooldown_bypasses
                        ),
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
                        "reactivation_grace_episodes": (
                            self.config.evolution.active_audit_reactivation_grace_episodes
                        ),
                        "reactivation_grace_audit_suppressions": (
                            reactivation_grace_audit_suppressions
                        ),
                        "reactivation_grace_protected_rollbacks": (
                            reactivation_grace_protected_rollbacks
                        ),
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
                        "early_causal_retirements": active_audit_early_retirements,
                        "early_causal_retirement_precision": (
                            active_audit_early_correct_retirements / active_audit_early_retirements
                            if active_audit_early_retirements
                            else 0.0
                        ),
                        "early_causal_false_retirement_rate": (
                            active_audit_early_false_retirements / active_audit_early_retirements
                            if active_audit_early_retirements
                            else 0.0
                        ),
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
                        "circuit_breaker": {
                            **(
                                circuit_breaker.snapshot()
                                if circuit_breaker is not None
                                else {
                                    "enabled": False,
                                    "lineage_control_enabled": False,
                                    "probes": 0,
                                    "lineage_probes": 0,
                                    "registrations": 0,
                                    "lineage_registrations": 0,
                                    "interventions": 0,
                                    "lineage_interventions": 0,
                                    "confirmations": 0,
                                    "cancellations": 0,
                                    "expirations": 0,
                                    "invalidations": 0,
                                    "invalidated_awaiting_revert": 0,
                                    "pending": 0,
                                }
                            ),
                            "correct_confirmations": circuit_correct_confirmations,
                            "false_confirmations": circuit_false_confirmations,
                            "confirmation_precision": (
                                circuit_correct_confirmations
                                / (circuit_correct_confirmations + circuit_false_confirmations)
                                if circuit_correct_confirmations + circuit_false_confirmations
                                else None
                            ),
                            "false_confirmation_rate": (
                                circuit_false_confirmations
                                / (circuit_correct_confirmations + circuit_false_confirmations)
                                if circuit_correct_confirmations + circuit_false_confirmations
                                else 0.0
                            ),
                            "unconfirmed_persistent_transitions": (
                                circuit_unconfirmed_persistent_transitions
                            ),
                            "mean_post_hoc_oracle_intervention_delta": (
                                statistics.fmean(circuit_oracle_intervention_deltas)
                                if circuit_oracle_intervention_deltas
                                else None
                            ),
                            "control_requests": circuit_control_requests,
                            "control_input_tokens": circuit_control_input_tokens,
                            "control_output_tokens": circuit_control_output_tokens,
                        },
                        "retirement_probation": {
                            **(
                                retirement_probation.snapshot()
                                if retirement_probation is not None
                                else {
                                    "enabled": False,
                                    "registrations": 0,
                                    "interventions": 0,
                                    "confirmations": 0,
                                    "cancellations": 0,
                                    "deferrals": 0,
                                    "expirations": 0,
                                    "invalidations": 0,
                                    "pending": 0,
                                }
                            ),
                            "correct_confirmations": (retirement_probation_correct_confirmations),
                            "false_confirmations": retirement_probation_false_confirmations,
                            "confirmation_precision": (
                                retirement_probation_correct_confirmations
                                / (
                                    retirement_probation_correct_confirmations
                                    + retirement_probation_false_confirmations
                                )
                                if (
                                    retirement_probation_correct_confirmations
                                    + retirement_probation_false_confirmations
                                )
                                else None
                            ),
                            "false_confirmation_rate": (
                                retirement_probation_false_confirmations
                                / (
                                    retirement_probation_correct_confirmations
                                    + retirement_probation_false_confirmations
                                )
                                if (
                                    retirement_probation_correct_confirmations
                                    + retirement_probation_false_confirmations
                                )
                                else 0.0
                            ),
                            "unconfirmed_persistent_transitions": (
                                retirement_probation_unconfirmed_persistent_transitions
                            ),
                            "mean_post_hoc_oracle_intervention_delta": (
                                statistics.fmean(retirement_probation_oracle_deltas)
                                if retirement_probation_oracle_deltas
                                else None
                            ),
                            "control_requests": retirement_probation_control_requests,
                            "control_input_tokens": (retirement_probation_control_input_tokens),
                            "control_output_tokens": (retirement_probation_control_output_tokens),
                        },
                        "dormant_revival": {
                            **(
                                dormant_revival.snapshot()
                                if dormant_revival is not None
                                else {
                                    "enabled": False,
                                    "status_indexed": False,
                                    "semantic_indexed": False,
                                    "probes": 0,
                                    "registrations": 0,
                                    "interventions": 0,
                                    "confirmations": 0,
                                    "cancellations": 0,
                                    "expirations": 0,
                                    "invalidations": 0,
                                    "pending": 0,
                                }
                            ),
                            "correct_confirmations": revival_correct_confirmations,
                            "false_confirmations": revival_false_confirmations,
                            "confirmation_precision": (
                                revival_correct_confirmations
                                / (revival_correct_confirmations + revival_false_confirmations)
                                if revival_correct_confirmations + revival_false_confirmations
                                else None
                            ),
                            "false_confirmation_rate": (
                                revival_false_confirmations
                                / (revival_correct_confirmations + revival_false_confirmations)
                                if revival_correct_confirmations + revival_false_confirmations
                                else 0.0
                            ),
                            "unconfirmed_persistent_transitions": (
                                revival_unconfirmed_persistent_transitions
                            ),
                            "mean_post_hoc_oracle_intervention_delta": (
                                statistics.fmean(revival_oracle_intervention_deltas)
                                if revival_oracle_intervention_deltas
                                else None
                            ),
                            "control_requests": revival_control_requests,
                            "control_input_tokens": revival_control_input_tokens,
                            "control_output_tokens": revival_control_output_tokens,
                        },
                    },
                    "audit": {
                        "frozen": self.frozen_audit,
                        "source_run_id": self.source_run_id,
                        "source_state_hash": self.source_state_hash,
                        "source_dataset_hash": self.source_dataset_hash,
                        "expected_state_hash": expected_state_hash,
                        "audit_variant": self.audit_variant,
                        "excluded_memory_id": self.excluded_memory_id,
                        "excluded_memory_version": self.excluded_memory_version,
                        "initial_active_memories": len(self.initial_memories),
                        "final_state_hash": final_state_hash,
                        "state_unchanged": (
                            not self.frozen_audit
                            or not expected_state_hash
                            or final_state_hash == expected_state_hash
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
