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
from evoshift.evaluation import compute_stream_metrics, score_sample
from evoshift.evolution import ExperienceCritic, PageHinkleyShiftDetector
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
            detector = PageHinkleyShiftDetector(self.config.shift)
            verifier = ReplayVerifier(
                agent,
                self.config.evolution,
                protected_phases=self.config.benchmark.protected_phases,
            )
            behavior = behavior_for(self.config.algorithm)
            episodes: List[Episode] = []
            failures: List[FailureRecord] = []
            decisions: List[PromotionDecision] = []
            rollbacks = 0

            for index, sample in enumerate(samples):
                active_before = memory.active()
                prediction = await agent.solve(
                    sample,
                    policy,
                    use_memory=behavior.use_memory,
                    self_refine=behavior.self_refine,
                )
                score = score_sample(sample, prediction.output.answer)
                selected_ids = [item.item.memory_id for item in prediction.retrieved]
                novelty = (
                    1.0 - max(item.relevance for item in prediction.retrieved)
                    if active_before and prediction.retrieved
                    else 0.0
                )
                shift = detector.update(score.primary, novelty, index, sample.domain)
                episode = Episode(
                    episode_id=f"ep-{uuid.uuid4().hex[:16]}",
                    run_id=run_id,
                    index=index,
                    sample=sample,
                    output=prediction.output,
                    score=score,
                    selected_memory_ids=selected_ids,
                    policy_version=policy.version,
                    usage=prediction.usage,
                    shift=shift,
                )
                episodes.append(episode)
                store.save_episode(episode)
                artifacts.append_episode(episode)

                if not self.frozen_audit:
                    credited_ids = prediction.output.applied_memory_ids or selected_ids
                    retired = memory.record_outcome(credited_ids, score.success, policy)
                    for item in retired:
                        rollbacks += 1
                        payload = {"reason": "posterior utility below rollback threshold"}
                        store.record_event(
                            run_id,
                            "memory_rollback",
                            item.memory_id,
                            payload,
                        )

                should_extract = not score.success or (
                    policy.learn_from_success_every > 0
                    and (index + 1) % policy.learn_from_success_every == 0
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
                        candidate = memory.stage(failure.proposed_memory, policy)
                        store.record_event(
                            run_id,
                            "memory_staged",
                            f"{candidate.memory_id}@v{candidate.version}",
                            {"status": candidate.status.value},
                        )
                        if candidate.status != MemoryStatus.REJECTED:
                            if not behavior.verify_before_promotion:
                                active = memory.activate(candidate, 0.0, 0.0, 0.0)
                                store.record_event(
                                    run_id,
                                    "memory_promoted_unverified",
                                    f"{active.memory_id}@v{active.version}",
                                    {"algorithm": self.config.algorithm.value},
                                )
                            elif len(episodes) >= self.config.evolution.min_validation_examples:
                                decision = await verifier.validate_memory(
                                    candidate, episodes, policy
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
                                else:
                                    memory.reject(candidate)

                if (
                    shift.detected
                    and not self.frozen_audit
                    and self.config.evolution.enabled
                    and self.config.evolution.policy_evolution_enabled
                    and self.config.evolution.policy_evolve_on_shift
                    and behavior.evolve_policy
                    and failures
                    and len(episodes) >= self.config.evolution.min_validation_examples
                ):
                    recent_failures = failures[-self.config.evolution.validation_window :]
                    patch = propose_bounded_policy_patch(policy, recent_failures, shift)
                    challenger = apply_policy_patch(policy, patch)
                    decision = await verifier.validate_policy(challenger, policy, episodes)
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
            promoted = sum(int(decision.promote) for decision in decisions)
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
                        "candidates_evaluated": len(decisions),
                        "candidates_promoted": promoted,
                        "candidates_rejected": len(decisions) - promoted,
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
