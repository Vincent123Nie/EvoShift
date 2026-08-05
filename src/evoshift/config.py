from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evoshift.errors import ConfigurationError
from evoshift.schemas import Algorithm, PolicyGenome


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderConfig(ConfigModel):
    kind: str = "responses"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "EVOSHIFT_OPENAI_API_KEY"
    model: str = "gpt-5.6"
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_retries: int = Field(default=4, ge=0, le=20)
    retry_base_seconds: float = Field(default=1.0, ge=0.0)
    max_output_tokens: int = Field(default=512, ge=1)
    reasoning_effort: Optional[str] = "low"
    allow_sampling_params: bool = False
    send_metadata: bool = False
    input_price_per_million: float = Field(default=0.0, ge=0.0)
    output_price_per_million: float = Field(default=0.0, ge=0.0)

    def resolved_base_url(self) -> str:
        return os.getenv("EVOSHIFT_OPENAI_BASE_URL", self.base_url).rstrip("/")

    def resolved_model(self) -> str:
        return os.getenv("EVOSHIFT_OPENAI_MODEL", self.model)


class BudgetConfig(ConfigModel):
    max_requests: int = Field(default=1000, ge=1)
    max_total_tokens: int = Field(default=2_000_000, ge=1)
    max_cost_usd: float = Field(default=50.0, ge=0.0)


class StorageConfig(ConfigModel):
    database_path: str = "data/evoshift.sqlite3"
    cache_path: str = "data/llm_cache.sqlite3"
    runs_dir: str = "runs"
    cache_enabled: bool = True
    isolate_runs: bool = True


class ShiftConfig(ConfigModel):
    enabled: bool = True
    delta: float = Field(default=0.02, ge=0.0)
    threshold: float = Field(default=2.5, gt=0.0)
    min_instances: int = Field(default=8, ge=2)
    novelty_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    novelty_ewma_alpha: float = Field(default=0.2, gt=0.0, le=1.0)
    cooldown_episodes: int = Field(default=8, ge=0)


class EvolutionConfig(ConfigModel):
    enabled: bool = True
    validation_window: int = Field(default=12, ge=1)
    min_validation_examples: int = Field(default=4, ge=1)
    bootstrap_samples: int = Field(default=1000, ge=100)
    confidence_level: float = Field(default=0.90, gt=0.5, lt=1.0)
    min_mean_gain: float = Field(default=0.03, ge=-1.0, le=1.0)
    min_ci_lower_bound: float = Field(default=-0.01, ge=-1.0, le=1.0)
    max_regression_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    max_protected_slice_regression: float = Field(default=0.05, ge=0.0, le=1.0)
    max_cost_increase_ratio: float = Field(default=0.30, ge=0.0)
    paired_replay: bool = True
    feedback_mode: str = "reward_only"
    feedback_default_trust: float = Field(default=1.0, ge=0.0, le=1.0)
    feedback_source_trust: Dict[str, float] = Field(default_factory=dict)
    dynamic_feedback_trust_enabled: bool = False
    dynamic_feedback_context_field: str = "feedback_context"
    dynamic_feedback_min_consistent_observations: int = Field(default=2, ge=2, le=100)
    dynamic_feedback_change_min_span: int = Field(default=0, ge=0, le=10000)
    dynamic_feedback_cold_start_trust: float = Field(default=0.40, ge=0.0, le=1.0)
    dynamic_feedback_conflict_trust: float = Field(default=0.10, ge=0.0, le=1.0)
    dynamic_feedback_prior_strength: float = Field(default=8.0, ge=0.0, le=1000.0)
    dynamic_feedback_max_contexts: int = Field(default=10000, ge=1, le=1000000)
    min_feedback_trust_for_drift: float = Field(default=0.60, ge=0.0, le=1.0)
    min_feedback_trust_for_memory_update: float = Field(default=0.60, ge=0.0, le=1.0)
    min_feedback_trust_for_candidate: float = Field(default=0.60, ge=0.0, le=1.0)
    shadow_candidate_enabled: bool = False
    min_feedback_trust_for_shadow_candidate: float = Field(default=0.10, ge=0.0, le=1.0)
    shadow_eprocess_enabled: bool = False
    shadow_eprocess_null_match_probability: float = Field(default=0.25, gt=0.0, lt=1.0)
    shadow_eprocess_alternative_match_probability: float = Field(default=0.75, gt=0.0, lt=1.0)
    shadow_eprocess_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    min_feedback_trust_for_replay: float = Field(default=0.60, ge=0.0, le=1.0)
    candidate_min_observations: int = Field(default=1, ge=1, le=1000)
    candidate_min_trusted_observations: int = Field(default=1, ge=0, le=1000)
    candidate_min_new_observations: int = Field(default=1, ge=1, le=1000)
    candidate_cooldown_episodes: int = Field(default=4, ge=0, le=10000)
    replay_current_regime_only: bool = True
    candidate_replay_since_first_evidence: bool = False
    max_candidates_per_round: int = Field(default=1, ge=1, le=8)
    policy_evolution_enabled: bool = True
    policy_evolve_on_shift: bool = True
    policy_evolve_if_memory_promoted: bool = False
    future_audit_enabled: bool = False
    future_audit_min_observations: int = Field(default=4, ge=1, le=100)
    future_audit_max_observations: int = Field(default=8, ge=1, le=1000)
    future_audit_early_harm_observations: int = Field(default=0, ge=0, le=100)
    active_audit_enabled: bool = False
    active_audit_max_per_episode: int = Field(default=1, ge=1, le=20)
    active_audit_min_observations: int = Field(default=2, ge=1, le=100)
    active_audit_min_negative_observations: int = Field(default=2, ge=1, le=100)
    active_audit_retire_mean_delta: float = Field(default=-0.25, ge=-1.0, le=0.0)
    active_audit_early_retire_enabled: bool = False
    active_audit_early_retire_delta: float = Field(default=-0.75, ge=-1.0, le=0.0)
    active_audit_cooldown_episodes: int = Field(default=0, ge=0, le=10000)
    active_audit_restore_predecessors: bool = False
    min_feedback_trust_for_active_audit: float = Field(default=0.60, ge=0.0, le=1.0)
    active_audit_circuit_breaker_enabled: bool = False
    active_audit_circuit_breaker_min_trust: float = Field(default=0.10, ge=0.0, le=1.0)
    active_audit_circuit_breaker_delta: float = Field(default=-0.75, ge=-1.0, le=0.0)
    active_audit_circuit_breaker_max_age: int = Field(default=8, ge=1, le=10000)
    active_audit_lineage_control_enabled: bool = False
    dormant_revival_enabled: bool = False
    dormant_revival_min_trust: float = Field(default=0.10, ge=0.0, le=1.0)
    dormant_revival_delta: float = Field(default=0.75, ge=0.0, le=1.0)
    dormant_revival_max_age: int = Field(default=8, ge=1, le=10000)
    dormant_revival_min_retired_age: int = Field(default=18, ge=1, le=10000)
    dormant_revival_status_index_enabled: bool = False
    conflict_supersession_enabled: bool = True

    @model_validator(mode="after")
    def validate_feedback_mode(self) -> "EvolutionConfig":
        if self.feedback_mode not in {"reward_only", "grader_feedback", "reference_upper_bound"}:
            raise ValueError("unsupported feedback_mode")
        invalid_sources = sorted(
            source
            for source, trust in self.feedback_source_trust.items()
            if not source.strip() or not 0.0 <= trust <= 1.0
        )
        if invalid_sources:
            raise ValueError(
                "feedback_source_trust requires non-empty sources and values in [0, 1]: "
                + ", ".join(invalid_sources)
            )
        if not self.dynamic_feedback_context_field.strip():
            raise ValueError("dynamic_feedback_context_field must not be empty")
        if (
            self.shadow_candidate_enabled
            and self.min_feedback_trust_for_shadow_candidate > self.min_feedback_trust_for_candidate
        ):
            raise ValueError(
                "shadow candidate trust threshold must not exceed the trusted candidate threshold"
            )
        if self.shadow_candidate_enabled and not self.paired_replay:
            raise ValueError("shadow candidate admission requires paired replay")
        if self.shadow_candidate_enabled and not self.future_audit_enabled:
            raise ValueError("shadow candidate admission requires future audit")
        if self.shadow_eprocess_enabled and not self.shadow_candidate_enabled:
            raise ValueError("shadow e-process requires shadow candidate admission")
        if (
            self.shadow_eprocess_enabled
            and self.shadow_eprocess_alternative_match_probability
            <= self.shadow_eprocess_null_match_probability
        ):
            raise ValueError(
                "shadow e-process alternative probability must exceed the null probability"
            )
        if (
            self.shadow_candidate_enabled
            and self.min_feedback_trust_for_replay < self.min_feedback_trust_for_candidate
        ):
            raise ValueError(
                "shadow candidate admission requires replay trust to be at least the "
                "trusted candidate threshold"
            )
        if self.future_audit_min_observations > self.future_audit_max_observations:
            raise ValueError(
                "future_audit_min_observations must not exceed future_audit_max_observations"
            )
        if self.future_audit_early_harm_observations > self.future_audit_max_observations:
            raise ValueError(
                "future_audit_early_harm_observations must not exceed future_audit_max_observations"
            )
        if self.active_audit_min_negative_observations > self.active_audit_min_observations:
            raise ValueError(
                "active_audit_min_negative_observations must not exceed "
                "active_audit_min_observations"
            )
        if self.active_audit_circuit_breaker_enabled and not self.active_audit_enabled:
            raise ValueError("active audit circuit breaker requires active audit")
        if self.active_audit_circuit_breaker_enabled and not self.dynamic_feedback_trust_enabled:
            raise ValueError("active audit circuit breaker requires dynamic feedback trust")
        if (
            self.active_audit_circuit_breaker_enabled
            and self.active_audit_circuit_breaker_min_trust
            >= self.min_feedback_trust_for_active_audit
        ):
            raise ValueError(
                "active audit circuit breaker trust floor must be below the ordinary "
                "active audit threshold"
            )
        if (
            self.active_audit_lineage_control_enabled
            and not self.active_audit_circuit_breaker_enabled
        ):
            raise ValueError("lineage control requires the active audit circuit breaker")
        if self.dormant_revival_enabled and not self.dynamic_feedback_trust_enabled:
            raise ValueError("dormant memory revival requires dynamic feedback trust")
        if self.dormant_revival_status_index_enabled and not self.dormant_revival_enabled:
            raise ValueError("status-indexed lifecycle requires dormant memory revival")
        if (
            self.dormant_revival_enabled
            and self.dormant_revival_min_trust >= self.min_feedback_trust_for_active_audit
        ):
            raise ValueError(
                "dormant memory revival trust floor must be below the ordinary "
                "active audit threshold"
            )
        return self


class BenchmarkConfig(ConfigModel):
    kind: str = "jsonl"
    path: Optional[str] = "examples/demo_stream.jsonl"
    name: str = "demo_stream"
    split: str = "test"
    revision: str = ""
    subsets: List[str] = Field(default_factory=list)
    limit: int = Field(default=0, ge=0)
    shuffle: bool = False
    phase_size: int = Field(default=0, ge=0)
    protected_phases: List[str] = Field(default_factory=list)
    feedback_noise_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    feedback_attack_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    feedback_shared_source: bool = False
    feedback_shared_source_name: str = "customer_support_portal"
    feedback_attack_burst_length: int = Field(default=0, ge=0, le=1000)
    policy_schedule: List[str] = Field(default_factory=list)
    coverage_balanced: bool = False
    coverage_min_per_slice: int = Field(default=2, ge=1, le=100)

    @model_validator(mode="after")
    def validate_feedback_source(self) -> "BenchmarkConfig":
        if self.feedback_shared_source and not self.feedback_shared_source_name.strip():
            raise ValueError("feedback_shared_source_name must not be empty")
        invalid_versions = sorted(
            {version for version in self.policy_schedule if version not in {"v1", "v2", "v3"}}
        )
        if invalid_versions:
            raise ValueError(
                "policy_schedule supports only v1, v2, and v3: " + ", ".join(invalid_versions)
            )
        return self


class EvaluationConfig(ConfigModel):
    seed: int = 42
    bootstrap_samples: int = Field(default=2000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    recovery_fraction: float = Field(default=0.90, gt=0.0, le=1.0)
    report_every: int = Field(default=10, ge=1)


class EvoShiftConfig(ConfigModel):
    project: str = "evoshift"
    algorithm: Algorithm = Algorithm.EVOSHIFT
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    shift: ShiftConfig = Field(default_factory=ShiftConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    policy: PolicyGenome = Field(default_factory=PolicyGenome)

    @model_validator(mode="after")
    def validate_algorithm_safety(self) -> "EvoShiftConfig":
        if self.evolution.shadow_candidate_enabled and self.algorithm != Algorithm.EVOSHIFT:
            raise ValueError("shadow candidate admission is supported only by evoshift")
        return self

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid override value: {raw}") from exc


def _apply_override(data: Dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ConfigurationError(f"override must be key=value: {expression}")
    path, raw_value = expression.split("=", 1)
    keys = [part for part in path.split(".") if part]
    if not keys:
        raise ConfigurationError(f"empty override key: {expression}")
    cursor: Dict[str, Any] = data
    for key in keys[:-1]:
        current = cursor.get(key)
        if current is None:
            current = {}
            cursor[key] = current
        if not isinstance(current, dict):
            raise ConfigurationError(f"override traverses non-object field: {path}")
        cursor = current
    cursor[keys[-1]] = _parse_scalar(raw_value)


def load_config(
    path: Optional[Path] = None,
    overrides: Optional[List[str]] = None,
) -> EvoShiftConfig:
    data: Dict[str, Any] = {}
    if path is not None:
        if not path.exists():
            raise ConfigurationError(f"config file does not exist: {path}")
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"cannot read config: {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigurationError("top-level config must be a mapping")
        data = _deep_merge(data, loaded)
    for expression in overrides or []:
        _apply_override(data, expression)
    try:
        return EvoShiftConfig.model_validate(data)
    except Exception as exc:
        raise ConfigurationError(str(exc)) from exc
