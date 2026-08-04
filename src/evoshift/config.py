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
    max_candidates_per_round: int = Field(default=1, ge=1, le=8)
    policy_evolution_enabled: bool = True
    policy_evolve_on_shift: bool = True

    @model_validator(mode="after")
    def validate_feedback_mode(self) -> "EvolutionConfig":
        if self.feedback_mode not in {"reward_only", "grader_feedback", "reference_upper_bound"}:
            raise ValueError("unsupported feedback_mode")
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
