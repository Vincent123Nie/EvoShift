from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MemoryKind(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryStatus(str, Enum):
    SHADOW = "shadow"
    PROBATION = "probation"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETIRED = "retired"
    REJECTED = "rejected"


class FailureType(str, Enum):
    WRITE_MISS = "write_miss"
    RETRIEVAL_MISS = "retrieval_miss"
    RANKING_ERROR = "ranking_error"
    CONFLICT_ERROR = "conflict_error"
    REASONING_ERROR = "reasoning_error"
    FORMAT_ERROR = "format_error"
    KNOWLEDGE_GAP = "knowledge_gap"
    TOOL_ERROR = "tool_error"
    UNKNOWN = "unknown"


class Algorithm(str, Enum):
    STATIC = "static"
    SELF_REFINE = "self_refine"
    REFLEXION = "reflexion"
    EVOSHIFT = "evoshift"


class RunMode(str, Enum):
    PREQUENTIAL = "prequential"
    FROZEN_AUDIT = "frozen_audit"


class LLMUsage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    cached: bool = False


class GenerationRequest(StrictModel):
    messages: List[Dict[str, Any]]
    model: Optional[str] = None
    max_output_tokens: int = Field(default=512, ge=1, le=32768)
    reasoning_effort: Optional[str] = "low"
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GenerationResponse(StrictModel):
    text: str
    model: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    response_id: str = ""
    raw: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkSample(StrictModel):
    sample_id: str
    prompt: str
    reference: Any
    domain: str = "default"
    phase: str = "stream"
    evaluator: str = "exact_match"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScoreBundle(StrictModel):
    primary: float = Field(ge=0.0, le=1.0)
    success: bool
    metrics: Dict[str, float] = Field(default_factory=dict)
    feedback: str = ""


class SolverOutput(StrictModel):
    answer: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale_summary: str = ""
    applied_memory_ids: List[str] = Field(default_factory=list)


class MemoryItem(StrictModel):
    memory_id: str
    version: int = Field(default=1, ge=1)
    kind: MemoryKind = MemoryKind.PROCEDURAL
    status: MemoryStatus = MemoryStatus.SHADOW
    trigger: str = Field(min_length=1, max_length=1200)
    scope: str = Field(default="general", max_length=500)
    directive: str = Field(min_length=1, max_length=2000)
    anti_pattern: str = Field(default="", max_length=1200)
    evidence: str = Field(default="", max_length=1200)
    tags: List[str] = Field(default_factory=list, max_length=20)
    source_domains: List[str] = Field(default_factory=list, max_length=20)
    provenance_episode_ids: List[str] = Field(default_factory=list)
    supersedes_memory_ids: List[str] = Field(default_factory=list, max_length=20)
    valid_from_episode_id: str = ""
    valid_from_index: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    alpha: float = Field(default=1.0, gt=0.0)
    beta: float = Field(default=1.0, gt=0.0)
    use_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    causal_audit_count: int = Field(default=0, ge=0)
    causal_positive_count: int = Field(default=0, ge=0)
    causal_negative_count: int = Field(default=0, ge=0)
    causal_neutral_count: int = Field(default=0, ge=0)
    causal_delta_sum: float = 0.0
    causal_last_audit_index: Optional[int] = Field(default=None, ge=0)
    validation_gain: float = 0.0
    validation_lcb: float = 0.0
    regression_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def posterior_utility(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def causal_mean_delta(self) -> float:
        return self.causal_delta_sum / self.causal_audit_count if self.causal_audit_count else 0.0


class RetrievedMemory(StrictModel):
    item: MemoryItem
    relevance: float = 0.0
    utility: float = 0.0
    exploration: float = 0.0
    final_score: float = 0.0


class AgentPrediction(StrictModel):
    output: SolverOutput
    retrieved: List[RetrievedMemory] = Field(default_factory=list)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    raw_text: str = ""


class PolicyGenome(StrictModel):
    version: int = Field(default=1, ge=1)
    top_k: int = Field(default=4, ge=0, le=20)
    memory_token_budget: int = Field(default=900, ge=0, le=8000)
    bm25_k1: float = Field(default=1.5, ge=0.1, le=5.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    relevance_weight: float = Field(default=0.62, ge=0.0, le=2.0)
    utility_weight: float = Field(default=0.28, ge=0.0, le=2.0)
    exploration_weight: float = Field(default=0.10, ge=0.0, le=2.0)
    mmr_lambda: float = Field(default=0.75, ge=0.0, le=1.0)
    allow_cross_domain_transfer: bool = False
    write_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    dedup_similarity_threshold: float = Field(default=0.86, ge=0.0, le=1.0)
    rollback_utility_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    rollback_min_uses: int = Field(default=8, ge=1, le=1000)
    critic_on_failure: bool = True
    learn_from_success_every: int = Field(default=0, ge=0, le=10000)
    prompt_versions: Dict[str, str] = Field(
        default_factory=lambda: {"solver": "v1", "critic": "v1"}
    )


EVOLVABLE_POLICY_FIELDS = {
    "top_k",
    "memory_token_budget",
    "bm25_k1",
    "bm25_b",
    "relevance_weight",
    "utility_weight",
    "exploration_weight",
    "mmr_lambda",
    "write_confidence_threshold",
    "dedup_similarity_threshold",
}


class PolicyPatch(StrictModel):
    patch_id: str
    base_version: int = Field(ge=1)
    changes: Dict[str, Any]
    hypothesis: str = Field(min_length=1, max_length=1200)
    failure_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("changes")
    @classmethod
    def changes_are_allowlisted(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        illegal = sorted(set(value) - EVOLVABLE_POLICY_FIELDS)
        if illegal:
            raise ValueError(f"non-evolvable policy fields: {', '.join(illegal)}")
        if not value:
            raise ValueError("a policy patch must change at least one field")
        return value


class FailureRecord(StrictModel):
    failure_id: str
    episode_id: str
    failure_type: FailureType = FailureType.UNKNOWN
    signature: str
    evidence: str = ""
    proposed_memory: Optional[MemoryItem] = None
    proposed_patch: Optional[PolicyPatch] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ShiftReport(StrictModel):
    detected: bool = False
    detector: str = "page_hinkley"
    statistic: float = 0.0
    threshold: float = 0.0
    novelty: float = Field(default=0.0, ge=0.0)
    domain: str = "global"
    episode_index: int = Field(default=0, ge=0)
    reason: str = ""


class Episode(StrictModel):
    episode_id: str
    run_id: str
    index: int = Field(ge=0)
    sample: BenchmarkSample
    output: SolverOutput
    score: ScoreBundle
    feedback_score: Optional[ScoreBundle] = None
    feedback_trust: float = Field(default=1.0, ge=0.0, le=1.0)
    feedback_eligible: bool = True
    feedback_trust_reason: str = "default_trust"
    selected_memory_ids: List[str] = Field(default_factory=list)
    policy_version: int = Field(default=1, ge=1)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    shift: Optional[ShiftReport] = None
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def adaptation_score(self) -> ScoreBundle:
        """Feedback visible to the online learner; defaults to the oracle score."""

        return self.feedback_score or self.score


class ValidationResult(StrictModel):
    candidate_id: str
    candidate_type: str
    n: int = Field(ge=0)
    control_mean: float = 0.0
    candidate_mean: float = 0.0
    mean_delta: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    regression_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    cost_delta_ratio: float = 0.0
    protected_slice_regression: float = 0.0
    deltas: List[float] = Field(default_factory=list)
    oracle_control_mean: Optional[float] = None
    oracle_candidate_mean: Optional[float] = None
    oracle_mean_delta: Optional[float] = None
    observation_start_index: Optional[int] = None
    observation_end_index: Optional[int] = None


class PromotionDecision(StrictModel):
    promote: bool
    reason: str
    result: ValidationResult
    gate_checks: Dict[str, bool] = Field(default_factory=dict)


class RunManifest(StrictModel):
    run_id: str
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: Optional[datetime] = None
    git_commit: str = "unknown"
    git_dirty: bool = False
    config_hash: str
    dataset_hash: str = ""
    model: str
    algorithm: Algorithm
    run_mode: RunMode = RunMode.PREQUENTIAL
    source_run_id: str = ""
    source_state_hash: str = ""
    source_dataset_hash: str = ""
    seed: int
    python_version: str
    platform: str
