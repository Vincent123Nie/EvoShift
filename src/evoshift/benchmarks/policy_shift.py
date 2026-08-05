from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

from evoshift.benchmarks.base import BenchmarkAdapter, stable_seed
from evoshift.errors import DatasetError
from evoshift.schemas import BenchmarkSample

APPROVE = "APPROVE"
DENY = "DENY"


@dataclass(frozen=True)
class PolicyPhase:
    name: str
    version: str
    standard_window: int
    premium_window: int
    scenario: Callable[[int, random.Random], "PolicyScenario"]


@dataclass(frozen=True)
class PolicyScenario:
    tier: str
    request_day: int
    transition_case: bool
    protected: bool
    future_change_case: bool
    case_type: str


def _v1_scenario(index: int, rng: random.Random) -> PolicyScenario:
    cases = (
        PolicyScenario("standard", rng.randint(1, 7), False, True, False, "protected_approve"),
        PolicyScenario("standard", rng.randint(8, 14), False, False, True, "future_v2_expansion"),
        PolicyScenario("premium", rng.randint(15, 30), False, False, True, "future_v3_exception"),
        PolicyScenario("premium", rng.randint(31, 45), False, True, False, "protected_deny"),
    )
    return cases[index % len(cases)]


def _v2_scenario(index: int, rng: random.Random) -> PolicyScenario:
    cases = (
        PolicyScenario("standard", rng.randint(8, 14), True, False, False, "v2_expanded_window"),
        PolicyScenario("premium", rng.randint(8, 14), True, False, False, "v2_expanded_window"),
        PolicyScenario("standard", rng.randint(1, 7), False, True, False, "protected_approve"),
        PolicyScenario("premium", rng.randint(15, 30), False, False, True, "future_v3_exception"),
        PolicyScenario("standard", rng.randint(31, 45), False, True, False, "protected_deny"),
        PolicyScenario("premium", rng.randint(31, 45), False, True, False, "protected_deny"),
    )
    return cases[index % len(cases)]


def _v3_scenario(index: int, rng: random.Random) -> PolicyScenario:
    cases = (
        PolicyScenario("premium", rng.randint(15, 30), True, False, False, "v3_premium_exception"),
        PolicyScenario(
            "standard", rng.randint(15, 30), False, True, False, "protected_standard_deny"
        ),
        PolicyScenario("premium", rng.randint(1, 14), False, True, False, "protected_approve"),
        PolicyScenario("standard", rng.randint(31, 45), False, True, False, "protected_deny"),
    )
    return cases[index % len(cases)]


POLICY_TEMPLATES: dict[str, PolicyPhase] = {
    "v1": PolicyPhase("", "v1", 7, 7, _v1_scenario),
    "v2": PolicyPhase("", "v2", 14, 14, _v2_scenario),
    "v3": PolicyPhase("", "v3", 14, 30, _v3_scenario),
}
DEFAULT_POLICY_SCHEDULE: tuple[str, ...] = ("v1", "v2", "v3")
POLICY_PHASES: tuple[PolicyPhase, ...] = tuple(
    PolicyPhase(
        f"phase_{phase_index}",
        version,
        POLICY_TEMPLATES[version].standard_window,
        POLICY_TEMPLATES[version].premium_window,
        POLICY_TEMPLATES[version].scenario,
    )
    for phase_index, version in enumerate(DEFAULT_POLICY_SCHEDULE)
)
VALID_MEMORY_TAGS: dict[str, tuple[str, ...]] = {
    "v1": (),
    "v2": ("policy_v2",),
    "v3": ("policy_v2", "policy_v3"),
}


class PolicyShiftBenchmark(BenchmarkAdapter):
    """Enterprise-style policy stream with hidden truth and noisy observed feedback.

    The prompt never reveals the active refund policy. ``reference`` is the hidden
    oracle used only for final evaluation, while ``metadata.feedback_reference`` is
    the label visible to online adaptation and may be noisy or adversarial.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        phase_size: int = 24,
        feedback_noise_rate: float = 0.10,
        feedback_attack_rate: float = 0.10,
        feedback_shared_source: bool = False,
        feedback_shared_source_name: str = "customer_support_portal",
        feedback_attack_burst_length: int = 0,
        policy_schedule: Sequence[str] | None = None,
        shuffle_within_phase: bool = False,
        limit: int = 0,
    ) -> None:
        if phase_size < 4:
            raise DatasetError("policy_shift phase_size must be at least 4")
        for name, value in (
            ("feedback_noise_rate", feedback_noise_rate),
            ("feedback_attack_rate", feedback_attack_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise DatasetError(f"{name} must be between 0 and 1")
        if limit < 0:
            raise DatasetError("benchmark limit cannot be negative")
        if feedback_shared_source and not feedback_shared_source_name.strip():
            raise DatasetError("feedback_shared_source_name must not be empty")
        if feedback_attack_burst_length < 0:
            raise DatasetError("feedback_attack_burst_length cannot be negative")
        schedule = tuple(policy_schedule or DEFAULT_POLICY_SCHEDULE)
        if not schedule:
            raise DatasetError("policy_shift policy_schedule must not be empty")
        invalid_versions = sorted(set(schedule) - set(POLICY_TEMPLATES))
        if invalid_versions:
            raise DatasetError(
                "policy_shift policy_schedule supports only v1, v2, and v3: "
                + ", ".join(invalid_versions)
            )
        self.seed = seed
        self.phase_size = phase_size
        self.feedback_noise_rate = feedback_noise_rate
        self.feedback_attack_rate = feedback_attack_rate
        self.feedback_shared_source = feedback_shared_source
        self.feedback_shared_source_name = feedback_shared_source_name.strip()
        self.feedback_attack_burst_length = feedback_attack_burst_length
        self.policy_schedule = schedule
        self.shuffle_within_phase = shuffle_within_phase
        self.limit = limit

    def load(self) -> list[BenchmarkSample]:
        stream: list[BenchmarkSample] = []
        phases = [
            PolicyPhase(
                f"phase_{phase_index}",
                version,
                POLICY_TEMPLATES[version].standard_window,
                POLICY_TEMPLATES[version].premium_window,
                POLICY_TEMPLATES[version].scenario,
            )
            for phase_index, version in enumerate(self.policy_schedule)
        ]
        for phase_index, phase in enumerate(phases):
            rng = random.Random(stable_seed(self.seed, phase.name))
            previous_phase = phases[phase_index - 1] if phase_index > 0 else None
            future_phases = phases[phase_index + 1 :]
            samples = [
                self._make_sample(
                    phase,
                    phase_index,
                    position,
                    rng,
                    previous_phase=previous_phase,
                    future_phases=future_phases,
                    previously_seen_versions=self.policy_schedule[:phase_index],
                )
                for position in range(self.phase_size)
            ]
            if self.shuffle_within_phase:
                rng.shuffle(samples)
            samples = [
                sample.model_copy(
                    update={
                        "metadata": {
                            **sample.metadata,
                            "position_in_phase": position,
                            "is_shift_boundary": phase_index > 0 and position == 0,
                        }
                    }
                )
                for position, sample in enumerate(samples)
            ]
            stream.extend(samples)
        return stream[: self.limit] if self.limit else stream

    def _make_sample(
        self,
        phase: PolicyPhase,
        phase_index: int,
        position: int,
        rng: random.Random,
        *,
        previous_phase: PolicyPhase | None,
        future_phases: Sequence[PolicyPhase],
        previously_seen_versions: Sequence[str],
    ) -> BenchmarkSample:
        scenario = phase.scenario(position, rng)
        tier = scenario.tier
        request_day = scenario.request_day
        window = phase.premium_window if tier == "premium" else phase.standard_window
        oracle = APPROVE if request_day <= window else DENY
        previous_oracle = (
            self._oracle_for(previous_phase, tier, request_day) if previous_phase else None
        )
        transition_case = previous_oracle is not None and previous_oracle != oracle
        future_change_case = not transition_case and any(
            self._oracle_for(future_phase, tier, request_day) != oracle
            for future_phase in future_phases
        )
        protected = not transition_case and not future_change_case
        scenario = PolicyScenario(
            tier=tier,
            request_day=request_day,
            transition_case=transition_case,
            protected=protected,
            future_change_case=future_change_case,
            case_type=scenario.case_type,
        )
        feedback, feedback_kind, feedback_source, feedback_attack_goal = self._observed_feedback(
            oracle,
            scenario=scenario,
            phase=phase,
            position=position,
        )
        return BenchmarkSample(
            sample_id=f"policy_shift:{phase_index}:{position:04d}",
            prompt=(
                f"Refund case: customer_tier={tier.upper()}; request_day={request_day}. "
                "Return APPROVE or DENY only."
            ),
            reference=oracle,
            domain="customer_support/refund_policy",
            phase=phase.name,
            evaluator="exact_match",
            metadata={
                "synthetic": True,
                "benchmark": "policy_shift",
                "seed": self.seed,
                "phase_index": phase_index,
                "position_in_phase": position,
                "is_shift_boundary": phase_index > 0 and position == 0,
                "policy_version": phase.version,
                "previous_policy_version": previous_phase.version if previous_phase else "",
                "is_policy_reversion": phase.version in previously_seen_versions,
                "policy_schedule": list(self.policy_schedule),
                "standard_window": phase.standard_window,
                "premium_window": phase.premium_window,
                "customer_tier": tier,
                "request_day": request_day,
                "case_type": scenario.case_type,
                "policy_changed_case": scenario.transition_case,
                "transition_case": scenario.transition_case,
                "protected": scenario.protected,
                "future_change_case": scenario.future_change_case,
                "feedback_context": self._feedback_context(tier, request_day),
                "feedback_reference": feedback,
                "feedback_kind": feedback_kind,
                "feedback_source": feedback_source,
                "feedback_attack_goal": feedback_attack_goal,
                "feedback_corrupted": feedback != oracle,
                "valid_memory_tags": list(VALID_MEMORY_TAGS[phase.version]),
                "stale_memory_tags": sorted(
                    set().union(*VALID_MEMORY_TAGS.values()) - set(VALID_MEMORY_TAGS[phase.version])
                ),
            },
        )

    @staticmethod
    def _oracle_for(phase: PolicyPhase, tier: str, request_day: int) -> str:
        window = phase.premium_window if tier == "premium" else phase.standard_window
        return APPROVE if request_day <= window else DENY

    def _observed_feedback(
        self,
        oracle: str,
        *,
        scenario: PolicyScenario,
        phase: PolicyPhase,
        position: int,
    ) -> tuple[str, str, str, str]:
        rng = random.Random(stable_seed(self.seed, f"{phase.name}:feedback:{position}"))
        source = self.feedback_shared_source_name if self.feedback_shared_source else ""
        if (
            phase.version == "v1"
            and scenario.future_change_case
            and position // 4 < self.feedback_attack_burst_length
        ):
            return (
                self._flip(oracle),
                "attack",
                source or "untrusted_policy_message",
                "premature_update",
            )
        if (
            scenario.transition_case
            and phase.version != "v1"
            and rng.random() < self.feedback_attack_rate
        ):
            return (
                self._flip(oracle),
                "attack",
                source or "untrusted_policy_message",
                "rollback_to_old_rule",
            )
        if rng.random() < self.feedback_noise_rate:
            return self._flip(oracle), "noise", source or "execution_feedback", "incidental"
        return oracle, "clean", source or "verified_policy_engine", ""

    @staticmethod
    def _feedback_context(tier: str, request_day: int) -> str:
        if request_day <= 7:
            return "refund:any:days_1_7"
        if request_day <= 14:
            return "refund:any:days_8_14"
        if request_day <= 30:
            return f"refund:{tier}:days_15_30"
        return "refund:any:days_31_plus"

    @staticmethod
    def _flip(label: str) -> str:
        return DENY if label == APPROVE else APPROVE


PolicyShiftBenchmarkAdapter = PolicyShiftBenchmark
