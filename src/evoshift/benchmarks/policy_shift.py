from __future__ import annotations

import random
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


POLICY_PHASES: tuple[PolicyPhase, ...] = (
    PolicyPhase("phase_0", "v1", 7, 7, _v1_scenario),
    PolicyPhase("phase_1", "v2", 14, 14, _v2_scenario),
    PolicyPhase("phase_2", "v3", 14, 30, _v3_scenario),
)


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
        self.seed = seed
        self.phase_size = phase_size
        self.feedback_noise_rate = feedback_noise_rate
        self.feedback_attack_rate = feedback_attack_rate
        self.feedback_shared_source = feedback_shared_source
        self.feedback_shared_source_name = feedback_shared_source_name.strip()
        self.feedback_attack_burst_length = feedback_attack_burst_length
        self.shuffle_within_phase = shuffle_within_phase
        self.limit = limit

    def load(self) -> list[BenchmarkSample]:
        stream: list[BenchmarkSample] = []
        for phase_index, phase in enumerate(POLICY_PHASES):
            rng = random.Random(stable_seed(self.seed, phase.name))
            samples = [
                self._make_sample(phase, phase_index, position, rng)
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
    ) -> BenchmarkSample:
        scenario = phase.scenario(position, rng)
        tier = scenario.tier
        request_day = scenario.request_day
        window = phase.premium_window if tier == "premium" else phase.standard_window
        oracle = APPROVE if request_day <= window else DENY
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
            },
        )

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
