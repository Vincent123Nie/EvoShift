from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

from evoshift.benchmarks.base import BenchmarkAdapter, stable_seed
from evoshift.errors import DatasetError
from evoshift.schemas import BenchmarkSample

Rule = Callable[[int, int], str]
PromptBuilder = Callable[[int, int], str]


@dataclass(frozen=True)
class SyntheticPhaseSpec:
    name: str
    domain: str
    shift_type: str
    prompt_builder: PromptBuilder
    rule: Rule


def _integer(value: int) -> str:
    return str(value)


SYNTHETIC_PHASES: tuple[SyntheticPhaseSpec, ...] = (
    SyntheticPhaseSpec(
        name="phase_0_addition",
        domain="arithmetic/addition",
        shift_type="baseline",
        prompt_builder=lambda x, y: f"Compute {x} + {y}. Return only the integer.",
        rule=lambda x, y: _integer(x + y),
    ),
    SyntheticPhaseSpec(
        name="phase_1_affine",
        domain="arithmetic/affine",
        shift_type="operation_shift",
        prompt_builder=lambda x, y: (
            f"Let F(x, y) = 2*x - y. Compute F({x}, {y}). Return only the integer."
        ),
        rule=lambda x, y: _integer(2 * x - y),
    ),
    SyntheticPhaseSpec(
        name="phase_2_conditional",
        domain="arithmetic/conditional",
        shift_type="control_flow_shift",
        prompt_builder=lambda x, y: (
            f"Given x={x} and y={y}, return x+y if x is even; otherwise return x-y. "
            "Return only the integer."
        ),
        rule=lambda x, y: _integer(x + y if x % 2 == 0 else x - y),
    ),
    SyntheticPhaseSpec(
        name="phase_3_symbolic",
        domain="arithmetic/symbolic_output",
        shift_type="output_schema_shift",
        prompt_builder=lambda x, y: (
            f"Compute z = 3*x + y for x={x}, y={y}. Return POS if z>0, ZERO if z=0, otherwise NEG."
        ),
        rule=lambda x, y: "POS" if 3 * x + y > 0 else ("ZERO" if 3 * x + y == 0 else "NEG"),
    ),
)


class SyntheticShiftBenchmark(BenchmarkAdapter):
    """Deterministic stream with known shifts and old-domain regression probes."""

    def __init__(
        self,
        *,
        seed: int = 42,
        phase_size: int = 12,
        protected_phases: Sequence[str] = ("phase_0_addition",),
        protected_probes_per_phase: int = 2,
        shuffle_within_phase: bool = False,
        limit: int = 0,
    ) -> None:
        if phase_size < 1:
            raise DatasetError("synthetic phase_size must be at least 1")
        if protected_probes_per_phase < 0:
            raise DatasetError("protected_probes_per_phase cannot be negative")
        known_phases = {phase.name for phase in SYNTHETIC_PHASES}
        unknown = sorted(set(protected_phases) - known_phases)
        if unknown:
            raise DatasetError(f"unknown protected synthetic phases: {', '.join(unknown)}")
        self.seed = seed
        self.phase_size = phase_size
        self.protected_phases = tuple(protected_phases)
        self.protected_probes_per_phase = protected_probes_per_phase
        self.shuffle_within_phase = shuffle_within_phase
        self.limit = limit

    def load(self) -> list[BenchmarkSample]:
        stream: list[BenchmarkSample] = []
        phase_by_name = {phase.name: phase for phase in SYNTHETIC_PHASES}

        for phase_index, phase in enumerate(SYNTHETIC_PHASES):
            rng = random.Random(stable_seed(self.seed, phase.name))
            regular = [
                self._make_sample(
                    phase=phase,
                    current_phase=phase,
                    phase_index=phase_index,
                    position=index,
                    x=rng.randint(-25, 25),
                    y=rng.randint(-25, 25),
                    protected_probe=False,
                )
                for index in range(self.phase_size)
            ]
            if self.shuffle_within_phase:
                rng.shuffle(regular)
            regular = [
                sample.model_copy(
                    update={
                        "metadata": {
                            **sample.metadata,
                            "position_in_phase": index,
                            "is_shift_boundary": phase_index > 0 and index == 0,
                        }
                    }
                )
                for index, sample in enumerate(regular)
            ]
            stream.extend(regular)

            if phase_index == 0 or not self.protected_phases:
                continue
            probe_rng = random.Random(stable_seed(self.seed, f"{phase.name}:protected"))
            available = [
                phase_by_name[name]
                for name in self.protected_phases
                if list(phase_by_name).index(name) < phase_index
            ]
            if not available:
                continue
            for probe_index in range(self.protected_probes_per_phase):
                source_phase = available[probe_index % len(available)]
                stream.append(
                    self._make_sample(
                        phase=source_phase,
                        current_phase=phase,
                        phase_index=phase_index,
                        position=self.phase_size + probe_index,
                        x=probe_rng.randint(-25, 25),
                        y=probe_rng.randint(-25, 25),
                        protected_probe=True,
                        probe_index=probe_index,
                    )
                )

        if self.limit < 0:
            raise DatasetError("benchmark limit cannot be negative")
        return stream[: self.limit] if self.limit else stream

    def _make_sample(
        self,
        *,
        phase: SyntheticPhaseSpec,
        current_phase: SyntheticPhaseSpec,
        phase_index: int,
        position: int,
        x: int,
        y: int,
        protected_probe: bool,
        probe_index: int = 0,
    ) -> BenchmarkSample:
        kind = "probe" if protected_probe else "main"
        identifier = (
            f"synthetic:{current_phase.name}:{kind}:{phase.name}:"
            f"{probe_index if protected_probe else position:04d}"
        )
        metadata: dict[str, object] = {
            "synthetic": True,
            "seed": self.seed,
            "phase_index": phase_index,
            "position_in_phase": position,
            "shift_type": current_phase.shift_type,
            "source_phase": phase.name,
            "protected_probe": protected_probe,
            "protected": protected_probe,
            "is_shift_boundary": phase_index > 0 and position == 0 and not protected_probe,
            "operands": {"x": x, "y": y},
        }
        return BenchmarkSample(
            sample_id=identifier,
            prompt=phase.prompt_builder(x, y),
            reference=phase.rule(x, y),
            domain=phase.domain,
            phase=current_phase.name,
            evaluator="exact_match",
            metadata=metadata,
        )


SyntheticShiftBenchmarkAdapter = SyntheticShiftBenchmark
