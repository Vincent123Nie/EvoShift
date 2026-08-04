from __future__ import annotations

import pytest

from evoshift.config import EvolutionConfig
from evoshift.evolution.replay import ReplayVerifier, select_replay_buffer
from evoshift.schemas import (
    AgentPrediction,
    BenchmarkSample,
    Episode,
    MemoryItem,
    PolicyGenome,
    ScoreBundle,
    SolverOutput,
)


def _episode(index: int, phase: str, prompt: str, protected: bool = False) -> Episode:
    return Episode(
        episode_id=f"e{index}",
        run_id="r",
        index=index,
        sample=BenchmarkSample(
            sample_id=f"s{index}",
            prompt=prompt,
            reference="x",
            phase=phase,
            metadata={"protected": protected},
        ),
        output=SolverOutput(answer="x"),
        score=ScoreBundle(primary=1.0, success=True),
    )


def test_replay_buffer_mixes_related_and_protected_examples() -> None:
    episodes = [
        _episode(0, "old", "calendar weekday", True),
        _episode(1, "new", "affine arithmetic one"),
        _episode(2, "new", "affine arithmetic two"),
        _episode(3, "new", "unrelated"),
    ]
    selected, mask = select_replay_buffer(
        episodes, 3, query="affine arithmetic", protected_phases=["old"]
    )
    assert len(selected) == 3
    assert any(mask)
    assert {episode.index for episode in selected} >= {0, 1}


class _FeedbackAgent:
    async def solve(
        self,
        sample: BenchmarkSample,
        policy: PolicyGenome,
        extra_memories: list[MemoryItem] | None = None,
    ) -> AgentPrediction:
        del sample, policy
        answer = "B" if extra_memories else "A"
        return AgentPrediction(output=SolverOutput(answer=answer))


@pytest.mark.asyncio
async def test_replay_promotion_uses_observable_feedback_labels() -> None:
    sample = BenchmarkSample(
        sample_id="policy",
        prompt="choose",
        reference="A",
        metadata={"feedback_reference": "B"},
    )
    episode = Episode(
        episode_id="e",
        run_id="r",
        index=0,
        sample=sample,
        output=SolverOutput(answer="A"),
        score=ScoreBundle(primary=1.0, success=True),
        feedback_score=ScoreBundle(primary=0.0, success=False),
    )
    verifier = ReplayVerifier(
        _FeedbackAgent(),  # type: ignore[arg-type]
        EvolutionConfig(
            validation_window=1,
            min_validation_examples=1,
            bootstrap_samples=100,
            min_mean_gain=0.5,
            min_ci_lower_bound=0.0,
            max_regression_rate=0.0,
            max_cost_increase_ratio=1.0,
        ),
    )

    decision = await verifier.validate_memory(
        MemoryItem(memory_id="m", trigger="choose", directive="answer B"),
        [episode],
        PolicyGenome(),
    )

    assert decision.promote is True
    assert decision.result.control_mean == 0.0
    assert decision.result.candidate_mean == 1.0
