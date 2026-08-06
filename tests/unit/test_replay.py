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


def _episode(
    index: int,
    phase: str,
    prompt: str,
    protected: bool = False,
    trust: float = 1.0,
    context: str = "",
) -> Episode:
    return Episode(
        episode_id=f"e{index}",
        run_id="r",
        index=index,
        sample=BenchmarkSample(
            sample_id=f"s{index}",
            prompt=prompt,
            reference="x",
            phase=phase,
            metadata={
                "protected": protected,
                "feedback_source": "portal",
                "feedback_context": context or prompt,
            },
        ),
        output=SolverOutput(answer="x"),
        score=ScoreBundle(primary=1.0, success=True),
        feedback_trust=trust,
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


def test_replay_buffer_filters_untrusted_feedback_and_old_regime_examples() -> None:
    episodes = [
        _episode(0, "old", "refund day 10", protected=False),
        _episode(1, "old", "refund invariant", protected=True),
        _episode(10, "current", "refund day 10", protected=False),
        _episode(11, "current", "refund corrupted", protected=False, trust=0.1),
        _episode(12, "current", "refund invariant current", protected=True),
    ]

    selected, mask = select_replay_buffer(
        episodes,
        5,
        query="refund",
        protected_phases=["old"],
        min_feedback_trust=0.6,
        regime_start_index=10,
    )

    assert {episode.index for episode in selected} == {0, 1, 10, 12}
    assert 11 not in {episode.index for episode in selected}
    assert sum(mask) == 2


def test_hidden_protected_metadata_does_not_change_online_replay_selection() -> None:
    plain = [_episode(0, "old", "refund old"), _episode(1, "new", "refund new")]
    annotated = [_episode(0, "old", "refund old", True), _episode(1, "new", "refund new")]

    plain_selected, plain_mask = select_replay_buffer(plain, 2, query="refund")
    annotated_selected, annotated_mask = select_replay_buffer(annotated, 2, query="refund")

    assert [episode.index for episode in annotated_selected] == [
        episode.index for episode in plain_selected
    ]
    assert annotated_mask == plain_mask == [False, False]


def test_replay_buffer_reserves_observable_historical_context_anchor() -> None:
    episodes = [
        _episode(0, "old", "refund old a", context="refund:a"),
        _episode(1, "old", "refund old b", context="refund:b"),
        _episode(2, "old", "refund old a recent", context="refund:a"),
        _episode(10, "current", "refund current a", context="refund:a"),
        _episode(11, "current", "refund current b", context="refund:b"),
        _episode(12, "current", "refund current c", context="refund:c"),
    ]

    selected, mask = select_replay_buffer(
        episodes,
        3,
        query="refund",
        regime_start_index=10,
        historical_context_anchors_enabled=True,
        historical_context_anchor_fraction=1.0 / 3.0,
    )

    selected_indices = {episode.index for episode in selected}
    assert 2 in selected_indices
    assert 0 not in selected_indices
    assert len([index for index in selected_indices if index < 10]) == 1
    assert mask == [False, False, False]


def test_replay_buffer_excludes_candidate_provenance_context_from_history() -> None:
    episodes = [
        _episode(0, "old", "refund old changed", context="refund:changed"),
        _episode(1, "old", "refund old stable", context="refund:stable"),
        _episode(2, "old", "refund old changed recent", context="refund:changed"),
        _episode(10, "current", "refund current changed", context="refund:changed"),
    ]

    selected, mask = select_replay_buffer(
        episodes,
        3,
        query="refund changed",
        regime_start_index=10,
        historical_context_anchors_enabled=True,
        historical_context_anchor_fraction=1.0 / 3.0,
        excluded_historical_contexts=(("portal", "refund:changed"),),
    )

    assert [episode.index for episode in selected] == [1, 10]
    assert mask == [False, False]


def test_replay_buffer_rejects_anchor_quota_that_rounds_to_zero() -> None:
    with pytest.raises(ValueError, match="window \* anchor fraction"):
        select_replay_buffer(
            [_episode(0, "old", "refund")],
            2,
            historical_context_anchors_enabled=True,
            historical_context_anchor_fraction=1.0 / 3.0,
        )


def test_replay_buffer_rejects_oracle_context_metadata_field() -> None:
    with pytest.raises(ValueError, match="typed learner-visible feedback_context"):
        select_replay_buffer(
            [_episode(0, "old", "refund")],
            3,
            observable_context_field="protected",
        )


def test_replay_buffer_backfills_remaining_trusted_history_without_oracle_labels() -> None:
    episodes = [
        _episode(0, "old", "refund old a", context="refund:a"),
        _episode(1, "old", "refund old b", context="refund:b"),
        _episode(2, "old", "refund old a recent", context="refund:a"),
        _episode(10, "current", "refund current", context="refund:c"),
    ]

    selected, mask = select_replay_buffer(
        episodes,
        4,
        query="refund",
        regime_start_index=10,
        historical_context_anchors_enabled=True,
        historical_context_anchor_fraction=0.25,
    )

    assert [episode.index for episode in selected] == [0, 1, 2, 10]
    assert mask == [False, False, False, False]


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
    assert decision.result.replay_sample_ids == ["policy"]
    assert decision.result.replay_episode_indices == [0]
    assert decision.result.replay_protected_mask == [False]
