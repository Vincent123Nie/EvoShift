from evoshift.evolution.replay import select_replay_buffer
from evoshift.schemas import BenchmarkSample, Episode, ScoreBundle, SolverOutput


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
