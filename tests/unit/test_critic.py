import pytest

from evoshift.config import EvolutionConfig, ProviderConfig
from evoshift.evolution.critic import ExperienceCritic, extract_json_object
from evoshift.schemas import (
    BenchmarkSample,
    Episode,
    GenerationResponse,
    ScoreBundle,
    SolverOutput,
)


class StubClient:
    async def generate(self, request: object) -> GenerationResponse:
        return GenerationResponse(
            text="""```json
            {"failure_type":"reasoning_error","signature":"missed parity check",
             "evidence":"wrong result","confidence":0.8,
             "memory":{"kind":"procedural","trigger":"parity problems",
             "scope":"logic","directive":"Verify parity before answering.",
             "anti_pattern":"Do not guess.","evidence":"episode failure","tags":["parity"]}}
            ```""",
            model="fake",
        )


def _episode() -> Episode:
    return Episode(
        episode_id="e1",
        run_id="r1",
        index=0,
        sample=BenchmarkSample(sample_id="s1", prompt="Is 3 even?", reference="no", domain="logic"),
        output=SolverOutput(answer="yes"),
        score=ScoreBundle(primary=0.0, success=False),
    )


def test_extract_json_object_from_fence() -> None:
    assert extract_json_object('prefix ```json {"x": 1} ``` suffix') == {"x": 1}


@pytest.mark.asyncio
async def test_critic_returns_typed_memory() -> None:
    critic = ExperienceCritic(StubClient(), ProviderConfig(model="fake"), EvolutionConfig())
    result = await critic.analyze(_episode(), [])
    assert result.failure_type.value == "reasoning_error"
    assert result.proposed_memory is not None
    assert result.proposed_memory.directive.startswith("Verify parity")
    assert result.proposed_memory.provenance_episode_ids == ["e1"]
