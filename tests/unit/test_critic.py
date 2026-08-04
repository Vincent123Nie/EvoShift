import pytest

from evoshift.config import EvolutionConfig, ProviderConfig
from evoshift.evolution.critic import ExperienceCritic, extract_json_object
from evoshift.schemas import (
    BenchmarkSample,
    Episode,
    GenerationRequest,
    GenerationResponse,
    ScoreBundle,
    SolverOutput,
)


class StubClient:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
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
    client = StubClient()
    critic = ExperienceCritic(client, ProviderConfig(model="fake"), EvolutionConfig())
    result = await critic.analyze(_episode(), [])
    assert result.failure_type.value == "reasoning_error"
    assert result.proposed_memory is not None
    assert result.proposed_memory.directive.startswith("Verify parity")
    assert result.proposed_memory.provenance_episode_ids == ["e1"]
    assert result.proposed_memory.source_domains == ["logic"]


@pytest.mark.asyncio
async def test_critic_uses_observable_feedback_instead_of_oracle_score() -> None:
    client = StubClient()
    critic = ExperienceCritic(client, ProviderConfig(model="fake"), EvolutionConfig())
    episode = _episode().model_copy(
        update={"feedback_score": ScoreBundle(primary=1.0, success=True, feedback="accepted")}
    )

    await critic.analyze(episode, [])

    request = client.requests[0]
    content = request.messages[-1]["content"]
    assert '"reward": 1.0' in content
    assert '"feedback": "success"' in content
