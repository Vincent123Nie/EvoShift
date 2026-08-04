from pathlib import Path

import pytest

from evoshift.agents.solver import MemoryAgent, parse_solver_output
from evoshift.config import ProviderConfig
from evoshift.memory import MemoryManager
from evoshift.schemas import (
    BenchmarkSample,
    GenerationRequest,
    GenerationResponse,
    MemoryItem,
    MemoryStatus,
    PolicyGenome,
)
from evoshift.storage import SQLiteStore


class CapturingClient:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request: object) -> GenerationResponse:
        self.requests.append(request)
        return GenerationResponse(
            text=(
                '{"answer":"4","confidence":0.9,"rationale_summary":"checked",'
                '"applied_memory_ids":["m1","invented"]}'
            ),
            model="fake",
        )


def test_solver_parser_filters_hallucinated_memory_ids() -> None:
    output = parse_solver_output('{"answer":"x","applied_memory_ids":["valid","fake"]}', ["valid"])
    assert output.applied_memory_ids == ["valid"]


@pytest.mark.asyncio
async def test_memory_agent_injects_retrieved_experience(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    store.save_memory(
        MemoryItem(
            memory_id="m1",
            status=MemoryStatus.ACTIVE,
            trigger="addition arithmetic",
            directive="Add operands carefully.",
        )
    )
    client = CapturingClient()
    agent = MemoryAgent(client, ProviderConfig(model="fake"), MemoryManager(store))
    prediction = await agent.solve(
        BenchmarkSample(sample_id="s", prompt="addition: 2+2", reference="4"),
        PolicyGenome(top_k=1),
    )
    assert prediction.output.answer == "4"
    assert prediction.output.applied_memory_ids == ["m1"]
    assert len(prediction.retrieved) == 1
    request = client.requests[0]
    assert isinstance(request, GenerationRequest)
    user_payload = str(request.messages[-1]["content"])
    assert '"phase"' not in user_payload
    store.close()
