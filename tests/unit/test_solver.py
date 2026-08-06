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
    assert '"sample_id"' not in user_payload
    assert request.metadata == {"purpose": "solve"}
    store.close()


@pytest.mark.asyncio
async def test_solver_request_is_invariant_to_hidden_benchmark_identity(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    client = CapturingClient()
    agent = MemoryAgent(client, ProviderConfig(model="fake"), MemoryManager(store))
    visible = {"prompt": "refund request on day 10", "domain": "customer_support"}

    await agent.solve(
        BenchmarkSample(
            sample_id="policy_shift:0:0000",
            reference="deny",
            phase="phase_0",
            metadata={"protected": True, "policy_version": "v1"},
            **visible,
        ),
        PolicyGenome(),
    )
    await agent.solve(
        BenchmarkSample(
            sample_id="policy_shift:5:9999",
            reference="approve",
            phase="phase_5",
            metadata={"protected": False, "policy_version": "v3"},
            **visible,
        ),
        PolicyGenome(),
    )

    first, second = client.requests
    assert isinstance(first, GenerationRequest)
    assert isinstance(second, GenerationRequest)
    assert first == second
    store.close()


@pytest.mark.asyncio
async def test_memory_agent_can_exclude_one_exact_memory_version(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    memory = MemoryItem(
        memory_id="m1",
        version=2,
        status=MemoryStatus.ACTIVE,
        trigger="addition arithmetic",
        directive="Add operands carefully.",
    )
    store.save_memory(memory)
    client = CapturingClient()
    agent = MemoryAgent(client, ProviderConfig(model="fake"), MemoryManager(store))

    prediction = await agent.solve(
        BenchmarkSample(sample_id="s", prompt="addition: 2+2", reference="4"),
        PolicyGenome(top_k=1),
        exclude_memory_versions=[("m1", 2)],
    )

    assert prediction.retrieved == []
    assert prediction.output.applied_memory_ids == []
    store.close()
