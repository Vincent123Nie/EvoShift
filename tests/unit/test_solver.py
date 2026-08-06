from __future__ import annotations

from pathlib import Path

import pytest

from evoshift.agents.solver import MemoryAgent, parse_rerank_output, parse_solver_output
from evoshift.config import ProviderConfig
from evoshift.errors import ProviderError
from evoshift.memory import MemoryManager
from evoshift.schemas import (
    BenchmarkSample,
    GenerationRequest,
    GenerationResponse,
    LLMUsage,
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
            usage=LLMUsage(cached=True),
        )


def test_solver_parser_filters_hallucinated_memory_ids() -> None:
    output = parse_solver_output('{"answer":"x","applied_memory_ids":["valid","fake"]}', ["valid"])
    assert output.applied_memory_ids == ["valid"]


def test_rerank_parser_filters_injected_and_duplicate_memory_ids() -> None:
    ranked = parse_rerank_output(
        '{"ranked_memory_ids":["v3","invented","v3","v2"]}',
        ["v2", "v3"],
    )

    assert ranked == ["v3", "v2"]
    assert parse_rerank_output('{"other":[]}', ["v3"]) is None


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
    assert prediction.usage.cached is True
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


class RerankClient:
    def __init__(self, rerank_response: str | BaseException) -> None:
        self.rerank_response = rerank_response
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        if request.metadata.get("purpose") == "memory_rerank":
            if isinstance(self.rerank_response, BaseException):
                raise self.rerank_response
            return GenerationResponse(
                text=self.rerank_response,
                model="fake",
                usage=LLMUsage(input_tokens=6, output_tokens=4, total_tokens=10),
            )
        return GenerationResponse(
            text=(
                '{"answer":"APPROVE","confidence":0.9,'
                '"rationale_summary":"used selected policy","applied_memory_ids":["v3"]}'
            ),
            model="fake",
            usage=LLMUsage(input_tokens=12, output_tokens=8, total_tokens=20),
        )


def _retrieval_memory(memory_id: str, trigger: str, directive: str) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        status=MemoryStatus.ACTIVE,
        trigger=trigger,
        scope="customer_support/refund_policy",
        directive=directive,
        source_domains=["customer_support/refund_policy"],
    )


@pytest.mark.asyncio
async def test_memory_agent_reranks_allowlisted_candidates_and_accounts_usage(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    store.save_memory(
        _retrieval_memory(
            "distractor",
            "business account refund requests",
            "Deny business-account returns.",
        )
    )
    store.save_memory(
        _retrieval_memory(
            "v2",
            "refund request after day 7 through day 14",
            "Approve requests through day 14.",
        )
    )
    store.save_memory(
        _retrieval_memory(
            "v3",
            "premium customer refund request after day 14 through day 30",
            "Approve premium customers through day 30.",
        )
    )
    client = RerankClient('{"ranked_memory_ids":["v3","invented","v2"]}')
    agent = MemoryAgent(client, ProviderConfig(model="fake"), MemoryManager(store))
    sample = BenchmarkSample(
        sample_id="hidden-id",
        prompt="A VIP buyer requests reimbursement 23 days after purchase.",
        reference="APPROVE",
        domain="customer-support/refund-policy",
        phase="hidden-phase",
        metadata={"policy_version": "hidden-v3", "protected": True},
    )

    prediction = await agent.solve(
        sample,
        PolicyGenome(
            top_k=1,
            llm_rerank_enabled=True,
            llm_rerank_candidate_k=3,
        ),
    )

    assert [item.item.memory_id for item in prediction.retrieved] == ["v3"]
    assert prediction.output.applied_memory_ids == ["v3"]
    assert prediction.usage.total_tokens == 30
    assert prediction.rerank_attempted is True
    assert prediction.rerank_applied is True
    assert prediction.rerank_fallback is False
    rerank_request = client.requests[0]
    assert rerank_request.metadata == {"purpose": "memory_rerank"}
    rerank_payload = str(rerank_request.messages[-1]["content"])
    assert "hidden-id" not in rerank_payload
    assert "hidden-phase" not in rerank_payload
    assert "hidden-v3" not in rerank_payload
    assert "protected" not in rerank_payload
    assert "invented" not in [item.item.memory_id for item in prediction.retrieved]
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rerank_response",
    ["not-json", ProviderError("reranker unavailable")],
)
async def test_memory_agent_rerank_falls_back_to_sparse_order(
    tmp_path: Path,
    rerank_response: str | BaseException,
) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    store.save_memory(_retrieval_memory("first", "refund reimbursement", "First sparse result."))
    store.save_memory(_retrieval_memory("second", "refund policy", "Second sparse result."))
    client = RerankClient(rerank_response)
    agent = MemoryAgent(client, ProviderConfig(model="fake"), MemoryManager(store))

    prediction = await agent.solve(
        BenchmarkSample(
            sample_id="s",
            prompt="refund reimbursement",
            reference="APPROVE",
            domain="customer_support/refund_policy",
        ),
        PolicyGenome(top_k=1, llm_rerank_enabled=True, llm_rerank_candidate_k=2),
    )

    assert [item.item.memory_id for item in prediction.retrieved] == ["first"]
    assert prediction.rerank_attempted is True
    assert prediction.rerank_applied is False
    assert prediction.rerank_fallback is True
    store.close()


@pytest.mark.asyncio
async def test_memory_agent_rerank_preserves_provenance_domain_gate(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    store.save_memory(_retrieval_memory("refund-a", "refund reimbursement", "Refund rule A."))
    store.save_memory(_retrieval_memory("refund-b", "refund eligibility", "Refund rule B."))
    store.save_memory(
        _retrieval_memory("hr-secret", "refund reimbursement", "HR-only rule.").model_copy(
            update={"source_domains": ["human_resources/leave_policy"]}
        )
    )
    client = RerankClient('{"ranked_memory_ids":["hr-secret","refund-b"]}')
    agent = MemoryAgent(client, ProviderConfig(model="fake"), MemoryManager(store))

    prediction = await agent.solve(
        BenchmarkSample(
            sample_id="s",
            prompt="refund reimbursement",
            reference="APPROVE",
            domain="customer_support/refund_policy",
        ),
        PolicyGenome(top_k=1, llm_rerank_enabled=True, llm_rerank_candidate_k=3),
    )

    rerank_payload = str(client.requests[0].messages[-1]["content"])
    assert "hr-secret" not in rerank_payload
    assert [item.item.memory_id for item in prediction.retrieved] == ["refund-b"]
    store.close()
