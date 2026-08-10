import pytest

from evoshift.config import EvolutionConfig, ProviderConfig
from evoshift.evolution.critic import ExperienceCritic, extract_json_object
from evoshift.providers.demo import HeuristicDemoClient
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


@pytest.mark.asyncio
async def test_critic_persists_only_explicit_learner_visible_policy_tag() -> None:
    client = StubClient()
    critic = ExperienceCritic(client, ProviderConfig(model="fake"), EvolutionConfig())
    episode = _episode().model_copy(
        update={
            "sample": _episode().sample.model_copy(
                update={
                    "metadata": {
                        "learner_visible_memory_tag": (
                            "tau3_policy:cancel_reason_duplicate_order:allow"
                        ),
                        "valid_memory_tags": ["hidden-oracle-tag-not-allowed"],
                    }
                }
            )
        }
    )

    result = await critic.analyze(episode, [])

    assert result.proposed_memory is not None
    assert "tau3_policy:cancel_reason_duplicate_order:allow" in result.proposed_memory.tags
    assert "hidden-oracle-tag-not-allowed" not in result.proposed_memory.tags


@pytest.mark.asyncio
async def test_critic_cluster_key_ignores_model_tags_and_hidden_metadata() -> None:
    class MaliciousClient(StubClient):
        async def generate(self, request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(
                text=(
                    '{"failure_type":"reasoning_error","signature":"x",'
                    '"evidence":"x","memory":{"trigger":"t","scope":"s",'
                    '"directive":"d","evidence_cluster_key":"model-chosen"}}'
                ),
                model="fake",
            )

    base = _episode().model_copy(
        update={
            "sample": _episode().sample.model_copy(
                update={
                    "metadata": {
                        "feedback_source": "grader-a",
                        "feedback_context": "refund:any:days_8_14",
                        "feedback_reference": "ALLOW",
                        "oracle_policy_version": "v1",
                        "corruption": "clean",
                    }
                }
            )
        }
    )
    flipped = base.model_copy(
        update={
            "sample": base.sample.model_copy(
                update={
                    "metadata": {
                        **base.sample.metadata,
                        "oracle_policy_version": "v99",
                        "corruption": "attack",
                        "phase": "hidden-shift",
                    }
                }
            )
        }
    )
    critic = ExperienceCritic(MaliciousClient(), ProviderConfig(model="fake"), EvolutionConfig())

    first = await critic.analyze(base, [])
    second = await critic.analyze(flipped, [])

    assert first.proposed_memory is not None
    assert second.proposed_memory is not None
    assert first.proposed_memory.evidence_cluster_key == second.proposed_memory.evidence_cluster_key
    assert first.proposed_memory.evidence_cluster_key != "model-chosen"


@pytest.mark.asyncio
async def test_demo_paraphrase_and_cluster_key_ignore_hidden_oracle_fields() -> None:
    sample = BenchmarkSample(
        sample_id="refund-fixture",
        prompt=(
            "Refund case: customer_tier=STANDARD; request_day=10. Return APPROVE or DENY only."
        ),
        reference="APPROVE",
        domain="customer_support/refund_policy",
        metadata={
            "feedback_source": "customer_support_portal",
            "feedback_context": "refund:any:days_8_14",
            "feedback_reference": "APPROVE",
            "policy_version": "v2",
            "protected": False,
            "valid_memory_tags": ["policy_v2"],
        },
    )
    base = Episode(
        episode_id="visible-base",
        run_id="run",
        index=0,
        sample=sample,
        output=SolverOutput(answer="DENY"),
        score=ScoreBundle(primary=0.0, success=False),
        feedback_score=ScoreBundle(primary=0.0, success=False),
    )
    hidden_mutation = base.model_copy(
        update={
            "episode_id": "visible-mutated",
            "sample": sample.model_copy(
                update={
                    "reference": "DENY",
                    "metadata": {
                        **sample.metadata,
                        "policy_version": "v99",
                        "protected": True,
                        "valid_memory_tags": ["hidden-malicious-tag"],
                        "feedback_corrupted": True,
                    },
                }
            ),
        }
    )
    evolution = EvolutionConfig(
        shadow_candidate_enabled=True,
        future_audit_enabled=True,
        shadow_eprocess_enabled=True,
        shadow_hierarchical_eprocess_enabled=True,
    )
    critic = ExperienceCritic(
        HeuristicDemoClient(model="demo", critic_paraphrase_mode="stable_cycle"),
        ProviderConfig(model="demo"),
        evolution,
    )

    first = await critic.analyze(base, [])
    second = await critic.analyze(hidden_mutation, [])

    assert first.proposed_memory is not None
    assert second.proposed_memory is not None
    assert first.proposed_memory.evidence_cluster_key
    assert first.proposed_memory.evidence_cluster_key == second.proposed_memory.evidence_cluster_key
    assert first.proposed_memory.trigger == second.proposed_memory.trigger
    assert first.proposed_memory.directive == second.proposed_memory.directive
    assert first.proposed_memory.anti_pattern == second.proposed_memory.anti_pattern
