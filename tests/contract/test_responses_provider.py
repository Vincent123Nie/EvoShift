from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel

from evoshift.config import ProviderConfig
from evoshift.errors import ProviderError
from evoshift.providers import (
    FakeLLMClient,
    HeuristicDemoClient,
    ResponsesClient,
    create_client,
)
from evoshift.runtime import BudgetLedger, SQLiteLLMCache
from evoshift.schemas import GenerationRequest

SECRET = "sk-local-contract-secret-123456"
BASE_URL = "https://gateway.example.test/v1"


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def provider_config(**updates: Any) -> ProviderConfig:
    values: dict[str, Any] = {
        "base_url": BASE_URL,
        "model": "gpt-test",
        "max_retries": 0,
        "max_output_tokens": 32,
    }
    values.update(updates)
    return ProviderConfig(**values)


def request() -> GenerationRequest:
    return GenerationRequest(
        messages=[{"role": "user", "content": "Reply only OK"}],
        max_output_tokens=32,
    )


def clear_provider_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVOSHIFT_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("EVOSHIFT_OPENAI_MODEL", raising=False)


def test_responses_request_contract_usage_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_provider_overrides(monkeypatch)
    body = {
        "id": "resp_123",
        "model": "gpt-test-2026",
        "output_text": "OK",
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    with respx.mock(assert_all_called=True) as router:
        route = router.post(f"{BASE_URL}/responses").mock(
            return_value=httpx.Response(200, json=body)
        )

        async def exercise() -> Any:
            async with ResponsesClient(
                provider_config(
                    input_price_per_million=2.0,
                    output_price_per_million=8.0,
                ),
                api_key=SECRET,
            ) as client:
                return await client.generate(request())

        response = run(exercise())

    assert response.text == "OK"
    assert response.model == "gpt-test-2026"
    assert response.response_id == "resp_123"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 15
    assert response.usage.cost_usd == pytest.approx(0.00006)
    sent = route.calls[0].request
    payload = json.loads(sent.content)
    assert payload == {
        "model": "gpt-test",
        "input": [{"role": "user", "content": "Reply only OK"}],
        "max_output_tokens": 32,
        "reasoning": {"effort": "low"},
    }
    assert sent.headers["authorization"] == f"Bearer {SECRET}"
    assert sent.headers["idempotency-key"].startswith("evoshift-")
    assert SECRET not in repr(response)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "first"},
                            {"type": "text", "text": "second"},
                        ],
                    }
                ]
            },
            "first\nsecond",
        ),
        (
            {
                "choices": [{"message": {"role": "assistant", "content": "legacy"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            },
            "legacy",
        ),
    ],
)
def test_responses_parses_compatible_shapes(
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, Any],
    expected: str,
) -> None:
    clear_provider_overrides(monkeypatch)
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{BASE_URL}/responses").mock(return_value=httpx.Response(200, json=body))

        async def exercise() -> str:
            async with ResponsesClient(provider_config(), api_key=SECRET) as client:
                result = await client.generate(request())
                return result.text

        assert run(exercise()) == expected


def test_retry_after_and_idempotency_are_stable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clear_provider_overrides(monkeypatch)
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    caplog.set_level(logging.WARNING)
    with respx.mock(assert_all_called=True) as router:
        route = router.post(f"{BASE_URL}/responses").mock(
            side_effect=[
                httpx.Response(
                    429,
                    headers={"Retry-After": "2"},
                    json={"error": {"message": f"rate limited {SECRET}"}},
                ),
                httpx.Response(200, json={"output_text": "OK"}),
            ]
        )

        async def exercise() -> str:
            async with ResponsesClient(
                provider_config(max_retries=1, retry_base_seconds=1.0),
                api_key=SECRET,
                sleep=record_sleep,
                random_value=lambda: 0.5,
            ) as client:
                return (await client.generate(request())).text

        assert run(exercise()) == "OK"

    assert sleeps == [2.0]
    assert len(route.calls) == 2
    keys = [call.request.headers["idempotency-key"] for call in route.calls]
    assert keys[0] == keys[1]
    assert SECRET not in caplog.text


def test_full_jitter_is_used_without_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_provider_overrides(monkeypatch)
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    with respx.mock(assert_all_called=True) as router:
        router.post(f"{BASE_URL}/responses").mock(
            side_effect=[
                httpx.Response(503, json={"error": {"message": "temporary"}}),
                httpx.Response(200, json={"output_text": "done"}),
            ]
        )

        async def exercise() -> None:
            async with ResponsesClient(
                provider_config(max_retries=1, retry_base_seconds=2.0),
                api_key=SECRET,
                sleep=record_sleep,
                random_value=lambda: 0.5,
            ) as client:
                await client.generate(request())

        run(exercise())

    assert sleeps == [1.0]


def test_remote_errors_and_logs_are_redacted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clear_provider_overrides(monkeypatch)
    other_secret = "sk-other-secret-999999"
    message = f"bad credential {SECRET}; Authorization: Bearer {other_secret}"
    caplog.set_level(logging.WARNING)
    with respx.mock(assert_all_called=True) as router:
        router.post(f"{BASE_URL}/responses").mock(
            return_value=httpx.Response(400, json={"error": {"message": message}})
        )

        async def exercise() -> None:
            async with ResponsesClient(provider_config(), api_key=SECRET) as client:
                await client.generate(request())

        with pytest.raises(ProviderError) as caught:
            run(exercise())

    rendered = str(caught.value)
    assert SECRET not in rendered
    assert other_secret not in rendered
    assert "[REDACTED]" in rendered
    assert SECRET not in caplog.text
    assert other_secret not in caplog.text


def test_cache_hit_bypasses_network_and_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clear_provider_overrides(monkeypatch)
    cache = SQLiteLLMCache(tmp_path / "cache.sqlite3", namespace="provider-test")
    budget = BudgetLedger(max_requests=1, max_total_tokens=1000, max_cost_usd=1.0)
    try:
        with respx.mock(assert_all_called=True) as router:
            route = router.post(f"{BASE_URL}/responses").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "output_text": "cached result",
                        "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
                    },
                )
            )

            async def exercise() -> Any:
                async with ResponsesClient(
                    provider_config(), api_key=SECRET, cache=cache, budget=budget
                ) as client:
                    first = await client.generate(request())
                    second = await client.generate(request())
                    return first, second

            first, second = run(exercise())

        assert first.usage.cached is False
        assert second.usage.cached is True
        assert second.usage.cost_usd == 0.0
        assert len(route.calls) == 1
        assert budget.snapshot().requests == 1
    finally:
        cache.close()


def test_fake_client_supports_handler_sequence_and_structured_json() -> None:
    class Answer(BaseModel):
        value: int

    async def exercise() -> None:
        handled = FakeLLMClient(handler=lambda req: f"seen:{req.messages[-1]['content']}")
        assert (await handled.generate(request())).text == "seen:Reply only OK"

        scripted = FakeLLMClient(responses=["one", lambda _: "two", {"value": 3}])
        assert (await scripted.generate(request())).text == "one"
        assert (await scripted.generate(request())).text == "two"
        parsed = await scripted.generate_json(request(), Answer)
        assert parsed.value == 3
        with pytest.raises(ProviderError, match="exhausted"):
            await scripted.generate(request())

    run(exercise())


def demo_request(marker: str, payload: dict[str, Any], purpose: str) -> GenerationRequest:
    return GenerationRequest(
        model="demo",
        messages=[
            {"role": "system", "content": "deterministic demo"},
            {
                "role": "user",
                "content": marker + "\n" + json.dumps(payload, sort_keys=True),
            },
        ],
        metadata={"purpose": purpose},
    )


def test_demo_client_requires_matching_experience_on_shifted_phases() -> None:
    cases = [
        (
            "Let F(x, y) = 2*x - y. Compute F(4, 1). Return only the integer.",
            "For affine tasks, evaluate exactly 2*x - y; do not substitute addition.",
            "5",
            "7",
        ),
        (
            "Given x=3 and y=2, return x+y if x is even; otherwise return x-y. "
            "Return only the integer.",
            "Branch on x parity: use x+y when x is even, otherwise use x-y.",
            "5",
            "1",
        ),
        (
            "Compute z = 3*x + y for x=-1, y=1. Return POS if z>0, ZERO if z=0, otherwise NEG.",
            "Compute z=3*x+y, then map its sign to POS, ZERO, or NEG exactly.",
            "-2",
            "NEG",
        ),
    ]

    async def exercise() -> None:
        client = HeuristicDemoClient(model="demo")
        addition = await client.generate(
            demo_request(
                "<EVOSHIFT_TASK>",
                {
                    "task": "Compute 2 + 3. Return only the integer.",
                    "experience_cards": "",
                },
                "solve",
            )
        )
        assert json.loads(addition.text)["answer"] == "5"

        for task, directive, initial, learned in cases:
            without_card = await client.generate(
                demo_request(
                    "<EVOSHIFT_TASK>",
                    {"task": task, "experience_cards": ""},
                    "solve",
                )
            )
            assert json.loads(without_card.text)["answer"] == initial

            card = (
                "[experience:mem-unrelated@v1]\nWhen: unrelated\nDo: Ignore this card.\n\n"
                f"[experience:mem-demo@v1]\nWhen: shifted task\nDo: {directive}"
            )
            with_card = await client.generate(
                demo_request(
                    "<EVOSHIFT_TASK>",
                    {"task": task, "experience_cards": card},
                    "solve",
                )
            )
            parsed = json.loads(with_card.text)
            assert parsed["answer"] == learned
            assert parsed["applied_memory_ids"] == ["mem-demo"]
            assert with_card.usage.total_tokens > 0

    run(exercise())


def test_demo_client_critic_and_self_refine_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_provider_overrides(monkeypatch)
    task = "Let F(x, y) = 2*x - y. Compute F(4, 1). Return only the integer."

    async def exercise() -> None:
        budget = BudgetLedger(max_requests=2, max_total_tokens=10_000, max_cost_usd=0.0)
        created = create_client(ProviderConfig(kind="demo", model="demo"), budget=budget)
        assert isinstance(created, HeuristicDemoClient)
        client = created
        critique = await client.generate(
            demo_request(
                "<EVOSHIFT_CRITIQUE>",
                {"task": task, "agent_answer": "5", "reward": 0.0},
                "experience_critic",
            )
        )
        memory = json.loads(critique.text)["memory"]
        assert memory["kind"] == "procedural"
        assert "2*x - y" in memory["directive"]

        refined = await client.generate(
            demo_request(
                "<EVOSHIFT_SELF_REFINE>",
                {"task": task, "draft": {"answer": "5"}, "experience_cards": ""},
                "self_refine",
            )
        )
        assert json.loads(refined.text)["answer"] == "7"
        snapshot = budget.snapshot()
        assert snapshot.requests == 2
        assert snapshot.total_tokens > 0

    run(exercise())
