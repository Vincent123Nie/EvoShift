from __future__ import annotations

from evoshift.config import ProviderConfig
from evoshift.errors import ConfigurationError
from evoshift.providers.base import LLMClient
from evoshift.providers.demo import HeuristicDemoClient
from evoshift.providers.fake import FakeLLMClient
from evoshift.providers.responses import ResponsesClient
from evoshift.runtime.budget import BudgetLedger
from evoshift.runtime.cache import LLMCache


def create_client(
    provider_config: ProviderConfig,
    cache: LLMCache | None = None,
    budget: BudgetLedger | None = None,
) -> LLMClient:
    """Create the configured provider without exposing credentials to callers."""

    kind = provider_config.kind.strip().lower().replace("-", "_")
    if kind in {"responses", "openai", "openai_responses"}:
        return ResponsesClient(provider_config, cache=cache, budget=budget)
    if kind in {"fake", "scripted"}:
        return FakeLLMClient(model=provider_config.resolved_model())
    if kind == "demo":
        return HeuristicDemoClient(model=provider_config.resolved_model(), budget=budget)
    raise ConfigurationError(f"unsupported provider kind: {provider_config.kind}")
