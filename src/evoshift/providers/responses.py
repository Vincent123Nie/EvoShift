from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Awaitable, Mapping
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from evoshift.config import ProviderConfig
from evoshift.errors import ProviderError
from evoshift.providers.base import ModelT, validate_json_text
from evoshift.providers.retry import full_jitter_delay, is_retryable_status, parse_retry_after
from evoshift.runtime.budget import BudgetLedger, BudgetReservation
from evoshift.runtime.cache import LLMCache
from evoshift.schemas import GenerationRequest, GenerationResponse, LLMUsage

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]
RandomValue = Callable[[], float]

_BEARER_PATTERN = re.compile(r"(?i)(authorization\s*[:=]?\s*bearer\s+)[^\s,;}\]]+")
_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def idempotency_key_for_payload(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"evoshift-{digest}"


def _endpoint_identity(base_url: str) -> str:
    """Return a cache-safe URL identity without credentials or query values."""

    parsed = urlsplit(base_url)
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme.lower()}://{hostname.lower()}{port}{parsed.path.rstrip('/')}"


class ResponsesClient:
    """OpenAI Responses-compatible async client with bounded retries.

    Direct HTTP keeps the adapter compatible with lightweight reverse proxies
    while the rest of EvoShift depends only on normalized request/response
    models. Request bodies and authorization values are never logged.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        api_key: str | None = None,
        cache: LLMCache | None = None,
        budget: BudgetLedger | None = None,
        http_client: httpx.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
        random_value: RandomValue = random.random,
        max_concurrency: int = 8,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self.config = config
        self.base_url = config.resolved_base_url()
        self.model = config.resolved_model()
        resolved_api_key = api_key if api_key is not None else os.getenv(config.api_key_env)
        self._api_key = (resolved_api_key or "").strip()
        if not self._api_key:
            raise ProviderError(f"missing API key in environment variable {config.api_key_env}")
        self._endpoint = f"{self.base_url}/responses"
        self._cache = cache
        self._budget = budget
        self._sleep = sleep
        self._random_value = random_value
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_seconds),
            follow_redirects=False,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "EvoShift/0.1",
            },
        )
        self._closed = False

    def __repr__(self) -> str:
        return (
            f"ResponsesClient(model={self.model!r}, base_url={_endpoint_identity(self.base_url)!r})"
        )

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._closed:
            raise ProviderError("Responses client is closed")
        payload = self._build_payload(request)
        idempotency_key = idempotency_key_for_payload(payload)
        cache_key = self._cache_key(payload)

        cached = self._cache_get(cache_key)
        if cached is not None:
            cached_usage = cached.usage.model_copy(
                update={"cached": True, "cost_usd": 0.0, "latency_ms": 0.0}
            )
            return cached.model_copy(update={"usage": cached_usage}, deep=True)

        estimated_input = self._estimate_input_tokens(payload)
        estimated_output = int(payload["max_output_tokens"])
        estimated_tokens = estimated_input + estimated_output
        estimated_cost = self._calculate_cost(estimated_input, estimated_output)

        async with self._semaphore:
            reservation: BudgetReservation | None = None
            if self._budget is not None:
                reservation = self._budget.reserve(
                    estimated_tokens=estimated_tokens,
                    estimated_cost_usd=estimated_cost,
                )
            try:
                response = await self._request_with_retries(payload, idempotency_key)
            except BaseException:
                if reservation is not None:
                    self._budget_abort(reservation)
                raise

            if reservation is not None:
                accounting_usage = response.usage
                if accounting_usage.total_tokens == 0:
                    accounting_usage = accounting_usage.model_copy(
                        update={
                            "input_tokens": estimated_input,
                            "output_tokens": estimated_output,
                            "total_tokens": estimated_tokens,
                            "cost_usd": estimated_cost,
                        }
                    )
                assert self._budget is not None
                self._budget.reconcile(reservation, accounting_usage)

        self._cache_put(cache_key, response)
        return response

    async def generate_json(
        self,
        request: GenerationRequest,
        schema: type[ModelT],
    ) -> ModelT:
        response = await self.generate(request)
        return validate_json_text(response.text, schema)

    async def _request_with_retries(
        self,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> GenerationResponse:
        started = time.perf_counter()
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                response = await self._client.post(
                    self._endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Idempotency-Key": idempotency_key,
                    },
                    json=payload,
                )
            except httpx.HTTPError:
                if attempt < attempts - 1:
                    delay = full_jitter_delay(
                        attempt,
                        base_seconds=self.config.retry_base_seconds,
                        random_value=self._random_value,
                    )
                    logger.warning(
                        "Responses request network failure; retry attempt=%d/%d delay=%.3fs",
                        attempt + 1,
                        attempts,
                        delay,
                    )
                    await self._sleep(delay)
                    continue
                raise ProviderError(
                    "Responses request failed after a network error",
                    retryable=True,
                ) from None

            status_code = response.status_code
            retryable = is_retryable_status(status_code)
            if retryable and attempt < attempts - 1:
                delay = full_jitter_delay(
                    attempt,
                    base_seconds=self.config.retry_base_seconds,
                    random_value=self._random_value,
                    retry_after=parse_retry_after(response.headers),
                )
                logger.warning(
                    "Responses request retryable status=%d attempt=%d/%d delay=%.3fs",
                    status_code,
                    attempt + 1,
                    attempts,
                    delay,
                )
                await self._sleep(delay)
                continue

            if not response.is_success:
                raise self._http_error(response, retryable=retryable)

            try:
                body = response.json()
            except ValueError:
                raise ProviderError(
                    f"Responses gateway returned non-JSON success (HTTP {status_code})",
                    status_code=status_code,
                ) from None
            if not isinstance(body, dict):
                raise ProviderError(
                    "Responses gateway returned a non-object JSON response",
                    status_code=status_code,
                )
            return self._normalize_response(
                body,
                requested_model=str(payload["model"]),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        raise ProviderError("Responses retry loop terminated unexpectedly")

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        normalized = request.model_dump(mode="json")
        model = request.model or self.model
        if not model:
            raise ProviderError("Responses model cannot be empty")
        if not request.messages:
            raise ProviderError("Responses request requires at least one message")
        payload: dict[str, Any] = {
            "model": model,
            "input": normalized["messages"],
            "max_output_tokens": request.max_output_tokens,
        }
        reasoning_effort = request.reasoning_effort
        if reasoning_effort is None:
            reasoning_effort = self.config.reasoning_effort
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if self.config.allow_sampling_params and request.temperature is not None:
            payload["temperature"] = request.temperature
        if self.config.send_metadata and request.metadata:
            payload["metadata"] = normalized["metadata"]
        return payload

    def _normalize_response(
        self,
        body: dict[str, Any],
        *,
        requested_model: str,
        latency_ms: float,
    ) -> GenerationResponse:
        text = self._extract_text(body)
        if not text:
            raise ProviderError("Responses gateway returned no usable text")
        usage = self._extract_usage(body.get("usage"), latency_ms=latency_ms)
        model_value = body.get("model")
        model = model_value if isinstance(model_value, str) and model_value else requested_model
        response_id_value = body.get("id")
        response_id = response_id_value if isinstance(response_id_value, str) else ""
        return GenerationResponse(
            text=text,
            model=model,
            usage=usage,
            response_id=response_id,
            raw=body,
        )

    @classmethod
    def _extract_text(cls, body: Mapping[str, Any]) -> str:
        output_text = body.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        chunks: list[str] = []
        output = body.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, Mapping):
                    continue
                content = item.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                    continue
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, Mapping):
                        continue
                    if part.get("type") not in {"output_text", "text"}:
                        continue
                    value = part.get("text")
                    if isinstance(value, str):
                        chunks.append(value)
        if chunks:
            return "\n".join(chunk for chunk in chunks if chunk).strip()

        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content.strip()
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                                chunks.append(str(part["text"]))
        return "\n".join(chunks).strip()

    def _extract_usage(self, value: Any, *, latency_ms: float) -> LLMUsage:
        usage = value if isinstance(value, Mapping) else {}
        input_tokens = self._nonnegative_int(usage.get("input_tokens"))
        if input_tokens == 0:
            input_tokens = self._nonnegative_int(usage.get("prompt_tokens"))
        output_tokens = self._nonnegative_int(usage.get("output_tokens"))
        if output_tokens == 0:
            output_tokens = self._nonnegative_int(usage.get("completion_tokens"))
        total_tokens = self._nonnegative_int(usage.get("total_tokens"))
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        raw_cost = usage.get("cost_usd", usage.get("cost"))
        cost = self._nonnegative_float(raw_cost)
        if cost is None:
            cost = self._calculate_cost(input_tokens, output_tokens)
        return LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=max(0.0, latency_ms),
        )

    @staticmethod
    def _nonnegative_int(value: Any) -> int:
        if isinstance(value, bool):
            return 0
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return 0
        return max(0, parsed)

    @staticmethod
    def _nonnegative_float(value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(parsed) or parsed < 0:
            return None
        return parsed

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.config.input_price_per_million
            + output_tokens * self.config.output_price_per_million
        ) / 1_000_000.0

    @staticmethod
    def _estimate_input_tokens(payload: Mapping[str, Any]) -> int:
        encoded = _canonical_json({"input": payload.get("input", [])})
        return max(1, (len(encoded) + 3) // 4)

    def _http_error(self, response: httpx.Response, *, retryable: bool) -> ProviderError:
        message = ""
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, Mapping):
            error = body.get("error")
            if isinstance(error, Mapping):
                value = error.get("message")
                if isinstance(value, str):
                    message = value
            elif isinstance(error, str):
                message = error
        sanitized = self._redact(message)[:500].strip()
        suffix = f": {sanitized}" if sanitized else ""
        return ProviderError(
            f"Responses gateway error (HTTP {response.status_code}){suffix}",
            status_code=response.status_code,
            retryable=retryable,
        )

    def _redact(self, text: str) -> str:
        redacted = text.replace(self._api_key, "[REDACTED]") if self._api_key else text
        redacted = _BEARER_PATTERN.sub(r"\1[REDACTED]", redacted)
        return _OPENAI_KEY_PATTERN.sub("[REDACTED]", redacted)

    def _cache_key(self, payload: Mapping[str, Any]) -> str | None:
        if self._cache is None:
            return None
        safe_payload = {
            "provider": "openai-responses",
            "endpoint": _endpoint_identity(self.base_url),
            "payload": payload,
        }
        try:
            return self._cache.make_key(safe_payload)
        except Exception:
            logger.warning("LLM cache key generation failed; cache bypassed")
            return None

    def _cache_get(self, key: str | None) -> GenerationResponse | None:
        if self._cache is None or key is None:
            return None
        try:
            return self._cache.get(key)
        except Exception:
            logger.warning("LLM cache read failed; treating request as a cache miss")
            return None

    def _cache_put(self, key: str | None, response: GenerationResponse) -> None:
        if self._cache is None or key is None:
            return
        try:
            self._cache.put(key, response)
        except Exception:
            logger.warning("LLM cache write failed; response remains usable")

    def _budget_abort(self, reservation: BudgetReservation) -> None:
        assert self._budget is not None
        try:
            self._budget.abort(reservation)
        except ValueError:
            logger.error("Budget reservation was already closed")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> ResponsesClient:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        await self.aclose()


OpenAIResponsesClient = ResponsesClient
