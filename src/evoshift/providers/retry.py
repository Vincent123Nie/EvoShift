from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable

RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES


def parse_retry_after(
    headers: Mapping[str, str],
    *,
    now: datetime | None = None,
) -> float | None:
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        seconds = float(raw.strip())
    except ValueError:
        try:
            target = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        seconds = (target - current).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def full_jitter_delay(
    retry_index: int,
    *,
    base_seconds: float,
    random_value: Callable[[], float],
    retry_after: float | None = None,
    maximum_seconds: float = 60.0,
) -> float:
    """Return exponential full jitter while respecting ``Retry-After``."""

    exponential_cap = max(0.0, base_seconds) * float(2 ** max(0, retry_index))
    cap = float(min(maximum_seconds, exponential_cap))
    fraction = min(1.0, max(0.0, float(random_value())))
    jitter = float(cap * fraction)
    if retry_after is None:
        return jitter
    return float(max(jitter, retry_after))
