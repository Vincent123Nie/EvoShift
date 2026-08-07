from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, cast

from evoshift.errors import DatasetError

LONGMEMEVAL_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
LONGMEMEVAL_FILENAME = "longmemeval_s_cleaned.json"
LONGMEMEVAL_BYTES = 277_383_467
LONGMEMEVAL_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
LONGMEMEVAL_URLS = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
    f"{LONGMEMEVAL_REVISION}/{LONGMEMEVAL_FILENAME}",
    "https://hf-mirror.com/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
    f"{LONGMEMEVAL_REVISION}/{LONGMEMEVAL_FILENAME}",
)

_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_JSON_CHUNK_CHARS = 1024 * 1024
_MAX_RECORD_CHARS = 32 * 1024 * 1024


@dataclass(frozen=True)
class LongMemEvalSession:
    session_id: str
    date: str
    index_text: str
    display_text: str
    document_id: str = ""


@dataclass(frozen=True)
class LongMemEvalQuestion:
    question_id: str
    question_type: str
    question: str
    sessions: tuple[LongMemEvalSession, ...]
    answer_session_ids: frozenset[str]


@dataclass(frozen=True)
class DatasetIntegrity:
    path: Path
    size_bytes: int
    sha256: str


def default_longmemeval_path(root: Path) -> Path:
    return (
        root / "data" / "benchmarks" / "longmemeval" / LONGMEMEVAL_REVISION / LONGMEMEVAL_FILENAME
    )


def verify_longmemeval_file(path: Path) -> DatasetIntegrity:
    """Reject any payload that differs from the pinned cleaned S release."""

    candidate = Path(path)
    try:
        size = candidate.stat().st_size
    except OSError as exc:
        raise DatasetError(f"LongMemEval file is unavailable: {candidate}") from exc
    if not candidate.is_file():
        raise DatasetError(f"LongMemEval path is not a file: {candidate}")
    if size != LONGMEMEVAL_BYTES:
        raise DatasetError(
            f"LongMemEval byte-size mismatch: expected {LONGMEMEVAL_BYTES}, got {size}"
        )

    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_DOWNLOAD_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetError(f"failed to read LongMemEval file: {candidate}") from exc
    actual = digest.hexdigest()
    if actual != LONGMEMEVAL_SHA256:
        raise DatasetError(
            f"LongMemEval SHA-256 mismatch: expected {LONGMEMEVAL_SHA256}, got {actual}"
        )
    return DatasetIntegrity(path=candidate, size_bytes=size, sha256=actual)


def download_longmemeval(
    destination: Path,
    *,
    urls: Sequence[str] = LONGMEMEVAL_URLS,
    timeout_seconds: float = 120.0,
) -> DatasetIntegrity:
    """Download to a temporary file and publish only a verified payload."""

    target = Path(destination)
    if target.exists():
        return verify_longmemeval_file(target)
    if timeout_seconds <= 0:
        raise DatasetError("LongMemEval download timeout must be positive")
    if not urls:
        raise DatasetError("LongMemEval download requires at least one source URL")
    target.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for source_url in urls:
        temporary_path: Path | None = None
        try:
            request = urllib.request.Request(
                source_url,
                headers={"User-Agent": "EvoShift/0.1", "Accept": "application/octet-stream"},
            )
            with (
                contextlib.closing(
                    urllib.request.urlopen(request, timeout=timeout_seconds)
                ) as response,
                tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{target.name}.",
                    suffix=".download",
                    dir=target.parent,
                    delete=False,
                ) as temporary,
            ):
                temporary_path = Path(temporary.name)
                total = 0
                digest = hashlib.sha256()
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > LONGMEMEVAL_BYTES:
                        raise DatasetError("LongMemEval download exceeded the pinned byte size")
                    digest.update(chunk)
                    temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
            if total != LONGMEMEVAL_BYTES or digest.hexdigest() != LONGMEMEVAL_SHA256:
                raise DatasetError("LongMemEval download failed pinned size/SHA-256 verification")
            assert temporary_path is not None
            temporary_path.replace(target)
            return verify_longmemeval_file(target)
        except (OSError, DatasetError, urllib.error.URLError) as exc:
            failures.append(f"{source_url}: {type(exc).__name__}")
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    raise DatasetError("all LongMemEval download sources failed: " + "; ".join(failures))


def iter_longmemeval_questions(
    path: Path,
    *,
    include_abstention: bool = False,
    limit: int = 0,
    max_per_type: int = 0,
) -> Iterator[LongMemEvalQuestion]:
    """Stream validated questions without loading the 277 MB array into memory."""

    if limit < 0 or max_per_type < 0:
        raise DatasetError("LongMemEval selection limits cannot be negative")
    emitted = 0
    type_counts: dict[str, int] = {}
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for index, value in enumerate(_iter_json_array(handle)):
                question = _parse_question(value, index=index)
                if not include_abstention and question.question_id.endswith("_abs"):
                    continue
                if max_per_type and type_counts.get(question.question_type, 0) >= max_per_type:
                    continue
                type_counts[question.question_type] = type_counts.get(question.question_type, 0) + 1
                yield question
                emitted += 1
                if limit and emitted >= limit:
                    return
    except OSError as exc:
        raise DatasetError(f"failed to read LongMemEval dataset: {path}") from exc


def _iter_json_array(handle: TextIO) -> Iterator[Mapping[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False

    def compact_and_fill() -> None:
        nonlocal buffer, position, eof
        if position:
            buffer = buffer[position:]
            position = 0
        chunk = handle.read(_JSON_CHUNK_CHARS)
        if chunk:
            buffer += chunk
        else:
            eof = True

    def require_character() -> str:
        while position >= len(buffer) and not eof:
            compact_and_fill()
        if position >= len(buffer):
            raise DatasetError("LongMemEval JSON ended unexpectedly")
        return buffer[position]

    compact_and_fill()
    while require_character().isspace():
        position += 1
    if require_character() != "[":
        raise DatasetError("LongMemEval payload must be a top-level JSON array")
    position += 1
    first = True

    while True:
        while require_character().isspace():
            position += 1
        character = require_character()
        if character == "]":
            position += 1
            break
        if not first:
            if character != ",":
                raise DatasetError("LongMemEval JSON array is missing a comma")
            position += 1
            while require_character().isspace():
                position += 1
            if require_character() == "]":
                raise DatasetError("LongMemEval JSON array has a trailing comma")

        record_start = position
        while True:
            try:
                value, end = decoder.raw_decode(buffer, position)
                position = end
                break
            except json.JSONDecodeError as exc:
                if eof:
                    raise DatasetError("LongMemEval contains invalid JSON") from exc
                if len(buffer) - record_start > _MAX_RECORD_CHARS:
                    raise DatasetError("LongMemEval record exceeds the safety limit") from exc
                consumed = position
                compact_and_fill()
                record_start = max(0, record_start - consumed)
        if not isinstance(value, Mapping):
            raise DatasetError("LongMemEval array entries must be JSON objects")
        yield cast(Mapping[str, Any], value)
        first = False

    while True:
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position < len(buffer):
            raise DatasetError("LongMemEval JSON has trailing non-whitespace data")
        if eof:
            return
        compact_and_fill()


def _parse_question(value: Mapping[str, Any], *, index: int) -> LongMemEvalQuestion:
    question_id = _required_string(value, "question_id", index=index)
    question_type = _required_string(value, "question_type", index=index)
    question = _required_string(value, "question", index=index)
    session_ids = _string_sequence(value.get("haystack_session_ids"), "haystack_session_ids")
    dates = _string_sequence(value.get("haystack_dates"), "haystack_dates")
    raw_sessions = value.get("haystack_sessions")
    answer_ids = _string_sequence(value.get("answer_session_ids"), "answer_session_ids")
    if not isinstance(raw_sessions, Sequence) or isinstance(raw_sessions, (str, bytes)):
        raise DatasetError(f"LongMemEval record {index} has invalid haystack_sessions")
    if not session_ids or len(session_ids) != len(dates) or len(session_ids) != len(raw_sessions):
        raise DatasetError(f"LongMemEval record {index} has misaligned session fields")
    if not answer_ids:
        if not question_id.endswith("_abs"):
            raise DatasetError(f"LongMemEval record {index} has no evidence sessions")
    elif not set(answer_ids).issubset(session_ids):
        raise DatasetError(f"LongMemEval record {index} has unknown evidence session IDs")

    sessions = tuple(
        _parse_session(
            session_id,
            date,
            raw_session,
            record_index=index,
            session_index=session_index,
        )
        for session_index, (session_id, date, raw_session) in enumerate(
            zip(session_ids, dates, raw_sessions)
        )
    )
    return LongMemEvalQuestion(
        question_id=question_id,
        question_type=question_type,
        question=question,
        sessions=sessions,
        answer_session_ids=frozenset(answer_ids),
    )


def _parse_session(
    session_id: str,
    date: str,
    raw_session: object,
    *,
    record_index: int,
    session_index: int,
) -> LongMemEvalSession:
    if not isinstance(raw_session, Sequence) or isinstance(raw_session, (str, bytes)):
        raise DatasetError(f"LongMemEval record {record_index} contains an invalid session")
    user_contents: list[str] = []
    rendered_turns: list[str] = []
    for turn in raw_session:
        if not isinstance(turn, Mapping):
            raise DatasetError(f"LongMemEval record {record_index} contains an invalid turn")
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise DatasetError(f"LongMemEval record {record_index} contains a malformed turn")
        if role == "user":
            user_contents.append(content)
        rendered_turns.append(f"{str(role).title()}: {content}")
    if not rendered_turns:
        raise DatasetError(f"LongMemEval record {record_index} contains an empty session")
    return LongMemEvalSession(
        session_id=session_id,
        date=date,
        index_text=" ".join(user_contents),
        display_text=f"Date: {date}\n" + "\n".join(rendered_turns),
        document_id=f"d{session_index:04d}",
    )


def _required_string(value: Mapping[str, Any], key: str, *, index: int) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise DatasetError(f"LongMemEval record {index} has invalid {key}")
    return item


def _string_sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DatasetError(f"LongMemEval {field} must be a string array")
    normalized = tuple(item for item in value if isinstance(item, str) and item)
    if len(normalized) != len(value):
        raise DatasetError(f"LongMemEval {field} contains a non-string or empty value")
    return normalized


__all__ = [
    "LONGMEMEVAL_BYTES",
    "LONGMEMEVAL_FILENAME",
    "LONGMEMEVAL_REVISION",
    "LONGMEMEVAL_SHA256",
    "DatasetIntegrity",
    "LongMemEvalQuestion",
    "LongMemEvalSession",
    "default_longmemeval_path",
    "download_longmemeval",
    "iter_longmemeval_questions",
    "verify_longmemeval_file",
]
