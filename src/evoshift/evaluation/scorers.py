"""Deterministic answer scorers used by benchmark adapters.

The scorers deliberately avoid model-as-a-judge calls.  This keeps the primary
benchmark signal reproducible and makes online adaptation costs auditable.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

from evoshift.schemas import BenchmarkSample, ScoreBundle

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff])")
_NUMBER_RE = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"(?:\s*/\s*[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?)?%?"
)
_EXPLICIT_CHOICE_RE = re.compile(
    r"(?:answer|option|choice)\s*(?:is|=|:)?\s*[\(\[]?([A-Z])(?:[\)\]])?\b",
    flags=re.IGNORECASE,
)
_BARE_CHOICE_RE = re.compile(r"^\s*[\(\[]?([A-Z])(?:[\)\]])?[\s\.:,-]*$", re.IGNORECASE)
_LEADING_CHOICE_RE = re.compile(r"^\s*[\(\[]?([A-Z])(?:[\)\]])?[\s\.:,-]+", re.IGNORECASE)
_TRAILING_PAREN_CHOICE_RE = re.compile(r"[\(\[]([A-Z])[\)\]][\s\.:,-]*$", re.IGNORECASE)
_BINARY_CHOICE_RE = re.compile(
    r"^\s*(?:(?:the\s+)?answer\s*(?:is|=|:)?\s*)?(yes|no|true|false)\b",
    re.IGNORECASE,
)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _reference_values(reference: Any) -> list[Any]:
    """Return accepted references while supporting common dataset shapes."""

    if isinstance(reference, Mapping):
        for key in ("answers", "aliases", "text"):
            value = reference.get(key)
            if isinstance(value, (list, tuple, set)):
                return list(value)
        for key in ("answer", "value", "label"):
            if key in reference:
                return _reference_values(reference[key])
    if isinstance(reference, (list, tuple, set)):
        return list(reference)
    return [reference]


def normalize_answer(value: Any) -> str:
    """Apply SQuAD-style normalization with Unicode punctuation handling."""

    text = unicodedata.normalize("NFKC", _as_text(value)).lower()
    text = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    text = _ARTICLE_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def exact_match_score(prediction: Any, reference: Any) -> float:
    """Return 1 when the stripped prediction exactly matches any reference."""

    predicted = _as_text(prediction).strip()
    return float(any(predicted == _as_text(item).strip() for item in _reference_values(reference)))


def exact_match(prediction: Any, reference: Any) -> float:
    """Alias for :func:`exact_match_score`."""

    return exact_match_score(prediction, reference)


def normalized_exact_match_score(prediction: Any, reference: Any) -> float:
    """Return 1 when normalized prediction matches any normalized reference."""

    predicted = normalize_answer(prediction)
    return float(any(predicted == normalize_answer(item) for item in _reference_values(reference)))


def normalized_exact_match(prediction: Any, reference: Any) -> float:
    """Alias for :func:`normalized_exact_match_score`."""

    return normalized_exact_match_score(prediction, reference)


def _answer_tokens(value: Any) -> list[str]:
    normalized = normalize_answer(value)
    # Splitting CJK characters avoids treating an entire Chinese sentence as one token.
    normalized = _CJK_RE.sub(r" \1 ", normalized)
    return normalized.split()


def _single_token_f1(prediction: Any, reference: Any) -> float:
    predicted_tokens = _answer_tokens(prediction)
    reference_tokens = _answer_tokens(reference)
    if not predicted_tokens and not reference_tokens:
        return 1.0
    if not predicted_tokens or not reference_tokens:
        return 0.0
    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def token_f1_score(prediction: Any, reference: Any) -> float:
    """Return maximum bag-of-tokens F1 over accepted references."""

    references = _reference_values(reference)
    if not references:
        return 0.0
    return max(_single_token_f1(prediction, item) for item in references)


def token_f1(prediction: Any, reference: Any) -> float:
    """Alias for :func:`token_f1_score`."""

    return token_f1_score(prediction, reference)


def _parse_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None

    text = unicodedata.normalize("NFKC", _as_text(value))
    matches = list(_NUMBER_RE.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(0).replace(",", "").replace(" ", "")
    percent = raw.endswith("%")
    if percent:
        raw = raw[:-1]
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            result = float(Fraction(numerator) / Fraction(denominator))
        else:
            result = float(raw)
    except (ValueError, ZeroDivisionError):
        return None
    if percent:
        result /= 100.0
    return result if math.isfinite(result) else None


def numeric_score(
    prediction: Any,
    reference: Any,
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-6,
) -> float:
    """Compare the last number in a response using configurable tolerances."""

    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("numeric tolerances must be non-negative")
    predicted = _parse_number(prediction)
    if predicted is None:
        return 0.0
    for item in _reference_values(reference):
        expected = _parse_number(item)
        if expected is not None and math.isclose(
            predicted,
            expected,
            abs_tol=absolute_tolerance,
            rel_tol=relative_tolerance,
        ):
            return 1.0
    return 0.0


def _choice_map(choices: Any | None) -> dict[str, str]:
    if choices is None:
        return {}
    if isinstance(choices, Mapping):
        return {str(key).strip().upper(): _as_text(value) for key, value in choices.items()}
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        return {chr(ord("A") + index): _as_text(value) for index, value in enumerate(choices)}
    raise TypeError("choices must be a mapping or a sequence")


def _extract_choice_label(value: Any, valid_labels: Iterable[str]) -> str | None:
    labels = {label.upper() for label in valid_labels}
    if isinstance(value, int) and not isinstance(value, bool):
        # Accept both common dataset conventions: 0 -> A and 1 -> A.  Ambiguous
        # integers are resolved by whichever label exists in the choice map.
        zero_based = chr(ord("A") + value) if value >= 0 else ""
        one_based = chr(ord("A") + value - 1) if value >= 1 else ""
        if zero_based in labels:
            return zero_based
        if one_based in labels:
            return one_based

    text = unicodedata.normalize("NFKC", _as_text(value)).strip()
    for pattern in (
        _EXPLICIT_CHOICE_RE,
        _BARE_CHOICE_RE,
        _LEADING_CHOICE_RE,
        _TRAILING_PAREN_CHOICE_RE,
    ):
        match = pattern.search(text)
        if match:
            label = match.group(1).upper()
            if not labels or label in labels:
                return label
    return None


def multiple_choice_score(
    prediction: Any,
    reference: Any,
    *,
    choices: Any | None = None,
) -> float:
    """Score A/B/C-style outputs, while also accepting exact choice text."""

    choice_map = _choice_map(choices)
    valid_labels = set(choice_map) or {chr(ord("A") + index) for index in range(26)}

    prediction_label = _extract_choice_label(prediction, valid_labels)
    predicted_normalized = normalize_answer(prediction)
    for item in _reference_values(reference):
        reference_label = _extract_choice_label(item, valid_labels)
        if reference_label is None and choice_map:
            normalized_reference = normalize_answer(item)
            reference_label = next(
                (
                    label
                    for label, text in choice_map.items()
                    if normalize_answer(text) == normalized_reference
                ),
                None,
            )
        if prediction_label is not None and prediction_label == reference_label:
            return 1.0

        normalized_reference = normalize_answer(item)
        if predicted_normalized == normalized_reference:
            return 1.0
        if (
            reference_label is not None
            and choice_map
            and predicted_normalized == normalize_answer(choice_map.get(reference_label, ""))
        ):
            return 1.0
    return 0.0


def _extract_binary_choice(value: Any) -> bool | None:
    text = unicodedata.normalize("NFKC", _as_text(value)).strip()
    match = _BINARY_CHOICE_RE.search(text)
    if match is None:
        return None
    return match.group(1).lower() in {"yes", "true"}


def binary_choice_score(prediction: Any, reference: Any) -> float:
    """Score an unambiguous leading Yes/No or True/False answer label."""

    predicted = _extract_binary_choice(prediction)
    if predicted is None:
        return 0.0
    for item in _reference_values(reference):
        expected = _extract_binary_choice(item)
        if expected is not None and predicted == expected:
            return 1.0
    return 0.0


def score_prediction(
    prediction: Any,
    reference: Any,
    evaluator: str = "exact_match",
    metadata: Mapping[str, Any] | None = None,
) -> ScoreBundle:
    """Dispatch a named deterministic evaluator and return a :class:`ScoreBundle`."""

    options = dict(metadata or {})
    name = evaluator.strip().lower().replace("-", "_")
    aliases = {
        "em": "exact_match",
        "exact": "exact_match",
        "normalized_em": "normalized_exact_match",
        "norm_em": "normalized_exact_match",
        "f1": "token_f1",
        "number": "numeric",
        "mcq": "multiple_choice",
        "choice": "multiple_choice",
        "binary": "binary_choice",
        "boolean": "binary_choice",
        "yes_no": "binary_choice",
    }
    name = aliases.get(name, name)

    if name == "exact_match":
        primary = exact_match_score(prediction, reference)
    elif name == "normalized_exact_match":
        primary = normalized_exact_match_score(prediction, reference)
    elif name == "token_f1":
        primary = token_f1_score(prediction, reference)
    elif name == "numeric":
        primary = numeric_score(
            prediction,
            reference,
            absolute_tolerance=float(
                options.get("absolute_tolerance", options.get("abs_tol", 1e-6))
            ),
            relative_tolerance=float(
                options.get("relative_tolerance", options.get("rel_tol", 1e-6))
            ),
        )
    elif name == "multiple_choice":
        primary = multiple_choice_score(prediction, reference, choices=options.get("choices"))
    elif name == "binary_choice":
        primary = binary_choice_score(prediction, reference)
    else:
        raise ValueError(f"unknown evaluator: {evaluator}")

    success_threshold = float(options.get("success_threshold", 1.0))
    if not 0.0 <= success_threshold <= 1.0:
        raise ValueError("success_threshold must be between 0 and 1")
    success = primary >= success_threshold
    return ScoreBundle(
        primary=primary,
        success=success,
        metrics={name: primary},
        feedback=f"{name}: {'pass' if success else 'fail'}",
    )


def score_sample(sample: BenchmarkSample, answer: str) -> ScoreBundle:
    """Score an answer using the evaluator and metadata declared by a sample."""

    return score_prediction(
        prediction=answer,
        reference=sample.reference,
        evaluator=sample.evaluator,
        metadata=sample.metadata,
    )


def score_feedback_sample(sample: BenchmarkSample, answer: str) -> ScoreBundle:
    """Score the feedback observable by adaptation, which may differ from oracle truth."""

    return score_prediction(
        prediction=answer,
        reference=sample.metadata.get("feedback_reference", sample.reference),
        evaluator=str(sample.metadata.get("feedback_evaluator", sample.evaluator)),
        metadata=sample.metadata,
    )


__all__ = [
    "binary_choice_score",
    "exact_match",
    "exact_match_score",
    "multiple_choice_score",
    "normalize_answer",
    "normalized_exact_match",
    "normalized_exact_match_score",
    "numeric_score",
    "score_feedback_sample",
    "score_prediction",
    "score_sample",
    "token_f1",
    "token_f1_score",
]
