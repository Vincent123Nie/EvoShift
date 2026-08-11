from __future__ import annotations

from scripts.audit_critic_paraphrase_common_response import _foreground_match


def _prediction(
    sample_id: str,
    answer: str,
    *,
    cached: bool,
) -> dict[str, object]:
    return {
        "sample": {"sample_id": sample_id},
        "output": {
            "answer": answer,
            "confidence": 0.9,
            "rationale_summary": "bounded",
            "applied_memory_ids": [],
        },
        "score": {"primary": float(answer == "APPROVE"), "success": answer == "APPROVE"},
        "usage": {"cached": cached},
    }


def test_foreground_match_reports_common_response_coverage_and_divergence() -> None:
    exact = [
        _prediction("sample-1", "APPROVE", cached=False),
        _prediction("sample-2", "DENY", cached=True),
    ]
    hierarchical = [
        _prediction("sample-1", "APPROVE", cached=True),
        _prediction("sample-2", "APPROVE", cached=True),
    ]

    report = _foreground_match(exact, hierarchical)

    assert report == {
        "n": 2,
        "exact_n": 2,
        "hierarchical_n": 2,
        "sample_ids_aligned": True,
        "identical_output_score_n": 1,
        "identical_output_score_rate": 0.5,
        "first_divergence_index": 1,
        "exact_cache_hit_n": 1,
        "hierarchical_cache_hit_n": 2,
        "hierarchical_cache_hit_rate": 1.0,
    }


def test_foreground_match_rejects_unaligned_stream_lengths() -> None:
    report = _foreground_match(
        [_prediction("sample-1", "APPROVE", cached=False)],
        [],
    )

    assert report["sample_ids_aligned"] is False
    assert report["exact_n"] == 1
    assert report["hierarchical_n"] == 0
