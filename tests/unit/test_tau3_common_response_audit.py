from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_tau3_hierarchical_common_response import _paired_alignment


def _write_predictions(run_dir: Path, rows: list[dict[str, object]]) -> None:
    run_dir.mkdir()
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    (run_dir / "predictions.jsonl").write_text(payload, encoding="utf-8")


def _prediction(
    sample_id: str,
    answer: str,
    *,
    cached: bool,
) -> dict[str, object]:
    return {
        "sample": {"sample_id": sample_id},
        "output": {"answer": answer},
        "score": {"primary": float(answer == "APPROVE")},
        "usage": {"cached": cached},
    }


def test_paired_alignment_reports_common_responses_and_first_divergence(tmp_path: Path) -> None:
    exact_dir = tmp_path / "exact"
    hierarchical_dir = tmp_path / "hierarchical"
    _write_predictions(
        exact_dir,
        [
            _prediction("sample-1", "APPROVE", cached=False),
            _prediction("sample-2", "DENY", cached=False),
        ],
    )
    _write_predictions(
        hierarchical_dir,
        [
            _prediction("sample-1", "APPROVE", cached=True),
            _prediction("sample-2", "APPROVE", cached=True),
        ],
    )

    report = _paired_alignment(exact_dir, hierarchical_dir)

    assert report == {
        "aligned": True,
        "n": 2,
        "exact_n": 2,
        "hierarchical_n": 2,
        "identical_output_score_n": 1,
        "identical_output_score_rate": 0.5,
        "first_divergence_index": 1,
        "hierarchical_cache_hit_n": 2,
    }


def test_paired_alignment_rejects_different_stream_lengths(tmp_path: Path) -> None:
    exact_dir = tmp_path / "exact"
    hierarchical_dir = tmp_path / "hierarchical"
    _write_predictions(exact_dir, [_prediction("sample-1", "APPROVE", cached=False)])
    _write_predictions(hierarchical_dir, [])

    report = _paired_alignment(exact_dir, hierarchical_dir)

    assert report["aligned"] is False
    assert report["exact_n"] == 1
    assert report["hierarchical_n"] == 0
    assert report["first_divergence_index"] == 0
