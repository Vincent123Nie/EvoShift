from __future__ import annotations

import io
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from evoshift.benchmarks import longmemeval as dataset_module
from evoshift.benchmarks.longmemeval import (
    LongMemEvalQuestion,
    LongMemEvalSession,
    download_longmemeval,
    iter_longmemeval_questions,
    verify_longmemeval_file,
)
from evoshift.config import ProviderConfig
from evoshift.errors import DatasetError, ProviderError
from evoshift.evaluation import longmemeval_retrieval as retrieval_module
from evoshift.evaluation.longmemeval_retrieval import (
    LongMemEvalBM25,
    LongMemEvalLLMReranker,
    bound_candidate_text,
    fuse_rankings,
    parse_candidate_ranking,
    retrieval_metrics_for_ranking,
    run_longmemeval_retrieval,
    write_retrieval_artifacts,
)
from evoshift.providers.fake import FakeLLMClient
from evoshift.schemas import GenerationResponse, LLMUsage


def _record(
    question_id: str,
    question_type: str,
    *,
    evidence_index: int = 0,
) -> dict[str, Any]:
    session_ids = [f"{question_id}-session-0", f"{question_id}-session-1"]
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": "Where was the project discussed?",
        "answer": "In Paris",
        "haystack_session_ids": session_ids,
        "haystack_dates": ["2025-01-01", "2025-01-02"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "We discussed the project in Paris."},
                {"role": "assistant", "content": "I will remember that."},
            ],
            [
                {"role": "user", "content": "The weather was clear."},
                {"role": "assistant", "content": "That sounds pleasant."},
            ],
        ],
        "answer_session_ids": [session_ids[evidence_index]],
    }


def _write_fixture(path: Path) -> Path:
    abstention = _record("q-abs_abs", "single")
    abstention["answer_session_ids"] = []
    path.write_text(
        json.dumps(
            [
                _record("q-1", "single"),
                _record("q-2", "single", evidence_index=1),
                _record("q-3", "temporal"),
                abstention,
            ]
        ),
        encoding="utf-8",
    )
    return path


def _patch_integrity_constants(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    import hashlib

    monkeypatch.setattr(dataset_module, "LONGMEMEVAL_BYTES", len(payload))
    monkeypatch.setattr(dataset_module, "LONGMEMEVAL_SHA256", hashlib.sha256(payload).hexdigest())


def test_pinned_integrity_and_download_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"pinned-test-payload"
    _patch_integrity_constants(monkeypatch, payload)
    source = tmp_path / "source.json"
    source.write_bytes(payload)

    integrity = verify_longmemeval_file(source)
    assert integrity.size_bytes == len(payload)
    source.write_bytes(payload + b"tamper")
    with pytest.raises(DatasetError, match="byte-size mismatch"):
        verify_longmemeval_file(source)

    monkeypatch.setattr(
        dataset_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )
    destination = tmp_path / "download" / "dataset.json"
    downloaded = download_longmemeval(destination, urls=["https://example.invalid/data"])
    assert downloaded.sha256 == dataset_module.LONGMEMEVAL_SHA256
    assert destination.read_bytes() == payload
    assert not list(destination.parent.glob("*.download"))


def test_streaming_loader_validates_and_selects_without_label_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _write_fixture(tmp_path / "tiny.json")
    monkeypatch.setattr(dataset_module, "_JSON_CHUNK_CHARS", 7)
    questions = list(iter_longmemeval_questions(dataset, max_per_type=1))

    assert [question.question_id for question in questions] == ["q-1", "q-3"]
    assert questions[0].sessions[0].index_text == "We discussed the project in Paris."
    assert questions[0].sessions[0].document_id == "d0000"
    assert "Assistant: I will remember that." in questions[0].sessions[0].display_text
    assert "has_answer" not in questions[0].sessions[0].display_text

    malformed = json.loads(dataset.read_text(encoding="utf-8"))
    malformed[0]["haystack_dates"] = []
    dataset.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(DatasetError, match="misaligned session fields"):
        list(iter_longmemeval_questions(dataset, limit=1))


def test_streaming_loader_assigns_unique_documents_to_duplicate_upstream_ids(
    tmp_path: Path,
) -> None:
    record = _record("q-duplicate", "single")
    record["haystack_session_ids"][1] = record["haystack_session_ids"][0]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps([record]), encoding="utf-8")

    question = next(iter_longmemeval_questions(path))
    assert question.sessions[0].session_id == question.sessions[1].session_id
    assert question.sessions[0].document_id != question.sessions[1].document_id


def test_bm25_metrics_and_bounded_text_match_frozen_contract() -> None:
    retriever = LongMemEvalBM25(
        ["project meeting paris", "weather clear", "project schedule review"]
    )
    ranking = retriever.rank("project paris")
    assert ranking[0] == 0

    metrics = retrieval_metrics_for_ranking(("noise", "e1", "e2"), frozenset({"e1", "e2"}))
    assert metrics["recall_all@5"] == 1.0
    assert metrics["recall_all@10"] == 1.0
    assert metrics["ndcg_any@5"] == pytest.approx((1.0 + 1.0 / math.log2(3)) / 2.0)
    assert metrics["mrr"] == 0.5

    bounded = bound_candidate_text("A" * 300 + "B" * 300, 256)
    assert len(bounded) == 256
    assert bounded.startswith("A") and bounded.endswith("B")
    assert "truncated" in bounded


def test_rerank_parser_rejects_unknown_and_duplicate_ids() -> None:
    assert (
        parse_candidate_ranking(
            '{"ranked_candidate_ids":["c01","injected","c01","c00"]}',
            ["c00", "c01"],
        )
        is None
    )
    assert parse_candidate_ranking('{"ranked_candidate_ids":["c01","c01"]}', ["c00", "c01"]) is None
    assert parse_candidate_ranking('{"ranked_candidate_ids":["c01","c00"]}', ["c00", "c01"]) == (
        "c01",
        "c00",
    )
    assert parse_candidate_ranking("not json", ["c00"]) is None
    assert parse_candidate_ranking('{"wrong":[]}', ["c00"]) is None


def test_rank_fusion_is_deterministic_and_fail_closed() -> None:
    assert fuse_rankings(
        ("a", "b", "c"),
        ("c", "a", "b"),
        bm25_rank_weight=0.4,
    ) == ("a", "c", "b")
    with pytest.raises(ValueError, match="identical document permutations"):
        fuse_rankings(("a", "b"), ("a", "injected"), bm25_rank_weight=0.4)
    with pytest.raises(ValueError, match="between zero and one"):
        fuse_rankings(("a",), ("a",), bm25_rank_weight=1.1)


@pytest.mark.asyncio
async def test_llm_reranker_is_allowlisted_oracle_isolated_and_backfills() -> None:
    question = LongMemEvalQuestion(
        question_id="hidden-question-id",
        question_type="hidden-question-type",
        question="Which session matters?",
        sessions=(
            LongMemEvalSession("answer-secret-id", "2025-01-01", "alpha", "User: alpha"),
            LongMemEvalSession("other-secret-id", "2025-01-02", "beta", "User: beta"),
            LongMemEvalSession("third-secret-id", "2025-01-03", "gamma", "User: gamma"),
        ),
        answer_session_ids=frozenset({"answer-secret-id"}),
    )
    client = FakeLLMClient(
        responses=[
            GenerationResponse(
                text='{"ranked_candidate_ids":["c01"]}',
                model="test-model",
                usage=LLMUsage(total_tokens=42),
            )
        ]
    )
    reranker = LongMemEvalLLMReranker(
        client,
        ProviderConfig(kind="fake", model="test-model"),
        candidate_k=3,
        output_k=2,
        max_candidate_chars=256,
        max_total_candidate_chars=768,
    )
    baseline = ("answer-secret-id", "other-secret-id", "third-secret-id")
    outcome = await reranker.rerank(question, baseline)

    assert outcome.ranking == ("other-secret-id", "answer-secret-id", "third-secret-id")
    assert outcome.applied is True
    assert set(outcome.ranking) == set(baseline)
    prompt = str(client.calls[0].messages[-1]["content"])
    assert "hidden-question-id" not in prompt
    assert "hidden-question-type" not in prompt
    assert "answer-secret-id" not in prompt
    assert "answer_session_ids" not in prompt
    assert "Which session matters?" in prompt
    assert client.calls[0].metadata == {"purpose": "longmemeval_retrieval_rerank"}


@pytest.mark.asyncio
async def test_gold_label_permutation_cannot_change_request_or_ranking() -> None:
    original = LongMemEvalQuestion(
        question_id="answer-containing-question-id",
        question_type="knowledge-update",
        question="What changed?",
        sessions=(
            LongMemEvalSession("answer_session_a", "d1", "alpha", "User: alpha"),
            LongMemEvalSession("noans_session_b", "d2", "beta", "User: beta"),
        ),
        answer_session_ids=frozenset({"answer_session_a"}),
    )
    permuted = replace(original, answer_session_ids=frozenset({"noans_session_b"}))
    client = FakeLLMClient(
        responses=[
            '{"ranked_candidate_ids":["c01"]}',
            '{"ranked_candidate_ids":["c01"]}',
        ]
    )
    reranker = LongMemEvalLLMReranker(
        client,
        ProviderConfig(kind="fake", model="test"),
        candidate_k=2,
        output_k=1,
        max_candidate_chars=256,
        max_total_candidate_chars=512,
    )
    first = await reranker.rerank(original, ("answer_session_a", "noans_session_b"))
    second = await reranker.rerank(permuted, ("answer_session_a", "noans_session_b"))

    assert client.calls[0].messages == client.calls[1].messages
    assert first.ranking == second.ranking
    assert first.ranking == ("noans_session_b", "answer_session_a")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        '{"ranked_candidate_ids":["c01","unknown"]}',
        '{"ranked_candidate_ids":["c01","c01"]}',
        ProviderError("unavailable"),
    ],
)
async def test_llm_reranker_failure_preserves_exact_bm25_order(
    response: str | BaseException,
) -> None:
    question = LongMemEvalQuestion(
        question_id="q",
        question_type="single",
        question="question",
        sessions=(
            LongMemEvalSession("a", "d1", "alpha", "User: alpha"),
            LongMemEvalSession("b", "d2", "beta", "User: beta"),
        ),
        answer_session_ids=frozenset({"a"}),
    )
    client = FakeLLMClient(responses=[response])
    reranker = LongMemEvalLLMReranker(
        client,
        ProviderConfig(kind="fake", model="test"),
        candidate_k=2,
        output_k=1,
        max_candidate_chars=256,
        max_total_candidate_chars=512,
    )
    outcome = await reranker.rerank(question, ("b", "a"))
    assert outcome.ranking == ("b", "a")
    assert outcome.fallback is True
    assert outcome.applied is False


@pytest.mark.asyncio
async def test_rank_fusion_failure_preserves_exact_bm25_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = LongMemEvalQuestion(
        question_id="q-fusion",
        question_type="single",
        question="question",
        sessions=(
            LongMemEvalSession("a", "d1", "alpha", "User: alpha"),
            LongMemEvalSession("b", "d2", "beta", "User: beta"),
        ),
        answer_session_ids=frozenset({"a"}),
    )
    client = FakeLLMClient(responses=['{"ranked_candidate_ids":["c01"]}'])
    reranker = LongMemEvalLLMReranker(
        client,
        ProviderConfig(kind="fake", model="test"),
        candidate_k=2,
        output_k=1,
        max_candidate_chars=256,
        max_total_candidate_chars=512,
        bm25_rank_weight=0.4,
    )

    def fail_fusion(*_args: Any, **_kwargs: Any) -> tuple[str, ...]:
        raise ValueError("synthetic fusion invariant")

    monkeypatch.setattr(retrieval_module, "fuse_rankings", fail_fusion)
    outcome = await reranker.rerank(question, ("b", "a"))

    assert outcome.ranking == ("b", "a")
    assert outcome.fallback is True
    assert outcome.fallback_reason == "fusion_invariant"
    assert outcome.applied is False


@pytest.mark.asyncio
async def test_tiny_end_to_end_evaluation_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _write_fixture(tmp_path / "tiny.json")
    monkeypatch.setattr(retrieval_module, "verify_longmemeval_file", lambda _path: None)
    client = FakeLLMClient(responses=['{"ranked_candidate_ids":["c01","c00"]}'] * 2)
    evaluation = await run_longmemeval_retrieval(
        dataset,
        method="bm25_llm_rerank_fused",
        client=client,
        provider=ProviderConfig(kind="fake", model="test"),
        max_per_type=1,
        concurrency=2,
        candidate_k=2,
        output_k=2,
        max_candidate_chars=256,
        max_total_candidate_chars=512,
        bm25_rank_weight=0.4,
    )

    assert evaluation.report["n_questions"] == 2
    assert evaluation.report["selection"]["question_type_counts"] == {
        "single": 1,
        "temporal": 1,
    }
    assert evaluation.report["reranking"]["attempts"] == 2
    ceiling = evaluation.report["first_stage_candidate_ceiling"]
    assert ceiling["candidate_k"] == 2
    assert ceiling["recall_any"] == 1.0
    assert ceiling["recall_all"] == 1.0
    assert set(ceiling["by_question_type"]) == {"single", "temporal"}
    protocol = evaluation.report["protocol"]
    assert protocol["storage"] == {"cache_enabled": False}
    assert protocol["dataset_non_abstention_questions"] == 3
    assert protocol["excluded_abstention_questions"] == 1
    assert protocol["reranker"] == {
        "candidate_k": 2,
        "output_k": 2,
        "max_candidate_chars": 256,
        "max_total_candidate_chars": 512,
        "concurrency": 2,
        "failure_policy": "exact BM25 fallback",
        "candidate_ids": "per-question opaque labels",
        "rank_fusion": {
            "enabled": True,
            "bm25_rank_weight": 0.4,
            "llm_rank_weight": 0.6,
            "tie_break": "original BM25 order",
        },
    }
    assert evaluation.report["selection"]["max_per_type"] == 1
    assert evaluation.report["screen_gate"]["exact_fallback"] is True
    confirmation_gate = evaluation.report["confirmation_gate"]
    assert confirmation_gate["minimum_target_improvement"] == 0.03
    assert confirmation_gate["target_metrics"] == ["recall_all@5", "ndcg_any@10"]
    assert confirmation_gate["exact_fallback"] is True
    assert confirmation_gate["frozen_selection"] is False
    assert confirmation_gate["frozen_protocol"] is False
    assert confirmation_gate["allowlist_guard"] is True
    assert confirmation_gate["passed"] == all(
        confirmation_gate[name]
        for name in (
            "primary_nonregression",
            "target_improvement",
            "target_ci_nonnegative",
            "exact_fallback",
            "frozen_selection",
            "allowlist_guard",
        )
    )
    run_dir = write_retrieval_artifacts(
        evaluation,
        tmp_path / "runs",
        config_hash="config",
        git_commit="commit",
        git_dirty=False,
        budget={"requests": 2},
    )
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert len((run_dir / "question_results.jsonl").read_text().splitlines()) == 2
    assert "SOTA claim" in (run_dir / "report.md").read_text(encoding="utf-8")
