# Pre-registration: public LongMemEval retrieval validation

## Question

Does EvoShift's optional semantic reranking improve memory-session retrieval on
the public LongMemEval_S benchmark, beyond the deterministic PolicyShift stress
test, when no answer labels or benchmark-only metadata are visible to the
reranker?

This is a retrieval validation, not a test of online policy adaptation, memory
writing, or end-to-end question answering. It is intentionally separate from
the prequential runner so that the current question's `answer_session_ids`
cannot enter model input or retrieval state before scoring.

## Pinned Data And Protocol

- Dataset: `xiaowu0162/longmemeval-cleaned`
- Revision: `98d7416c24c778c2fee6e6f3006e7a073259d48f`
- File: `longmemeval_s_cleaned.json`
- Expected bytes: `277383467`
- Expected SHA-256:
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- Upstream retrieval-code reference:
  `xiaowu0162/LongMemEval@9e0b455f4ef0e2ab8f2e582289761153549043fc`
- Granularity: session
- Abstention questions: excluded when `question_id` ends in `_abs`, matching
  the official retrieval evaluation rationale

Every corpus item is rendered from its timestamp and user-visible conversation
turns. Dataset loading is streaming and validates aligned session IDs, dates,
and conversations. Downloaded or supplied files are rejected unless both the
pinned byte size and SHA-256 digest match.

The cleaned release contains some repeated upstream session IDs. Every corpus
entry therefore receives a unique opaque document ID, while scoring maps it
back to the source session ID and preserves the official duplicate-document
relevance semantics. Upstream IDs (including IDs containing `answer_`) never
enter a model request.

## Compared Systems

`bm25` is a local, API-free session retriever. It uses the fixed query and
session corpus for each question and emits a deterministic full ranking.

`bm25_llm_rerank` takes the first 20 BM25 sessions and asks the configured
Responses model to return at most the first 10 anonymous candidate labels. The
model sees only the question, bounded candidate text, and candidate dates. It
does not see answer text, `answer_session_ids`, `has_answer`, question type,
question ID, or baseline scores. Any unknown or duplicate label invalidates the
whole response; unreturned valid candidates are backfilled in BM25 order.
Provider errors, malformed JSON, overlong output, or an empty valid ranking
preserve the exact BM25 ranking.

Candidate text is capped at 2,500 characters per session and 50,000 characters
per request. The cap includes both the beginning and end of an overlong
session. Model requests use bounded concurrency, and request, token, cost,
latency, application, and fallback counts are recorded.

## Metrics

Report macro and per-question-type values for:

- `Recall-all@5` and `Recall-all@10`: one only when every evidence session is
  present in the first `k` results;
- `NDCG-any@5` and `NDCG-any@10`: the official binary-relevance LongMemEval
  formulation; and
- MRR: reciprocal rank of the first evidence session.

Also report Recall-any and Recall-all inside the first-stage BM25 candidate
pool. This is the reranker's hard ceiling: evidence absent from the first 20
sessions cannot be recovered by any second-stage ordering method.

The offline BM25 baseline runs over every non-abstention question. The paid
real-model mechanism screen uses the deterministic first non-abstention
question from every question type (`max_per_type=1`); it is not a leaderboard
estimate and will be reported separately from the full baseline.

## Adoption Gates

The public benchmark path is adopted as infrastructure only if pinned-data
integrity, oracle-input isolation, allowlist/backfill/fallback behavior, metric
compatibility, bounded prompts, deterministic selection, and usage accounting
are covered by tests and the full project checks pass.

The LLM reranker is considered externally supported only if, on the frozen
real-model screen:

- no primary metric (`Recall-all@5`, `Recall-all@10`, `NDCG-any@5`,
  `NDCG-any@10`) is lower than paired BM25 on the same questions;
- at least one rank-sensitive metric (`NDCG-any@5`, `NDCG-any@10`, or MRR)
  is strictly higher;
- gold-label permutation leaves the candidate request and ranking bit-identical;
- unknown or duplicate candidate IDs trigger exact BM25 fallback; and
- fallback exactly preserves the baseline when a model request fails.

With one question per type, no confidence interval or SOTA claim is valid. A
failure is retained as a documented negative result rather than hidden by
changing the sample or gate.

## Results

### Formal validation on the clean feature commit

The full pinned file was validated locally at the expected `277383467` bytes
and SHA-256. The offline run evaluated all 470 non-abstention questions and
excluded 30 abstention questions:

| System | Recall-all@5 | Recall-all@10 | NDCG-any@5 | NDCG-any@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7404 | 0.8128 | 0.7769 | 0.7970 | 0.8000 |

Formal artifact: `runs/retrieval-formal/longmemeval-bm25-20260807T090412560343Z`.
The dependency-free implementation matched the upstream `rank_bm25==0.2.2`
primary session metrics for all 470 questions and matched the upstream top-10
source-session order for all 470 questions. The only supplemental MRR
difference was an unspecified NumPy tie order among zero-score documents.

The real-model mechanism screen used the deterministic first question from each
of the six question types (`n=6`) with the authorized reverse proxy and model
`gpt-5.6`. This clean run had six successful requests and zero fallbacks:

| System | Recall-all@5 | Recall-all@10 | NDCG-any@5 | NDCG-any@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 1.0000 | 1.0000 | 0.9077 | 0.9077 | 0.8056 |
| BM25 + rerank (dirty diagnostic) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9167 |
| Delta (dirty diagnostic) | 0.0000 | 0.0000 | +0.0923 | +0.0923 | +0.1111 |
| BM25 + rerank (clean) | 0.8333 | 1.0000 | 0.8333 | 0.8859 | 0.8519 |
| Delta (clean) | -0.1667 | 0.0000 | -0.0744 | -0.0218 | +0.0463 |

The first-stage top-20 candidate ceiling was `Recall-any=1.0` and
`Recall-all=1.0` on this six-question screen. Rerank fallback rate was 0.00;
the six requests used 88,984 total tokens and the summed reported latency was
158,086 ms. Formal artifact:
`runs/retrieval-formal-real/longmemeval-bm25_llm_rerank-20260807T090525691199Z`.

The earlier dirty-worktree diagnostic (`runs/retrieval-real/...
20260807T083441284271Z`) had three gateway `502` fallbacks and is retained only
as a provider-resilience diagnostic, not as model evidence.

The clean screen fails the frozen gate (`primary_nonregression=false`). One
failure is especially instructive: BM25 already placed the evidence session at
rank 1, but the unconstrained LLM reranker moved it below rank 5. Since the
candidate ceiling was perfect, this is a pure second-stage over-reranking
failure. The reranker is not adopted for LongMemEval and remains default-off.

This is mechanism and protocol evidence, not a public SOTA claim: six questions
cannot support a confidence interval and no downstream reader QA comparison has
been made yet. The public benchmark infrastructure is adopted; the model-level
candidate is rejected by the primary non-regression gate. The next experiment
will test a label-free selective rerank/veto rule on the same frozen questions
before any larger paid screen.
