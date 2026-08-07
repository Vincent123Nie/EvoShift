# Pre-registration: selective BM25/LLM rank fusion

## Motivation

The first clean LongMemEval screen rejected unconstrained LLM reranking: the
top-20 candidate ceiling was perfect, but the model moved an already rank-1
evidence session below rank 5. This experiment tests a label-free safeguard
that preserves sparse evidence instead of trusting the second stage absolutely.

## Candidate

Keep the same pinned data, user-turn BM25 first stage, anonymous candidate
labels, bounded full-session prompt, and strict response allowlist. After a
valid LLM order is returned, combine the complete BM25 and LLM candidate orders
with a fixed rank-only fusion:

```
fused_rank(document) = 0.4 * bm25_zero_based_rank(document)
                    + 0.6 * llm_zero_based_rank(document)
```

Lower fused rank wins; ties use original BM25 order. The LLM still cannot add,
remove, or mutate a document. Provider errors, malformed output, unknown IDs,
duplicate IDs, and fusion invariant failures preserve the exact BM25 order.
The `0.4/0.6` weight is frozen before confirmation; it was selected as a
development hypothesis from the prior six-question screen and is not tuned on
the confirmation questions.

## Frozen confirmation

Use `bm25_llm_rerank_fused`, candidate pool 20, output prefix 10, 2,500
characters per session, concurrency 4, no cache, and the deterministic first
two non-abstention questions from every question type (`max_per_type=2`, 12
paired questions). Compare against BM25 and the raw LLM reranker on exactly the
same question IDs and model/provider settings.

## Gates

Adopt the fused candidate only if:

- `Recall-all@5`, `Recall-all@10`, `NDCG-any@5`, and `NDCG-any@10` do not
  decrease against BM25;
- at least one of `NDCG-any@10` or `Recall-all@5` improves by `+0.03` or more;
- the paired 95% interval for that improved metric has a non-negative lower
  bound;
- no fallback changes the exact BM25 order;
- no unknown/duplicate candidate ID reaches the fusion function; and
- full tests, Ruff, strict mypy, and the oracle-isolation tests pass.

The 12-question confirmation is still a mechanism screen, not a SOTA claim.
Passing it authorizes a larger paid screen and same-reader QA transfer test;
failing it rejects the candidate and preserves the negative result.

## Results

### Clean frozen confirmation

The candidate was evaluated on clean `main` commit
`3e1659a2177747bd7add8ec4e75b8c0c9972e586` with `gpt-5.6`, config hash
`a777f901b79b6b29435748fd9a3ada1a30ab4929f624bc6e60099addc36cd968`,
and the pinned dataset SHA-256
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
The manifest reports `git_dirty=false`. The selection was the frozen first two
non-abstention questions from each of the six question types (`n=12`), and the
cache was disabled.

| System | Recall-all@5 | Recall-all@10 | NDCG-any@5 | NDCG-any@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.8333 | 0.8333 | 0.7471 | 0.7471 | 0.6931 |
| Raw LLM rerank | 0.8333 | 0.9167 | 0.8767 | 0.9064 | 0.9286 |
| Selective rank fusion | **0.9167** | **0.9167** | **0.8958** | 0.9047 | 0.8917 |

Against paired BM25, selective fusion improved Recall-all@5 and Recall-all@10
by `+0.0833`, NDCG-any@5 by `+0.1488`, NDCG-any@10 by `+0.1576`, and MRR by
`+0.1986`. Paired bootstrap 95% intervals were:

| Delta | 95% interval |
| --- | ---: |
| Recall-all@5 | `[0.0000, 0.2500]` |
| Recall-all@10 | `[0.0000, 0.2500]` |
| NDCG-any@5 | `[0.0501, 0.2603]` |
| NDCG-any@10 | `[0.0513, 0.2704]` |
| MRR | `[0.0514, 0.3639]` |

All 12 fusion requests were applied, with zero fallback and an allowlist guard
pass for every question. The first-stage top-20 ceiling was Recall-any `1.0`
and Recall-all `0.9167`; the remaining multi-session miss therefore cannot be
recovered by any second-stage order. The fusion run used 146,051 reported
tokens and 92,150 ms summed provider latency. The reverse proxy did not report
prices, so artifact cost remains `0.0` rather than an inferred dollar value.

The raw reranker had higher point MRR and slightly higher NDCG-any@10, but did
not improve Recall-all@5, degraded that metric on one single-session-user
question, and its paired NDCG-any@10 interval crossed zero
(`[-0.0520, 0.3758]`). Its strict confirmation gate therefore failed. The
fusion candidate passed every frozen gate, including method/protocol/selection
matching, primary non-regression, the `+0.03` target, non-negative paired CI,
exact fallback, and allowlist isolation.

Formal artifacts:

- fusion: `runs/retrieval-fusion-main-clean/longmemeval-bm25_llm_rerank_fused-20260807T094114968555Z`;
- raw comparator: `runs/retrieval-fusion-main-clean/longmemeval-bm25_llm_rerank-20260807T094220980718Z`.

An earlier dirty-worktree diagnostic encountered retryable gateway `502`
responses and is retained only as provider-resilience evidence. It completed
without fallback but is not used for adoption. This 12-question result is a
mechanism confirmation, not a public LongMemEval SOTA or an end-to-end reader
QA result.

### Decision

Adopt `bm25_llm_rerank_fused` as the supported optional second-stage retrieval
candidate with the frozen `0.4/0.6` weights. Keep raw LLM reranking available
for comparison, but do not treat it as the default or as passing the selective
fusion confirmation protocol.
