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

Pending implementation and confirmation.
