# Pre-registration: transactional retirement and semantic recurrence

## Motivation

The reactivation-grace confirmation removed many tail failures but still
allowed two failure modes on held-out streams:

1. two noisy causal observations can retire a memory that was newly promoted,
   before the reactivation grace can apply;
2. a scope-level recency filter can discard an older retired card that is more
   lexically applicable to the current query than the newest card in that
   scope.

The next candidate makes retirement a reversible transaction and fixes the
candidate-view ordering. The purpose is to reduce score variance and unsafe
tail events without slowing ordinary retrieval or feedback adaptation.

## Frozen candidate

Retain lineage-conditioned active controls, status-consistent lifecycle access,
the 15-episode dormant-revival age, and the eight-episode reactivation grace.
Keep `dynamic_feedback_change_min_span: 0`; the rejected global span gate is
not used.

### Transactional retirement

Add `active_audit_retirement_probation_enabled`, default `false`, and
`active_audit_retirement_probation_max_age`, default `8`. Candidate value is
enabled with age `8`.

When an active-audit or posterior-utility decision would retire an exact active
version, persist the same learner-visible retirement transition and restore any
verified predecessors as today, but register a pending retirement transaction
instead of treating it as final. On the next matching observable context:

- evaluate the normal post-retirement behavior and a forced exact old-memory
  counterfactual;
- finalize retirement only when feedback trust is at least the ordinary active
  audit threshold, the old memory was explicitly applied, and the normal
  behavior beats old-memory behavior by at least the existing `0.75` paired
  gain threshold;
- otherwise atomically roll back the exact retirement and re-supersede only the
  predecessors restored by that transaction.

Unresolved transactions expire after eight episodes and roll back. A pending
transaction is excluded from dormant revival and new failure extraction. The
transaction uses no oracle correctness or benchmark phase metadata.

### Semantic dormant candidate view

Add `dormant_revival_semantic_index_enabled`, default `false`, with candidate
value `true`. When enabled, build dormant candidates from the latest exact
`RETIRED` version per memory ID, apply scope-vacancy, age, trust, TTL, and paired
gain gates, then rank every eligible card by the existing BM25/domain retriever
before selecting the top applicable card. Retirement recency is only a
tiebreak among equal retrieval scores; it is no longer a pre-filter that can
hide a more applicable older rule. Exact-version checks remain mandatory.

## Development ablations

Use already inspected seeds
`[122, 133, 144, 155, 166, 177, 188, 199, 211, 222, 233, 244, 255, 266,
277, 288, 299, 311, 322, 333]` only for debugging and mechanism attribution:

- `current_full`;
- `lineage_revival_15`;
- `semantic_revival`;
- `transactional_retirement`;
- `transactional_semantic_recurrence`.

Cross feedback noise `[0.0, 0.10]` and attack-burst length `[0, 2]`, for 400
bounded development runs.

## Fresh confirmation

After implementation and tests are frozen, run only:

- `current_full`;
- `lineage_revival_15`;
- `transactional_semantic_recurrence`.

Use unseen seeds `[344, 355, 366, 377, 388, 399, 411, 422, 433, 444, 455,
466, 477, 488, 499, 511, 522, 533, 544, 555]` over the same four conditions,
for 240 confirmation runs.

## Adoption gates

- all four fresh score point estimates improve over `current_full`;
- paired hierarchical 95% score CI lower bounds versus `current_full` are
  non-negative in all four conditions;
- the full candidate is not below `lineage_revival_15` in any condition and
  strictly improves its noisy and noise-plus-burst point estimates;
- changed-case success improves, while invariant retention and future-case
  success do not decrease;
- false retirement, false revival, poison persistence, and unconfirmed
  persistent transitions do not increase;
- circuit, revival, and retirement-transaction confirmation precision are
  `1.0` whenever defined;
- every dormant candidate is an exact `RETIRED` version and every retirement
  rollback restores only the exact transaction's predecessors;
- no online decision reads oracle-only benchmark metadata.

Failure of any held-out gate rejects the candidate and forbids merging it into
`main`.
