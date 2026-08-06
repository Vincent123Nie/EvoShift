# Pre-registration: context-bound semantic revival without retirement probation

## Development finding

The oracle-safe 400-run development matrix at
`runs/sweeps/20260806T021621Z` showed that
`context_bound_sequential_recurrence` improved over `current_full`, but remained
slightly below the strongest revival and retirement comparators in all four
conditions. It therefore failed the frozen requirement for entering fresh
confirmation.

Two inspected-seed ablations isolated the cost:

1. disabling the semantic dormant index while retaining sequential retirement
   raised noisy performance and reduced harmful exposure, but clean score
   remained `0.9444` because retirement probation still delayed useful state
   changes;
2. retaining semantic revival and the exact retirement-context guard while
   disabling retirement probation restored clean score to `0.9514`, equal to
   `semantic_revival` and `lineage_revival_15`, and improved noisy scores to
   `0.9354` and `0.9288`.

This document freezes the second candidate before any unused confirmation seed
is run.

## Candidate

`context_bound_semantic_revival` uses:

- ordinary two-observation active-memory causal retirement;
- status-indexed and semantic dormant candidate discovery;
- exact `(feedback_source, feedback_context, memory_id, version)` eligibility;
- eight-episode reactivation grace;
- no retirement-probation transaction.

The candidate changes revival eligibility, not the active-audit retirement
threshold. The online guard still uses only the typed learner-visible
`feedback_context`. Hidden policy version, phase, reference, valid/stale tags,
corruption labels, and per-example protected markers are forbidden.

## Why removing probation is principled

Retirement probation addresses false retirement by requiring sequential
confirmation. In this benchmark, active retirement already requires two
negative paired-control observations. The additional transaction improved a
known retirement path but delayed beneficial recurrence and did not improve
the 20-seed development score.

The narrower candidate keeps the part supported by the observed failure:
global reactivation from evidence in the wrong context. It does not claim that
context equality proves retirement correctness. False-retirement rate and
post-hoc tag harm remain mandatory diagnostics.

## Targeted path check

Run `policy_shift_context_bound_semantic_targeted.yaml` on seeds 233 and 255,
10% feedback noise, and attack bursts 0 and 2. Compare:

- `semantic_revival`;
- `transactional_retirement`;
- `context_bound_sequential_recurrence`;
- `context_bound_semantic_revival`.

The candidate may enter fresh confirmation only if:

- invariant retention is `1.0` in all four candidate runs;
- candidate score is no lower than `semantic_revival` in each condition;
- its four-condition mean is no lower than every comparator;
- revival confirmation precision is `1.0` whenever defined;
- context-record coverage is `1.0` whenever a guarded opportunity exists;
- context-mismatch confirmations, false revival confirmations,
  post-confirmation tag-associated harmful exposure, and unconfirmed
  persistent transitions are all zero;
- the known seed-255 wrong-context version is excluded before any revival
  intervention in `refund:any:days_8_14`;
- the known seed-233 same-context revival is preserved when that path occurs;
- active-audit false-retirement rate is zero on the targeted slice.

## Frozen fresh confirmation

If the targeted check passes, run only the already registered unused seeds
`[344, 355, 366, 377, 388, 399, 411, 422, 433, 444, 455, 466, 477, 488, 499,
511, 522, 533, 544, 555]` over two noise levels and two burst lengths. Compare:

- `current_full`;
- `lineage_revival_15`;
- `semantic_revival`;
- `context_bound_semantic_revival`.

Adoption gates are frozen as follows:

- the candidate score point estimate is no lower than `semantic_revival` and
  `lineage_revival_15` in every condition and is strictly higher in at least one
  noisy condition;
- paired hierarchical 95% score-CI lower bounds versus `current_full` are
  non-negative in every condition;
- changed-case success, invariant retention, and future-change success are no
  lower than the strongest comparator in every condition;
- false retirement, false revival, poison persistence, context-mismatch
  confirmation, post-revival harmful exposure, and unconfirmed persistent
  transitions do not increase;
- revival and circuit confirmation precision are `1.0` whenever defined;
- the four-way hidden-protected-label firewall continues to report replay
  Jaccard `1.0`, decision flip rate `0.0`, identical online events, and matching
  semantic final state.

Failure of any fresh gate is non-mergeable. Development gains may motivate a
later candidate but may not be used to weaken these frozen confirmation gates.
