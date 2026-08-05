# Pre-registration: oracle-isolated validation paths

## Motivation

Before the context-bound revival candidate entered its full development
matrix, a decision-path audit found two inherited uses of the benchmark-only
per-sample `metadata["protected"]` marker:

- replay-buffer selection and its protected-slice mask;
- future-counterfactual-audit protected-slice gating.

The marker is an evaluation annotation, not learner-visible evidence. Keeping
it in online selection or promotion gates would invalidate the project's
claim that hidden oracle metadata is post-hoc only.

## Frozen correction

Apply the correction uniformly to every algorithm and ablation:

- configured `benchmark.protected_phases` may still define an explicit,
  deployment-known replay contract;
- per-sample `metadata["protected"]` must not affect replay selection, replay
  masks, future-audit decisions, or any other online transition;
- the hidden marker remains available only to post-hoc capability and safety
  metrics;
- no replacement may infer the hidden marker from policy version, phase index,
  reference answer, valid/stale tags, corruption kind, or oracle score;
- tests must show that toggling only `metadata["protected"]` leaves replay
  selection/masks unchanged when no protected phase is configured;
- runner tests must show that a hidden protected marker alone is never passed
  into a future-audit protection gate.

This is a validity correction, not a candidate-specific feature. Thresholds,
memory policies, and candidate ordering remain unchanged.

## Revalidation sequence

1. Rerun `policy_shift_context_bound_revival_targeted.yaml` unchanged after the
   correction. All previously frozen path and score gates must still pass.
2. If it passes, run a 400-run development matrix over the already inspected
   seeds `[122, ..., 333]`, two noise levels, two burst lengths, and:
   `current_full`, `lineage_revival_15`, `semantic_revival`,
   `transactional_retirement`, and `context_bound_sequential_recurrence`.
3. Only after freezing implementation and development analysis, run a 320-run
   confirmation on the unused seeds `[344, ..., 555]` with `current_full`,
   `lineage_revival_15`, `semantic_revival`, and
   `context_bound_sequential_recurrence`.

## Adoption gates

- the oracle-safe targeted slice retains every gate in
  `experiment_context_bound_revival.md`;
- on fresh confirmation, the candidate's score point estimate is not below
  `semantic_revival` or `lineage_revival_15` in any of the four conditions and
  strictly improves at least one noisy condition;
- paired hierarchical 95% score-CI lower bounds versus `current_full` are
  non-negative in every fresh condition;
- changed-case success does not decrease, and invariant retention plus
  future-case success do not decrease relative to the strongest comparator in
  each fresh condition;
- false retirement, false revival, poison persistence, unconfirmed persistent
  transitions, context-mismatch confirmations, and post-revival harmful
  exposure do not increase;
- circuit, revival, and retirement confirmation precision are `1.0` whenever
  defined;
- candidate context-record coverage is `1.0` whenever a guarded opportunity
  exists, and exact mismatch exclusions occur before any model probe;
- source inspection and tests confirm that hidden oracle-only metadata is
  absent from every online decision path.

Failure of any fresh gate is non-mergeable. The correction itself may merge
only together with a candidate that passes the full oracle-safe validation
sequence; no prior leaky score is adoption evidence.
