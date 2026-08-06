# Pre-registration: retirement-context-bound dormant revival

## Motivation

The sequential-retirement candidate removed the known seed-233 regression but
the combined semantic-recurrence variant failed its invariant gate. The failed
path was a granularity mismatch: a memory retired after evidence in
`refund:premium:days_15_30` was later selected and globally reactivated from
evidence in `refund:any:days_8_14`. The card helped locally in the latter
context, but its unrestricted reactivation harmed a different invariant
context.

The online evidence was not an oracle error. The state transition was broader
than the learner-visible evidence justified.

## Frozen candidate

Keep the sequential retirement transaction unchanged. Add an optional
retirement-context guard to dormant revival:

- record the learner-visible `(feedback source, feedback context)` whenever an
  exact memory version enters `RETIRED`;
- when the guard is enabled, a retired version may enter the dormant candidate
  set only in that same observable context;
- matching uses exact source, context, memory ID, and version;
- rollback or reactivation removes the stale retirement-context record;
- no benchmark phase, oracle tag, reference answer, or post-hoc score may
  affect eligibility;
- keep semantic retrieval ranking inside the context-consistent candidate set;
- expose mismatch-exclusion and context-consistency counters in metrics.

Add one off-compatible field:

- `dormant_revival_retirement_context_enabled: false`.

Also add post-hoc path metrics for retirement probation: correct/false
registrations, registration precision, confirmation coverage, provisional
valid-memory retirement exposure, and expiry rate. These metrics may inspect
oracle tags only after the online action and may never drive a transition.

Track revival path effects as well: confirmation coverage, applications after
confirmed revival, and post-revival harmful exposure. Exact context agreement
is only a guardrail, not proof that a globally reactivated composite card is
safe inside every subcase of that context.

## Frozen targeted matrix

Use seeds `[233, 255]`, feedback noise `0.10`, attack bursts `[0, 2]`, and four
variants:

- `semantic_revival`;
- `transactional_retirement`;
- `transactional_semantic_recurrence`;
- `context_bound_sequential_recurrence`.

This is a 16-run inspected diagnostic. Adopt the candidate for the larger
development matrix only if:

- on both seed-255 conditions, score is at least
  `transactional_retirement` and invariant retention is `1.0`;
- on both seed-233 conditions, score and invariant retention are no lower than
  `transactional_semantic_recurrence`;
- the four-condition mean score is at least both `semantic_revival` and
  `transactional_retirement`;
- revival and retirement confirmation precision are `1.0` whenever defined;
- no confirmed revival crosses its recorded retirement context;
- mismatch exclusions are non-zero on the known seed-255 path;
- post-revival harmful exposure is no higher than the unguarded sequential
  variant;
- if the unguarded variant has a context-consistent confirmed revival, the
  candidate must not reduce confirmation coverage for those safe paths;
- unconfirmed persistent transitions remain zero;
- the new path metrics are present and provisional valid-memory exposure does
  not exceed the unguarded sequential variant.

The exact known-path checks are frozen as follows:

- preserve the safe seed-233/burst-0 revival of
  `mem-cfb49bc0ef485383@v1` in `refund:any:days_8_14`;
- exclude `mem-61294fcf3a594201@v1`, retired in
  `refund:premium:days_15_30`, before any forced-on probe in
  `refund:any:days_8_14`; it must have zero mismatched probe, registration, and
  confirmation events;
- candidate post-confirmation tag-associated harmful exposure must be zero on
  the targeted slice.

The retirement path metrics are diagnostic, not an adoption claim for the
inherited retirement mechanism. Registration precision must report both the
post-hoc tag basis and exact paired-oracle coverage where an active-audit
control exists. Tag-associated provisional exposure is explicitly a proxy,
not causal attribution. A later retirement-focused candidate must set an
absolute path-safety gate rather than merely match this branch.

Failure is non-mergeable. Passing this inspected slice is necessary but not
sufficient for merging; the candidate still requires the larger development
matrix and fresh confirmation.

## Targeted result: mechanism gates passed

The frozen 16-run slice completed at
`runs/sweeps/20260805T163314Z`. Across its four seed/burst conditions, the
context-bound candidate scored `0.9236`, versus `0.9184` for both
`semantic_revival` and `transactional_retirement`. Candidate invariant
retention was `1.0` in all four runs.

The exact path checks also passed:

- seed 233 / burst 0 preserved the same-context revival of
  `mem-cfb49bc0ef485383@v1`;
- seed 233 / burst 2 and both seed-255 conditions excluded
  `mem-61294fcf3a594201@v1` before any mismatched forced-on request;
- mismatched probe, registration, and confirmation counts for that exact
  version were all zero;
- context-mismatch confirmations and post-confirmation tag-associated harmful
  exposure were zero for every candidate run;
- context-record coverage was `1.0`, and retirement confirmation precision was
  `1.0` whenever defined.

This is still only a mechanism screen. Before opening the larger matrices, an
audit found that the inherited replay and future-audit validation paths used
the benchmark-only `metadata["protected"]` marker in online gates. The marker
is valid for post-hoc metrics but is not learner-visible. The targeted result
therefore cannot support an adoption claim until the oracle-isolation
amendment in `experiment_oracle_isolated_validation.md` is implemented and the
same slice is rerun.
