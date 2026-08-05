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

Failure is non-mergeable. Passing this inspected slice is necessary but not
sufficient for merging; the candidate still requires the larger development
matrix and fresh confirmation.
