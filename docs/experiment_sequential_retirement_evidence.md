# Pre-registration: temporally separated sequential retirement evidence

## Motivation

Both symmetric and asymmetric one-shot retirement transactions failed the
inspected tail slice. A single paired intervention can still receive high
dynamic trust after adjacent corrupted feedback repeats. Conversely, an
eight-episode window can expire before a real policy change produces one clean
matching revisit.

## Frozen candidate

Retain the reversible exact-version retirement transaction and semantic
dormant candidate view. Change only its confirmation process:

- require `2` decisive post-retirement paired observations in the same
  direction;
- count observations only when feedback trust reaches the ordinary active
  audit threshold and the exact old memory is explicitly applied;
- require decisive observations to be separated by at least `2` episode
  indices, so adjacent repeated corruption contributes at most once;
- a current-over-old delta of at least `0.75` adds confirmation evidence;
- an old-over-current delta of at least `0.75` adds veto evidence;
- low-trust, unapplied, neutral, or temporally adjacent observations defer;
- confirm after two confirmation observations; atomically roll back after two
  veto observations;
- extend the transaction TTL from `8` to `24` episodes;
- unresolved transactions still roll back conservatively at TTL or stream end.

Evidence is scoped to the exact memory version and observable feedback
context. A contradictory decisive observation is retained in its own lane; no
oracle field, benchmark phase, reference answer, or post-hoc score may affect
the online transition.

Add configuration fields, defaulting off-compatible:

- `active_audit_retirement_probation_min_confirmations: 2`;
- `active_audit_retirement_probation_min_evidence_span: 2`.

The candidate uses max age `24` in diagnostic and confirmation variants.

## Development gates

First rerun the frozen 20-run inspected slice on seeds `[233, 255]`, noise
`0.10`, bursts `[0, 2]`, and the existing five variants. Proceed to the 400-run
development matrix only if `transactional_semantic_recurrence`:

- scores at least as high as `semantic_revival` on both seed-233 conditions;
- has retirement confirmation precision `1.0` whenever defined;
- has zero unconfirmed persistent transitions;
- does not reduce invariant retention relative to `semantic_revival`.

The full 400-run diagnostic and 240-run fresh confirmation retain the adoption
gates in `experiment_transactional_retirement.md`. Failure remains non-mergeable.
