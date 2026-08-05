# Pre-registration: cooldown-gated dormant memory revival

## Motivation

The reversible recurrence circuit breaker improves every PolicyShift point
estimate, but its noisy hierarchical score interval crosses zero. It also leaves
one avoidable clean error: after v2 was correctly retired during the v1 regime,
the returning v2 policy is relearned even though an exact, previously verified
v2 card remains in durable storage.

## Hypothesis

A symmetric revival circuit can recover a previously verified retired card on
the second observation of a recurring context. Requiring a minimum dormant
cooldown before the first probe should avoid resurrecting a card from an
isolated contradiction immediately after its retirement. Combined with the
active-memory circuit breaker, this should add a recurrent-policy gain large
enough to make the noisy score interval non-negative without weakening safety.

## Online protocol

1. Trigger only after a low-trust failed foreground answer with no explicitly
   applied active memory.
2. Consider only a latest-version `RETIRED` card that was retired in the current
   run at least 18 episodes earlier. Rank dormant cards with the ordinary
   learner-visible retrieval function; inspect at most one card.
3. Run an exact forced-memory-on probe. Register a revival canary only if the
   card is explicitly applied and `feedback_on - feedback_off >= 0.75`.
4. On the next sample with the same observable `(source, context)` within eight
   episodes, temporarily force that exact retired card into the foreground and
   run a paired memory-off control.
5. Reactivate only if feedback trust now reaches the ordinary active-audit
   threshold, the exact card is still the latest retired version, it is
   explicitly applied, and both learner-visible paired deltas are at least
   `+0.75`.
6. Confirmation resets regime-specific causal-audit counters before changing
   `RETIRED → ACTIVE`. Cancellation or expiry changes no persistent memory
   state. A context has at most one pending revival canary.

The decision cannot read hidden policy versions, phase labels, oracle answers,
valid/stale tags, feedback corruption annotations, or attack labels.

## Variants

- `current_full`: adopted recurrence-aware rollback without either speculative
  circuit.
- `recurrence_circuit_breaker`: reversible active-memory exclusion only.
- `bidirectional_recurrence_circuits`: active-memory exclusion plus cooldown-
  gated dormant-memory revival.

## Adoption gates

- all four clean/noise/burst score point estimates improve over `current_full`;
- hierarchical 95% score CI lower bounds are non-negative in all conditions;
- changed-case success improves and future/protected success does not decrease;
- false retirement, false revival, and poison persistence remain zero;
- active-circuit confirmation precision and revival precision are both 1.0;
- cancelled/expired revival canaries create no persistent state transition;
- the combined variant improves the standalone circuit's clean score and does
  not reduce its noisy score point estimate.

## Validation matrix

Run 3 variants × 5 seeds × clean/noise × no-burst/burst = 60 deterministic
PolicyShift assignments. Report score, changed/future/protected slices, paired
hierarchical intervals, both circuit ledgers, retirement/revival precision,
poison persistence, and total request/token cost.
