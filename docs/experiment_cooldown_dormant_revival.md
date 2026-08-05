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
   applied active memory and no `ACTIVE`/`PROBATION` memory in the dormant
   card's scope.
2. For each vacant scope, consider only its most recently retired latest-version
   card, and only when it was retired in the current run at least 18 episodes
   earlier. Rank the remaining dormant cards with the ordinary learner-visible
   retrieval function; inspect at most one card. The recency rule prevents
   falling back to an older policy merely because the immediate predecessor is
   still inside cooldown.
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

## Pre-formal pilot amendment

Pilot artifact `runs/20260805T103433Z-evoshift-aaaf5695` showed why the scope
vacancy condition is necessary. Two noisy premium-context observations revived
an old v3 card while another refund-policy card was still active, reducing
protected retention. The cooldown was satisfied, so tuning its numeric value
would not address the causal error. Before the formal matrix, the protocol was
strengthened to forbid dormant probes whenever the same memory scope already
has an active or probationary rule. The failed pilot is retained as the
pre-registered failure case; no formal result had been run when this safety
amendment was made.

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

## Results

Artifact: `runs/sweeps/20260805T103842Z` (60 deterministic runs).

| Condition | Current | Active circuit | Bidirectional | Current changed | Bidirectional changed |
|---|---:|---:|---:|---:|---:|
| clean | 0.9306 | 0.9444 | **0.9514** | 0.6875 | **0.7812** |
| clean + burst | 0.9306 | 0.9444 | **0.9514** | 0.6875 | **0.7812** |
| noise | 0.9194 | 0.9278 | **0.9306** | 0.6375 | **0.7000** |
| noise + burst | 0.9167 | 0.9250 | **0.9278** | 0.6250 | **0.6875** |

Paired hierarchical intervals for bidirectional minus `current_full`:

| Condition | Score delta | Changed-success delta |
|---|---:|---:|
| clean / clean+burst | `+0.0208 [+0.0111, +0.0319]` | `+0.0938 [+0.0500, +0.1375]` |
| noise / noise+burst | `+0.0111 [-0.0014, +0.0250]` | `+0.0625 [+0.0063, +0.1187]` |

Safety and mechanism results:

- every executed dormant revival was correct post hoc; no false revival was
  observed and protected retention remained 1.0;
- clean streams confirmed exactly one v2 revival per seed and removed the
  second phase-5 recurrence error;
- noisy seeds 33 and 55 confirmed the same safe revival; seeds 11, 22, and 44
  made no revival transition;
- false retirement and poison persistence remained zero;
- the combined variant reduced requests and tokens versus `current_full` in
  every clean condition and in noisy point estimates.

## Decision

Do not adopt the 18-episode cooldown configuration. It improves every point
estimate, strictly improves the standalone circuit, and clears all safety and
changed-case gates, but the noisy overall-score interval still crosses zero by
`0.0014`.

The failure is now narrow and interpretable: the fixed cooldown is conservative
enough to miss safe v2 revival in noisy seeds 22 and 44. Any shorter cooldown is
post-hoc with respect to these development seeds, so it must not be accepted on
this matrix. The next experiment will pre-register a 15-episode cooldown and
evaluate it on disjoint confirmation seeds before considering a merge.
