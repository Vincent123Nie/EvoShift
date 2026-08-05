# Pre-registration: reversible recurrence circuit breaker

## Motivation

After recurrence-aware rollback, the remaining clean PolicyShift errors are two
failures at the start of each transition. On recurrent contexts, the first
failure already contains a strong exact-version signal: removing the applied
active card would have matched learner-visible feedback, but dynamic trust
correctly treats that first contradiction as unconfirmed. Waiting for the
second foreground failure is safe but unnecessarily costly.

## Hypothesis

A one-context, reversible circuit breaker can use the first strong causal
disagreement as a canary without permanently trusting it. On the next occurrence
of the same observable feedback context, temporarily exclude only the suspect
memory version. If a high-trust second observation again shows that memory-off
beats memory-on, commit the ordinary causal rollback and predecessor restoration;
otherwise cancel the canary and leave persistent state unchanged.

## Online protocol

1. Trigger only after a failed foreground answer where an `ACTIVE` card was
   explicitly listed in `applied_memory_ids`.
2. The ordinary active-audit trust threshold has not yet been reached, but
   feedback trust is at least the circuit-breaker floor.
3. Run an exact-version memory-off control. Register a canary only when
   `feedback_on - feedback_off <= -0.75`.
4. Persist the first causal ledger observation with retirement disabled.
5. On the next sample with the same learner-visible `(source, context)` within
   the bounded TTL, exclude only that exact memory version from the foreground
   answer and run a forced memory-on paired control.
6. Commit retirement/restoration only if trust now reaches the ordinary active
   audit threshold and the combined causal ledger passes the existing retirement
   rule. Otherwise cancel the canary. One canary is allowed per context.

The intervention never reads hidden policy versions, phase labels, oracle
answers, corruption annotations, or valid/stale memory tags. Its foreground
risk is bounded to one matching context occurrence before confirmation.

## Variants

- `current_full`: recurrence-aware rollback without a circuit breaker.
- `recurrence_circuit_breaker`: identical configuration plus the reversible
  canary protocol above.

## Adoption gates

- clean and noisy overall score point estimates improve;
- paired hierarchical 95% CI for score is non-negative in clean, noise, burst,
  and noise+burst conditions;
- changed-case success improves without reducing future-change or protected
  success;
- false-retirement and poison-persistence rates do not increase;
- circuit-breaker confirmation precision is 1.0 post hoc;
- isolated corrupted feedback cannot cause an unconfirmed persistent state
  transition.

## Validation matrix

Run 2 variants × 5 seeds × clean/noise × no-burst/burst = 40 deterministic
PolicyShift assignments. Report direct score, transition slices, canary
registrations/interventions/confirmations/cancellations/expirations, causal
control cost, protected retention, false retirement, and poison persistence.

