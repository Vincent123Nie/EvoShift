# Pre-registration: context-local probation fast path

## Question

Can a replay-verified memory help the first changed sample before ordinary
retrieval happens, without turning a probationary hypothesis into a global
policy update?

The prior posterior and shadow candidates produced auditable signals but no
learner-visible capability change. The observed failure mode was temporal:
the first changed sample arrived before the new probationary memory was
retrieved. This experiment targets that activation delay only.

## Candidate v1 and rejection

The candidate is disabled by default and changes only the handling of a memory
that has already passed paired replay and entered future audit probation.

```yaml
evolution:
  context_local_probation_fast_path_enabled: true
  context_local_probation_fast_path_max_age: 8
  context_local_probation_fast_path_max_uses: 2
  context_local_probation_fast_path_min_trust: 0.60
```

For a probationary memory, the runner records the learner-visible
`(feedback_source, feedback_context)` at registration. The fast path may open
only when a later sample has the exact same pair. It runs two candidate-only
calls on that sample:

1. memory-off: no experience cards;
2. memory-on: exactly the probationary candidate card.

The memory-on answer remains the foreground answer, while the paired control is
fed into the existing future counterfactual auditor. A lease is consumed after
each intervention and expires after the fixed age or use budget. The candidate
never supersedes an active memory, mutates policy, or becomes visible in a
different observable context. Future audit completion is still required before
activation; failed or expired audits discard the lease.

The first 40-run targeted matrix (`runs/sweeps/20260806T083314Z`) showed the
mechanism but rejected v1. Under forced retrieval misses (`top_k=0`), changed
success improved by `+0.2857` clean and `+0.2571` under noise, with paired 95%
intervals above zero. Unrestricted candidate-only intervention regressed the
normal `top_k=3` path, however, and premature update increased by `0.0625` in
the retrieval-miss slice. Post-hoc event inspection identified two independent
causes: v1 intervened even when normal retrieval already selected the card,
and it could reuse a lease after the learner-visible feedback signal reverted.

## Revised candidate v2

Before evaluating v2, freeze these additional restrictions:

```yaml
evolution:
  context_local_probation_fast_path_enabled: true
  context_local_probation_fast_path_max_age: 12
  context_local_probation_fast_path_max_uses: 2
  context_local_probation_fast_path_min_trust: 0.10
```

- Open the fast path only when the exact probationary version is absent from
  ordinary retrieval. A normal retrieval hit is a no-call bypass.
- Bind the lease to the canonical learner-visible feedback signal observed at
  registration. A later sample must match source, context, and signal.
- A low-trust intervention still receives an explicit paired control, but its
  outcome cannot update future-audit lifecycle state until trust reaches the
  existing replay threshold.

The longer TTL bridges the observed registration-to-first-transition gap. It
does not weaken the two-use budget or permit a global state transition.

## v2 Result And Decision

The frozen v2 candidate was evaluated in
`runs/sweeps/20260806T084540Z` with five seeds (`122, 133, 144, 155, 166`),
clean and 10% noisy feedback, and both `top_k=0` and `top_k=3` retrieval. The
retrieval-miss slice (`top_k=0`) improved changed-case performance in every
seed:

| condition | score delta | changed-case success delta | old-rule leakage delta |
| --- | ---: | ---: | ---: |
| clean | `+0.0556` [`+0.0333`, `+0.0806`] | `+0.2857` [`+0.1857`, `+0.4000`] | `-0.2857` |
| 10% noise | `+0.0417` [`+0.0167`, `+0.0667`] | `+0.2143` [`+0.1000`, `+0.3429`] | `-0.2143` |

Invariant retention, premature update, attack-follow, false retirement, and
harmful active-memory exposure were unchanged. With normal retrieval
(`top_k=3`), capability, safety, request, and token metrics were exactly equal
to `Full`, confirming that a retrieval hit bypasses the fast path. The
first-changed-case success rate rose from `0.0` to `0.5` in the clean runs and
in three of five noisy seeds; the remaining noisy seeds show that a candidate
that does not exist before the transition cannot be recovered by this mechanism.

This supports adopting v2 as a narrowly scoped retrieval-miss recovery
mechanism, with the lifecycle guardrails retained. It is not evidence of a
general SOTA improvement: normal retrieval is intentionally unchanged, and the
candidate does not solve cold-start change detection or create a new candidate
before a policy transition. Those remain separate follow-up experiments.

## Online invariants

- no oracle fields, phase labels, hidden tags, or benchmark references are read;
- only the exact learner-visible source/context pair can match a lease;
- the canonical learner-visible feedback signal must also match;
- ordinary retrieval hits bypass the fast path without additional model calls;
- the candidate remains `PROBATION` until the existing future audit confirms it;
- a pending lease cannot change drift, trust, utility, retirement, revival, or
  policy state directly;
- every intervention has both memory-off and memory-on learner-visible calls;
- low-trust paired evidence is logged but cannot promote or reject a memory;
- a lease cannot create more than the configured number of interventions or
  survive its TTL; and
- the oracle firewall must show zero decision flips and identical state hashes.

## Evaluation

Compare `Full` and the candidate on the warm-start transient-burst stream that
previously showed first changed samples at indices 24, 48, and 120. Use the
same model/provider, prompts, phase order, budgets, cache policy, and seeds.
The primary endpoint is first changed-case success and activation delay. Report
context-local harmful exposure, false retirement, invariant retention,
premature update, poison persistence, future-audit precision, requests, and
tokens separately for clean, noisy, and burst conditions.

## Adoption gates

Adopt only if the candidate improves macro changed-case success by at least
`+0.03` with a 95% paired interval lower bound no lower than `0`, and at least
one stress condition improves. Invariant retention, premature update, poison
persistence, false retirement, and attack-follow rates may not worsen by more
than `0.02` absolute in any condition. Every intervention must have a paired
control, complete future audit, and pass the oracle firewall. Otherwise retain
the implementation only as a measured diagnostic and do not merge it as an
optimization.

This is a bounded mechanism experiment, not a claim of BOCPD/SOTA change-point
detection or a public-model leaderboard result.
