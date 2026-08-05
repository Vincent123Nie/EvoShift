# Pre-registration: shift-gated sequential causal retirement

## Motivation

The current active-memory auditor waits for multiple negative memory-on/off
observations before retiring a card. That is conservative under isolated noise,
but it can expose the agent to repeated failures immediately after a real
regime change. The proposed change is a narrow fast path for catastrophic,
learner-visible harm.

## Hypothesis

When a shift detector fires and the paired memory-on minus memory-off feedback
delta is strongly negative, retiring the exact active card on that first causal
audit will improve post-shift score and reduce harmful exposure. It should not
increase false retirement on clean or noisy streams because the fast path is
gated by both drift detection and a strict delta threshold.

## Pre-registered variants

- `current_full`: existing active-memory governance.
- `sequential_causal_retirement`: same config, with
  `active_audit_early_retire_enabled=true` and
  `active_audit_early_retire_delta=-0.75`.

All other prompts, models, seeds, budgets, benchmark order, and cache policy
remain unchanged. The fast path is disabled during frozen held-out audits.

## Online rule

For an active card that was actually applied to a failed episode, compute the
learner-visible paired delta between the card-on and exact-version card-off
answers. Retire immediately only if:

1. the current episode also triggered the configured shift detector;
2. the causal delta is at or below `early_retire_delta`; and
3. the card is still active and has no intervening version change.

Otherwise retain the existing minimum-observation and negative-count gates.
The hidden oracle and benchmark stale/valid tags are never used for this online
decision.

## Primary metrics and gates

- overall mean score and success rate;
- changed-case success rate and first-window post-shift gain;
- harmful active-memory exposure rate;
- selective-forgetting precision and recall;
- false-retirement rate and protected-slice retention.

The candidate is adopted only if, over paired seeds, it improves mean score and
changed-case success on the noisy/burst stress slice, has a paired 95% CI whose
lower bound is non-negative, and does not increase false retirement or reduce
protected retention. A faster retirement with no score gain is rejected.

## Validation matrix

Run the existing 40-assignment PolicyShift matrix (5 seeds × clean/noise/burst/
noise+burst) with `current_full` and the new variant. The deterministic demo is
the causal attribution benchmark; any real-model run is a separate confirmatory
check and must not expose API credentials in artifacts.

## Results

Artifact: `runs/sweeps/20260805T091131Z` (40 deterministic runs, 5 paired
seeds). The burst-length axis produced the same behavioral deltas at each noise
level.

| Condition | Score current → sequential | Changed success current → sequential | Stale retention current → sequential | False retirement |
|---|---:|---:|---:|---:|
| clean / clean+burst | 0.9167 → 0.9236 | 0.6563 → 0.6875 | 0.2222 → 0.1667 | 0.0000 → 0.0000 |
| noise / noise+burst | 0.8958 → 0.9042 | 0.5625 → 0.5938 | 0.3580 → 0.2816 | 0.0000 → 0.0000 |

Paired hierarchical bootstrap intervals (seed clusters, then paired episodes):

- clean score delta: `+0.0069 [-0.0028, +0.0181]`;
- noisy score delta: `+0.0083 [-0.0014, +0.0195]`;
- clean changed-case delta: `+0.0313 [+0.0063, +0.0625]`;
- noisy changed-case delta: `+0.0313 [+0.0063, +0.0688]`;
- protected/invariant delta: exactly `0.0000` in all 20 paired conditions;
- false-retirement delta: exactly `0.0000`.

The new fast path executed two early retirements per clean seed and 1.4 per
noisy seed on average. Mean retirement latency fell from 16.0 to 8.0 episodes
in clean streams and from 14.8 to 9.8 under noise.

## Decision

Do not adopt this mechanism as a standalone change. It improves the point
estimate and has a positive changed-case interval, but the primary overall-score
CI still crosses zero, so the pre-registered adoption gate is not met. The
failure analysis shows that fast retirement removes a stale successor earlier
but leaves its previously valid predecessor unavailable. The next experiment
will combine the shift-gated fast path with exact predecessor restoration and
measure whether recurrent policies convert the latency gain into a larger,
statistically defensible score gain.
