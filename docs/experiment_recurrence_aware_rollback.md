# Pre-registration: recurrence-aware causal rollback

## Motivation

The early-retirement experiment reduced stale-successor latency but left the
previously valid predecessor superseded. In recurring enterprise policies this
creates an avoidable cold start: the agent has already verified the old rule,
yet relearns it from scratch after the newer rule becomes harmful.

## Hypothesis

When an exact active successor is causally retired, atomically restoring its
explicitly declared superseded predecessor will improve recurrent-policy score
without weakening safety. Restoration is not a new promotion: only a previously
active, versioned predecessor named in `supersedes_memory_ids` is eligible.

## Pre-registered variants

- `current_full`: existing two-observation causal retirement, no restoration.
- `sequential_only`: shift-gated early retirement, no restoration.
- `recurrence_aware_rollback`: shift-gated early retirement plus exact
  predecessor restoration.

All other model, prompt, feedback, replay, future-audit, retrieval, budget, and
benchmark settings remain unchanged.

## Online rule and safety invariants

1. Run the ordinary exact-version memory-on/off causal audit using only
   learner-visible feedback.
2. Retire using either the ordinary repeated-evidence rule or the pre-registered
   shift-gated strong-harm fast path.
3. If retirement is actually applied, inspect only the retired card's
   `supersedes_memory_ids`.
4. Restore a predecessor only if its latest stored version is currently
   `SUPERSEDED`; rejected, probationary, already-active, unrelated, and absent
   cards are never restored.
5. Record restoration provenance and continue ordinary outcome monitoring; a
   restored predecessor can itself be causally retired if the recurrence is
   false.

No hidden policy version, stale tag, valid tag, phase label, or oracle score is
used by the online restoration decision.

## Primary adoption gates

- overall score delta versus `current_full` has non-negative paired hierarchical
  bootstrap 95% CI and a positive point estimate;
- changed-case success improves with non-negative CI;
- invariant/protected retention does not decrease;
- false-retirement rate remains unchanged;
- correct-reactivation rate is 1.0 whenever restoration occurs;
- no increase in poison-persistence error.

## Validation matrix

Run 3 variants × 5 seeds × clean/noise × no-burst/burst = 60 deterministic
PolicyShift assignments. Report score, changed/future/protected slices,
retirement latency, reactivation precision, stale retention, and paired
hierarchical confidence intervals. Real-model validation is confirmatory and
may be run later when the API credential is available through the environment.

## Results

Final artifact: `runs/sweeps/20260805T092647Z` (60 deterministic runs). The
atomic state-transition implementation was included in this final rerun.

| Condition | Current score | Recurrence-aware score | Current changed | Recurrence-aware changed | Current future | Recurrence-aware future |
|---|---:|---:|---:|---:|---:|---:|
| clean / clean+burst | 0.9167 | **0.9306** | 0.6563 | **0.6875** | 0.9627 | **1.0000** |
| noise | 0.8958 | **0.9194** | 0.5625 | **0.6375** | 0.9629 | **1.0000** |
| noise+burst | 0.8931 | **0.9167** | 0.5500 | **0.6250** | 0.9629 | **1.0000** |

Paired hierarchical bootstrap intervals versus `current_full` (seed clusters,
then paired episodes; 5,000 replicates):

| Condition | Score delta | Changed-success delta | Future-success delta | Premature-error delta |
|---|---:|---:|---:|---:|
| clean / clean+burst | `+0.0139 [+0.0056, +0.0236]` | `+0.0313 [+0.0063, +0.0625]` | `+0.0373 [+0.0074, +0.0728]` | `-0.0373 [-0.0728, -0.0074]` |
| noise / noise+burst | `+0.0236 [+0.0083, +0.0458]` | `+0.0750 [+0.0188, +0.1625]` | `+0.0371 [+0.0074, +0.0802]` | `-0.0371 [-0.0802, -0.0074]` |

Safety and mechanism checks:

- invariant/protected retention remained exactly `1.0000` in all 40 paired
  current-versus-recurrence conditions;
- false-retirement rate remained exactly `0.0000`;
- poison-persistence error remained exactly `0.0000`;
- every observed predecessor restoration was oracle-correct post hoc
  (`correct_reactivation_rate=1.0000`);
- recurrence-aware rollback restored one predecessor per clean seed and 0.8 per
  noisy seed on average;
- early-retirement latency remained 8.0 episodes in clean streams and about
  9.4 under noise, versus 16.0 and 14.8 for the original rule.

## Decision

Adopt. The combined mechanism clears every pre-registered gate and turns the
standalone early-retirement point gain into a statistically positive overall
score improvement. The accepted implementation also persists successor
retirement and predecessor activation in one SQLite transaction, preventing a
crash from leaving both rules active or both unavailable.

This is strong deterministic mechanism evidence, not a public-model SOTA
claim. A real-model paired mini remains a confirmatory follow-up.
