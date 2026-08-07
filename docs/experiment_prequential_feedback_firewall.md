# Pre-registration: prequential feedback firewall correction

## Problem

The context-local probation fast path called the stateful feedback assessor
before producing the current first-pass answer. The label was not sent to the
solver, but the intervention selector matched a lease against the current
sample's learner-visible `feedback_reference`. That violates the documented
predict, score, then adapt order and invalidates the earlier fast-path gain.

## Correction

Feedback trust now has two explicit operations:

1. `pre_predict(sample)` is read-only and uses only the observable
   source/context key plus trust state from episodes `< t`. It cannot inspect
   the current label.
2. `observe_feedback(sample, episode_index=...)` is the only stateful label
   consumer and runs after the first-pass answer has been scored.

A context-local lease must carry a non-empty canonical historical signal, is
registered only from dynamic feedback state, and matches only after the context
observation count advances beyond registration. Flipping the current feedback
label with all prior state fixed must not change the current prediction,
retrieved IDs, or intervention decision.

## Frozen evaluation

Re-run `configs/sweeps/policy_shift_context_local_probation_targeted.yaml` with
seeds `[122, 133, 144, 155, 166]`, clean and 10% noise, `top_k` in `[0, 3]`,
and Full versus the corrected fast path. Preserve the model, stream, budgets,
cache policy, and attack settings. The old artifact
`runs/sweeps/20260806T084540Z` is a historical counterexample, not a valid
adoption result.

Report score, changed-case and first-changed-case success, old-rule leakage,
invariant retention, premature update, attack-follow, requests, tokens, and
all context-probation interventions.

## Gates

- Current-label flips cannot change current prediction or persistent state
  before observation.
- Empty signals cannot create wildcard leases.
- Every intervention uses a signal and context observation committed by an
  earlier episode.
- Oracle firewall, full tests, Ruff, strict mypy, and package build pass.
- Any capability delta is reported only from the corrected rerun. If the prior
  gain disappears, the fast path remains default-off and is described as
  rejected rather than silently re-adopted.

Removing an invalid gain is a causal-validity success even if corrected score
is lower.

## Result and decision

Corrected artifact: `runs/sweeps/20260806T154440Z` (40/40 runs complete).

| Retrieval | Feedback | Score delta | Changed delta | First-changed delta | Premature-update delta |
| --- | --- | ---: | ---: | ---: | ---: |
| `top_k=0` | clean | `+0.0278` [`+0.0028`, `+0.0528`] | `+0.2143` [`+0.1286`, `+0.3143`] | `0.0000` | `+0.0625` |
| `top_k=0` | 10% noise | `+0.0139` [`-0.0167`, `+0.0417`] | `+0.1571` [`+0.0571`, `+0.2857`] | `0.0000` | `+0.0750` |
| `top_k=3` | clean | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `top_k=3` | 10% noise | `0.0000` | `0.0000` | `0.0000` | `0.0000` |

The corrected fast path performs four interventions per clean `top_k=0` seed
and 3.4 under noise. It does not change the first changed case because no
candidate exists before the first post-change feedback. Its historical signal
can rescue later retrieval misses, but the same mechanism follows the bounded
warm-up attack once, raising premature update beyond the original `0.02`
safety allowance. Clean score gain also misses the preregistered `+0.03`
threshold, and the noisy score interval crosses zero. With normal retrieval,
the exact-version retrieval-hit bypass makes every capability, safety, request,
and token metric identical to Full.

Decision: adopt the temporal firewall and non-empty-signal/configuration
guards as correctness fixes. Reject the context-local fast path as an
optimization and keep it default-off for diagnostics. The earlier artifact
remains useful only as a concrete example of why prompt-level oracle isolation
is insufficient when an auxiliary selector can cross the time boundary.
