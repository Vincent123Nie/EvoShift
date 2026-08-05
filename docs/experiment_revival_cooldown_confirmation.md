# Pre-registration: held-out confirmation of a 15-episode revival cooldown

## Development finding

On development seeds `[11, 22, 33, 44, 55]`, the 18-episode dormant-memory
cooldown was safe but overly conservative. It missed v2 revival in seeds 22 and
44 and left the noisy hierarchical score interval at
`+0.0111 [-0.0014, +0.0250]`. Those seeds are now treated as development data
and cannot establish the revised threshold.

## Frozen candidate

Keep the bidirectional recurrence algorithm unchanged and lower only
`dormant_revival_min_retired_age` from 18 to 15 episodes. The mechanistic
rationale is that the target predecessor is retired 15--19 episodes before the
next recurrence, while the observed same-regime false-revival opportunities
occurred within 12 episodes. Scope vacancy, most-recent-retirement selection,
exact-version matching, two strong paired gains, ordinary trust confirmation,
and all other safety gates remain unchanged.

## Disjoint confirmation set

Use seeds `[66, 77, 88, 99, 111]`, which were not used to choose the threshold.
Run:

- `current_full`;
- `recurrence_circuit_breaker`;
- `bidirectional_recurrence_circuits_15`.

Cross each variant with noise `[0.0, 0.10]` and attack-burst length `[0, 2]`
for 60 deterministic confirmation runs. The original development matrix may be
rerun only as a secondary consistency check.

## Adoption gates

- on held-out seeds, the calibrated bidirectional variant improves all four
  score point estimates over `current_full`;
- held-out hierarchical 95% score CI lower bounds are non-negative in clean,
  noise, burst, and noise+burst conditions;
- changed-case success improves; protected and future success do not decrease;
- false retirement, false revival, poison persistence, and unconfirmed
  persistent transitions remain zero;
- active-circuit and revival confirmation precision are 1.0 whenever defined;
- the held-out bidirectional score is not below the standalone active circuit;
- the fixed threshold also preserves or improves the prior development point
  estimates when rerun, but development results are not sufficient for adoption.

No threshold, seed, metric definition, or bootstrap setting may change after
the held-out sweep begins.

## Held-out result

The frozen sweep completed in `runs/sweeps/20260805T104145Z` without changing
the candidate, seeds, grid, metrics, or bootstrap procedure.

| Condition | Current full | Active circuit | Bidirectional-15 | Paired score delta vs current |
|---|---:|---:|---:|---:|
| clean | 0.9306 | 0.9444 | 0.9514 | `+0.0208 [+0.0111, +0.0319]` |
| attack burst | 0.9306 | 0.9444 | 0.9514 | `+0.0208 [+0.0111, +0.0319]` |
| 10% noise | 0.9167 | 0.9208 | 0.9250 | `+0.0083 [-0.0028, +0.0194]` |
| 10% noise + attack burst | 0.9056 | 0.9111 | 0.9153 | `+0.0097 [-0.0014, +0.0208]` |

The candidate improved every score point estimate, improved changed-case
success, preserved invariant retention, and reduced old-rule leakage. Revival
confirmation precision was `1.0` whenever revival executed. False retirement,
false revival, poison persistence, and unconfirmed persistent transitions did
not increase.

## Decision

Reject `dormant_revival_min_retired_age: 15` as the adoptable default. The
clean and burst-only intervals pass, but the score intervals under feedback
noise cross zero. This violates the pre-registered requirement that all four
held-out lower bounds be non-negative. The candidate must not be merged into
`main` or represented as confirmed.

The positive point estimates and changed-case intervals make dormant revival a
promising component, but another cooldown adjustment is not justified by this
experiment. The next experiment should address the episode-level evidence and
state-machine failures that prevented consistent noisy-condition gains.
