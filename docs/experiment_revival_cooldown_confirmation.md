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
