# Pre-registration: posterior-gated shadow hypothesis lane

## Motivation

The trace-only context posterior in `runs/sweeps/20260806T053024Z` produced an
observable soft signal but, correctly, changed no online behavior and therefore
changed no score. The earlier unrestricted shadow-admission experiment failed
because every low-trust contradiction could schedule replay and interfere with
the later trusted fast path.

This candidate asks a narrower question: can a context-level posterior crossing
open a bounded, non-blocking shadow hypothesis lane while all activation still
requires trusted paired replay and future audit?

## Learner-visible and lifecycle boundary

The online gate reads only `feedback_source`, `feedback_context`, and
`feedback_reference`. Hidden oracle labels, phase/version, attack annotations,
valid/stale tags, and corruption metadata remain post-hoc only.

A posterior crossing may:

- allow one low-trust failure analysis and candidate observation;
- enter only the existing shadow evidence lane; and
- request paired replay only under the existing replay/future-audit invariants.

It may not directly update drift, policy, memory utility, retirement, revival,
supersession, or activation. The existing independent trusted/shadow validation
clocks remain unchanged, so a later trusted observation can bypass an earlier
shadow cooldown.

## Warm-start transient attack condition

The existing premature burst occurs before an affected context has a correct
committed label, so it is a cold-start identifiability stress rather than a
change-point test. Add an opt-in deterministic benchmark condition:

- first observe two clean labels per affected v1 context;
- then inject a same-source contradictory burst of length `0`, `1`, or `2`;
- then resume clean labels before the real v2/v3 transition.

The warm-up/burst controls are benchmark-only generators. The learner receives
the same source/context/label channel and never sees the attack window.

## Frozen candidate and targeted matrix

Control: current fixed-consensus Full EvoShift with shadow admission disabled.

Candidate:

- context posterior enabled with `h=0.15`, `epsilon_0=epsilon_1=0.10`, and
  crossing threshold `0.60`;
- shadow admission enabled only when `change_point_crossed=true`;
- shadow trust floor `0.10`;
- minimum trusted candidate observations `0`;
- paired replay and future audit unchanged;
- no shadow e-process; and
- all policy, replay, retirement, revival, and model parameters unchanged.

Mechanism screen: seeds `[233, 255]`, noise `[0.0, 0.10]`, early cold-start
burst fixed at `0`, warm-start transient burst `[0, 1, 2]`, two variants: 24
runs. `feedback_attack_rate` is explicitly `0.0`.

## Adoption gates

Proceed to fresh confirmation only if:

- clean score and changed-case success do not decrease;
- at least one clean or noisy condition improves changed-case success;
- warm-start burst premature-update, poison-persistence, attack-follow,
  invariant retention, and false-retirement do not worsen by more than `0.02`;
- every shadow-derived probation has passing paired replay and every activation
  has completed future audit;
- trusted-candidate cooldown bypass remains available and the candidate does
  not delay a later trusted replay;
- requests/tokens increase by at most `30%`; and
- the oracle firewall, full tests, Ruff, and strict mypy pass.

If the mechanism screen passes, confirmation will use truly unused seeds
`[611, 622, 633, 644, 655, 666, 677, 688, 699, 711, 722, 733, 744, 755, 766,
777, 788, 799, 811, 822]` with the same six conditions (240 runs). Failure of
any targeted gate rejects the candidate and forbids merging it into `main`.

## Targeted result: rejected

Artifact: `runs/sweeps/20260806T075011Z` (24 completed runs, no missing seed or
condition).

| Noise | Warm burst | Score delta | Changed delta | Safety deltas | Mean request delta | Shadow activations / seed |
|---:|---:|---:|---:|---|---:|---:|
| 0.00 | 0 | `0.0000` | `0.0000` | all `0.0000` | `+69.0` | `3` |
| 0.00 | 1 | `0.0000` | `0.0000` | all `0.0000` | `+87.0` | `3` |
| 0.00 | 2 | `0.0000` | `0.0000` | all `0.0000` | `+72.0` | `3` |
| 0.10 | 0 | `0.0000` | `0.0000` | all `0.0000` | `+183.5` | `3` |
| 0.10 | 1 | `0.0000` | `0.0000` | all `0.0000` | `+183.5` | `3` |
| 0.10 | 2 | `0.0000` | `0.0000` | all `0.0000` | `+165.5` | `3` |

The posterior gate successfully narrowed shadow work to context crossings and
did not delay the trusted path. Paired replay and future audit also prevented
any measured safety regression. However, the three shadow-derived activations
per seed did not change a single first-pass answer, while total requests rose
about `25%--64%` depending on condition. The noisy conditions violate the
`30%` resource gate, and no condition satisfies the required capability gain.

This candidate is rejected and must not be merged into `main`. The result shows
that safer hypothesis scheduling alone is insufficient: the next capability
candidate must prove that an admitted memory changes the relevant future action
before paying repeated replay/audit cost.
