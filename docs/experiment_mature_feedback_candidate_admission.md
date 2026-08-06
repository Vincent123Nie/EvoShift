# Mature-feedback candidate admission

Status: rejected after the pre-registered targeted diagnostic; not eligible for
`main`.

## Failure mechanism

The dynamic trust model currently grants ordinary candidate eligibility on the
observation that establishes an initial two-label consensus for a previously
unseen context. This is necessary for fast cold-start adaptation, but an
internally consistent short attack burst can establish the same consensus.
The replay and future-audit layers limit persistent harm, yet the poisoned
candidate can still enter probation and reduce realized promotion precision.

The rejected context-scoped candidate experiment showed that separating
evidence buckets is insufficient when the minimum bucket size remains one.
The targeted seed-133 trace contains two proposals supported only by
`dynamic_initial_consensus`; one entered probation and later failed its
learner-visible future audit with post-hoc oracle mean delta `-1.0`.

## Candidate

Add an optional candidate-admission requirement for at least one *mature*
feedback observation. Mature evidence is defined entirely from learner-visible
trust state:

- `dynamic_consistent`: the label matches an already committed context regime;
- `dynamic_confirmed_change`: repeated contradictory evidence confirms a
  change from a previously committed regime; or
- static provenance mode, where no dynamic context state is available.

`dynamic_cold_start`, `dynamic_initial_consensus`, and unresolved pending-change
observations are provisional. They may still update the bounded evidence pool,
but cannot by themselves trigger replay when the gate is enabled.

The candidate setting is:

```yaml
evolution:
  candidate_min_mature_feedback_observations: 1
```

The default remains zero, so existing algorithms and baselines are unchanged.
No hidden policy version, phase, corruption flag, attack annotation, oracle
score, or valid/stale memory tag is used online.

## Targeted evaluation

Use the known promotion-precision failure seeds `[133, 155, 233, 277, 333]`
only for mechanism development. Compare the current contradiction-aware fast
retirement stack against mature-feedback admission at 10% noise and attack
bursts `0`, `2`, and `4` (30 paired deterministic runs).

Adopt for fresh confirmation only if:

- realized promotion precision strictly improves on burst `2` or `4`;
- harmful-promotion rate does not increase in any condition;
- score, changed-case success, and invariant retention do not decrease in any
  condition mean;
- the candidate blocks at least one replay attempt supported solely by initial
  consensus; and
- every admitted candidate event reports at least one mature observation.

Requests and tokens are reported but are not primary gates. If the targeted
gate passes, freeze the code and run fresh seeds before any merge. This is a
deterministic mechanism test, not a public-benchmark or SOTA claim.

## Reproduction

```bash
.venv/bin/evoshift sweep \
  --spec configs/sweeps/policy_shift_mature_feedback_candidate_admission_targeted.yaml
```

## Targeted result and decision

Artifact: `runs/sweeps/20260806T045058Z` (30 runs: two variants, five known
promotion-precision failure seeds, 10% noise, and attack-burst lengths `0`,
`2`, and `4`).

| Burst | Score delta | Changed delta | Realized-promotion-precision delta | Requests delta |
|---:|---:|---:|---:|---:|
| 0 | `0.0000` | `0.0000` | `0.0000` | `-2.4` |
| 2 | `0.0000` | `0.0000` | `0.0000` | `+4.0` |
| 4 | `0.0000` | `0.0000` | `0.0000` | `0.0` |

The maturity gate was active: candidate events reported
`insufficient_mature_feedback_evidence` for proposals supported only by an
initial consensus, and every staged candidate had at least one mature
observation. It did not, however, alter the final replay/audit outcomes. A
single later mature observation was enough to release the same candidate, so
the candidate failed the required strict precision improvement gate. The full
matrix and live-model confirmation were not run.

The next bounded candidate is a two-observation mature-evidence gate. It is a
new hypothesis with a separate preregistration because it trades more safety
against potentially slower genuine policy updates.
