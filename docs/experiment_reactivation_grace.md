# Pre-registration: reactivation-aware destructive-action grace

## Motivation

The rejected temporal-diversity experiment showed that a global spacing rule
slows genuine policy changes yet still fails when noise recurs after a longer
gap. The remaining false-retirement cases share a narrower observable pattern:
the target memory had just been reactivated as a verified predecessor, then was
retired again before it accumulated post-reactivation causal maturity.

The next candidate protects destructive memory actions rather than delaying all
feedback adaptation. It uses only learner-visible memory lifecycle events and
global episode indices. It does not read policy phase, valid/stale tags,
corruption labels, attack labels, oracle answers, or future cases.

## Frozen candidate

Retain lineage-conditioned causal controls, status-indexed dormant revival, the
15-episode dormant-revival age, and the shared status-consistent provisional
ledger fix. Disable the rejected global temporal span gate by setting
`dynamic_feedback_change_min_span: 0`.

Add `active_audit_reactivation_grace_episodes`, default `0`. The candidate value
is `8`, matching the existing bounded circuit TTL rather than a hidden benchmark
phase length.

Whenever an exact memory version becomes active through predecessor restoration
or confirmed dormant revival, record its learner-visible reactivation index.
For the next eight episodes:

- exclude that exact version from ordinary active-audit retirement and
  low-trust circuit registration;
- continue updating its non-destructive utility posterior, but do not allow a
  posterior rollback to retire it;
- invalidate any older pending causal canary for that exact version;
- leave retrieval, solving, trust estimation, shift detection, replay,
  promotion, and non-reactivated memories unchanged.

The grace ends when `episode_index - reactivation_index > 8`, the exact version
leaves active state, or a different version becomes active. No action is
protected merely because post-hoc oracle metrics say it is correct.

## Development ablations

Use already inspected seeds
`[122, 133, 144, 155, 166, 177, 188, 199, 211, 222]` only for debugging and
mechanism attribution:

- `current_full`;
- `lineage_revival_15`;
- `status_indexed_revival`;
- `reactivation_grace_only`;
- `reactivation_guarded_recurrence`.

Cross feedback noise `[0.0, 0.10]` and attack-burst length `[0, 2]`, for 200
bounded development runs.

## Fresh confirmation

After implementation and tests are frozen, run only:

- `current_full`;
- `lineage_revival_15`;
- `reactivation_guarded_recurrence`.

Use unseen seeds `[233, 244, 255, 266, 277, 288, 299, 311, 322, 333]` over the
same four conditions, for 120 confirmation runs.

## Adoption gates

- all four fresh score point estimates improve over `current_full`;
- paired hierarchical 95% score CI lower bounds versus `current_full` are
  non-negative in all four conditions;
- the full candidate is not below `lineage_revival_15` in any condition and
  strictly improves its noisy and noise-plus-burst point estimates;
- changed-case success improves, while invariant retention and future-case
  success do not decrease;
- false retirement, false revival, poison persistence, and unconfirmed
  persistent transitions do not increase;
- circuit and revival confirmation precision are `1.0` whenever defined;
- every grace suppression names an exact active version with a recorded
  learner-visible reactivation index and age at most eight;
- no online decision reads oracle-only benchmark metadata.

Failure of any held-out gate rejects the candidate and forbids merging it into
`main`.

## Result: rejected

Implementation was frozen in `0c06c6d`. The 200-run development artifact is
`runs/sweeps/20260805T121836Z`; the 120-run confirmation artifact is
`runs/sweeps/20260805T122017Z` with the pre-registered unseen seeds.

| Condition | Current full | Lineage revival 15 | Guarded recurrence | Guarded delta vs current (95% CI) |
|---|---:|---:|---:|---:|
| clean | 0.9306 | 0.9514 | 0.9514 | `+0.0208 [+0.0139, +0.0285]` |
| noise | 0.9271 | 0.9340 | 0.9361 | `+0.0090 [-0.0042, +0.0208]` |
| clean + burst | 0.9306 | 0.9514 | 0.9514 | `+0.0208 [+0.0139, +0.0285]` |
| noise + burst | 0.9208 | 0.9285 | 0.9306 | `+0.0097 [-0.0021, +0.0208]` |

The candidate improved every score point estimate and reduced requests, but it
failed adoption:

- both noisy score intervals still crossed zero;
- noisy invariant retention was `0.9965`, below the current control's `1.0`;
- false retirement, circuit false confirmation, and false revival remained
  non-zero on held-out seeds;
- future-case success decreased slightly.

Held-out seeds 233 and 255 exposed the remaining tail failures. Two consistent
noisy causal observations can retire a correct memory that was initially
promoted rather than recently reactivated, so a reactivation-only grace is too
narrow. Seed 233 also showed that choosing only the most recently retired card
per scope can hide a more query-relevant older rule: the semantic retriever
ranked the correct v2 card above v3, but the scope-recency pre-filter removed it
before ranking.

The next candidate must make every irreversible retirement transactional and
must rank all exact retired candidates semantically before applying a recency
tiebreak. This branch is not eligible for merge into `main`.
