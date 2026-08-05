# Experiment: causal governance for active memories

## Status

Completed and adopted from the pre-registered deterministic mechanism study.
This iteration prioritizes algorithmic completeness and measurable
memory-governance quality. Adaptation latency and retrieval-bandit optimization
remain secondary objectives; public-model validation is still open.

## Research question

After a memory has passed replay and probation and becomes active, can an agent
continue to distinguish a useful rule from a stale or harmful rule when policy
regimes are revoked, reverted, or recur? The online learner must selectively
retire invalid memories without deleting invariant capabilities or consulting
the benchmark's hidden oracle.

The current system gives probationary memories disjoint future
counterfactual attribution. Active memories, however, are governed mainly by a
mixed success posterior and explicit supersession edges. That posterior does
not isolate which memory caused a later failure, so an active memory can remain
stale after its policy is revoked or be retired because another component
failed.

## Hypotheses

1. A budgeted leave-one-memory-out audit triggered by trusted failures will
   reduce harmful active-memory exposure and stale-memory retention relative to
   posterior rollback and explicit supersession alone.
2. Per-memory causal evidence will improve selective-forgetting precision and
   recall without reducing invariant retention.
3. Versioned retirement plus predecessor restoration will correctly recover an
   earlier rule after a successor policy is revoked, and recurring regimes can
   reactivate or reacquire the appropriate rule without corrupting provenance.
4. The gains will remain visible after accounting for additional requests and
   tokens.

## Benchmark extension

Extend PolicyShift with deterministic, configurable regime schedules covering:

- **revocation**: `v1 -> v2 -> revoke(v2)`, where the successor rule becomes
  invalid and the earlier rule is valid again;
- **multi-step reversion**: `v1 -> v2 -> v3 -> revert(v1)`, where more than one
  learned successor becomes stale;
- **recurrence**: `v1 -> v2 -> v1 -> v2`, where a previously valid policy
  returns;
- **invariants across every phase**, which must never be forgotten;
- **ordinary feedback noise and adversarial feedback bursts**, kept distinct
  from the hidden ground-truth outcome.

Each sample exposes only normal task input, context, model output, and the
configured learner-visible feedback channel. Ground-truth phase and oracle
utility are used only for post-hoc metrics.

## Algorithm under test

Add a budgeted Active Memory Auditor:

1. Trigger only after an active memory was actually retrieved and a sufficiently
   trusted learner-visible failure was observed.
2. Rank the applied active memories by audit risk and spend at most the
   configured per-episode budget on one candidate by default.
3. Run a paired control solve with that exact memory version excluded while all
   other observable state is held fixed.
4. Convert the learner-visible treatment/control difference into a per-memory
   causal ledger: audit count, positive/negative/neutral evidence, cumulative
   utility delta, and last-audit episode.
5. Retire only after a fixed minimum evidence count and a negative causal mean
   or harmful-evidence threshold. Do not use hidden oracle utility in this
   decision.
6. If the retired memory superseded a predecessor, restore the predecessor when
   provenance and current state permit it.
7. Keep probation auditing and active-memory auditing disjoint: an item cannot
   be governed by both mechanisms at the same time.

The implementation must persist the ledger and every lifecycle transition in
SQLite so the decision can be replayed and inspected.

## Pre-registered variants

| Variant | Active-memory governance | Purpose |
|---|---|---|
| Current EvoShift | Mixed posterior rollback + explicit supersession | Iteration 3 baseline |
| Supersession only | Explicit successor edges, posterior rollback disabled | Tests lifecycle edges without causal monitoring |
| Causal auditor | Budgeted active leave-one-out monitoring | Primary method |
| Auditor disabled | Same new benchmark/config surface, monitoring off | Isolates benchmark effects |
| Oracle upper bound | Post-hoc perfect stale-memory retirement | Analysis only; never an online method |

If implementation cost permits, add a causal-ledger ablation that makes the
same number of control calls but attributes failures to all retrieved memories;
this tests whether leave-one-out credit assignment, rather than extra inference,
causes the improvement.

## Primary metrics

- **Harmful active-memory exposure**: number and rate of decisions made while an
  oracle-harmful active memory is applied, including exposure before retirement.
- **Stale-memory retention rate**: fraction of oracle-stale memory-episode
  opportunities for which the memory remains active and retrievable.
- **Selective-forgetting precision**: retired stale/harmful active memories over
  all causally retired active memories.
- **Selective-forgetting recall**: retired stale/harmful active memories over all
  active memories that became stale/harmful with adequate evaluation exposure.
- **False retirement rate**: useful or invariant memories retired by the active
  causal auditor over all its retirements.
- **Retirement latency**: learner-visible opportunities from first stale/harmful
  exposure to retirement; reported as diagnostic rather than the main objective.
- **Reactivation/reacquisition correctness**: correct regime memory restored or
  relearned after revocation/recurrence without simultaneously activating a
  conflicting rule.
- **Counterfactual audit coverage**: audited eligible active-memory failures over
  all eligible failures, alongside audit budget utilization.
- **Invariant retention**, changed-rule success, old-rule leakage, premature
  update, poison persistence, total score, requests, and tokens.

All forgetting labels are computed post-hoc from hidden benchmark metadata.
Online retirement uses learner-visible feedback only. Metrics must separately
report memories with insufficient exposure instead of silently counting them as
correctly retained.

## Adoption criteria

Adopt the causal auditor only if the pre-registered revocation/reversion sweep:

1. lowers both harmful active-memory exposure and stale-memory retention against
   current EvoShift;
2. improves selective-forgetting precision/recall with a non-zero recall;
3. does not reduce invariant retention below the baseline or introduce a
   material increase in premature update or poison persistence;
4. demonstrates at least one correct predecessor restoration or recurring-regime
   reacquisition in an end-to-end run;
5. keeps additional requests/tokens within the configured bounded audit budget;
6. passes unit, integration, lint, strict type-check, branch-coverage, config,
   and package-build gates.

A negative result is retained if causal monitoring reduces stale exposure but
causes false retirement, fails under feedback corruption, or spends excessive
inference. Thresholds must not be tuned on hidden oracle labels; threshold
changes require a documented learner-visible rationale and a fresh sweep.

## Required evidence and artifacts

- unit tests for exact-version exclusion, ledger updates, threshold decisions,
  no-oracle decision paths, restoration, and audit-budget enforcement;
- integration tests for at least one revocation and one recurrence trajectory;
- baseline and ablation sweep manifests with expanded assignments;
- per-run lifecycle/audit events in SQLite and metrics JSON;
- a results section appended here with adopted and rejected variants, costs,
  failure cases, and the final merge decision.

## Implementation outcome

The implemented stream is `v1 -> v2 -> v3 -> v2 -> v1 -> v2` with 24
episodes per regime. `MemoryItem` now persists a version-specific causal ledger:
audit count, positive/negative/neutral counts, learner-visible delta sum, and
last audit index. `ActiveMemoryAuditor.observe` has no oracle argument. The
runner separately attaches oracle delta and stale/valid tags to post-hoc events.

A first implementation eagerly restored a superseded predecessor after causal
retirement. The single-seed diagnostic exposed this as unsafe: retiring v3
after a full reversion to v1 reactivated v2, which was also stale. The adopted
implementation keeps the predecessor's provenance but does not activate it.
Its candidate signature is reopened, so a recurring policy must pass replay and
probation again before a new version becomes active.

## Final deterministic evidence

Artifacts:

- ablations: `runs/sweeps/20260805T021957Z` (120 runs);
- Static/Reflexion/EvoShift baselines: `runs/sweeps/20260805T022036Z` (60 runs);
- feedback/threshold stress: `runs/sweeps/20260805T022059Z` (80 runs).

Each row below is the mean of five seeds on the same 144-episode schedule.

| Noise / burst | Variant | Score | Changed | Invariant | Harm exposure | Stale retention | Forget P/R | False retire | Requests |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 / 0 | Causal auditor | 0.9167 | 0.6562 | 1.0000 | 3.0 | 0.2222 | 1.00 / 1.00 | 0.00 | 288.0 |
| 0 / 0 | Current posterior rollback | 0.8333 | 0.5625 | 0.8826 | 20.0 | 0.7500 | 0.00 / 0.25 | 0.00 | 296.0 |
| 0.10 / 0 | Causal auditor | 0.8958 | 0.5625 | 1.0000 | 4.6 | 0.3580 | 1.00 / 1.00 | 0.00 | 289.6 |
| 0.10 / 0 | Current posterior rollback | 0.8417 | 0.5375 | 0.9061 | 17.2 | 0.8000 | 0.00 / 0.20 | 0.00 | 279.0 |
| 0.10 / 2 | Causal auditor | 0.8931 | 0.5500 | 1.0000 | 4.6 | 0.3580 | 1.00 / 1.00 | 0.00 | 307.4 |
| 0.10 / 2 | Current posterior rollback | 0.8389 | 0.5250 | 0.9061 | 17.2 | 0.8000 | 0.00 / 0.20 | 0.00 | 296.8 |

The causal-auditor and causal-auditor-with-posterior-disabled rows are exactly
equal in all four primary conditions. On this benchmark, mixed posterior
rollback contributes no retirement; the measured gain comes from per-memory
counterfactual governance. The audit itself uses only 2.0 control calls per
clean seed and 2.8 under 10% noise. Total requests can be lower than the
baseline because early correct retirement prevents repeated critic and replay
work.

The recurrence path is exercised in end-to-end runs. Under 10% noise, 0.8
reacquisition events occur per seed on average; correctness is 1.0 in every run
where reacquisition occurs. Eager predecessor restoration is rejected: in the
clean sweep it raises harm exposure from 3.0 to 9.0, stale retention from
0.2222 to 0.2821, and requests from 288 to 355.

## Threshold stress and negative results

One negative observation gives a better in-distribution score, but it is not
adopted. Under static trust and only 10% noise its false-retirement rate is
0.497-0.517. At 25% noise it reaches 0.833-0.867. Dynamic trust is therefore a
material part of the safety result, not cosmetic plumbing.

Even the adopted dynamic two-evidence rule is not noise-proof. At 25% noise its
false-retirement rate is 0.10 without a burst and 0.00 with the tested burst;
invariant retention falls to 0.993. This stress result is retained as a limit,
not tuned away using oracle labels. Context-windowed or sequentially valid
causal evidence is a future research direction.

## Adoption decision

Adopt budgeted exact-version active-memory auditing, persisted learner-visible
causal ledgers, causal retirement, and replay/probation-based reacquisition.
Keep two negative observations and dynamic trust as the default. Reject eager
predecessor activation and the one-observation default. The deterministic demo
supports mechanism and metric claims only; public-model validation remains
required before any SOTA statement.

Final quality gate: 120 tests, 85.03% branch-aware coverage, Ruff, formatting,
strict mypy over 50 source files, source/wheel build, 31 YAML files with 668
expanded sweep assignments, and the 27-file pinned BBH checksum audit all pass.
