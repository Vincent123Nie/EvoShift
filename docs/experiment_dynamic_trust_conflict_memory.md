# Experiment: dynamic trust, conflict memory, and future counterfactual audit

Date: 2026-08-05
Branch: `codex/dynamic-trust-conflict-memory`
Status: adopted from deterministic mechanism evidence; public-model validation remains open

## Research question

Can an API-only agent distinguish a persistent policy update from intermittent
or bursty same-source corruption, and can it undo a memory that passed recent
replay but causes harm on later disjoint interactions?

This iteration removes the strongest shortcut in the previous experiment:
clean, noisy, and adversarial feedback may share the same observable source.
The online learner must not inspect hidden oracle correctness,
`feedback_kind`, `feedback_corrupted`, benchmark phase, or attack annotations.

## Hypotheses

1. A stateful source/context reliability posterior with pending-change evidence
   will quarantine isolated contradictions while accepting a repeated genuine
   policy update.
2. A short burst of internally consistent poisoning can still pass a recent
   replay buffer. A probationary memory plus disjoint future counterfactual
   audit should detect and roll back that candidate before the poison persists.
3. Explicit `supersedes_memory_ids` edges should remove obsolete prompt baggage
   after confirmation while allowing a rolled-back successor to restore the
   previous rule.
4. The complete method should reduce premature updates, harmful promotions, and
   old-rule leakage relative to unverified Reflexion and static-trust ablations
   under matched streams and declared total API budgets.

## Planned implementation

- Add an optional dynamic feedback trust model using only observable source,
  an application-provided context key, and learner-visible feedback labels.
- Track a Beta source reliability posterior plus per-context committed and
  pending labels. A contradictory label remains low-trust until repeated
  evidence confirms a possible change.
- Extend PolicyShift with a shared-source mode, observable context keys,
  isolated noise, and deterministic early poison bursts that can create a
  premature policy memory.
- Add probationary and superseded memory states, explicit temporal provenance,
  validated supersession edges, and restoration on successor rollback.
- After a replay gate passes, deploy a candidate in probation. On later
  relevant episodes, run a paired control with that candidate excluded. Confirm
  or reject it from learner-visible feedback, while recording hidden-oracle
  deltas only for post-hoc realized-utility metrics.
- Add final-promotion precision, harmful-promotion rate, audit latency,
  supersession/restoration counts, dynamic-switch counts, and full cost fields
  to run and sweep artifacts.

## Threat model and red lines

- Same-source intermittent and short-burst corruption is in scope.
- A coordinated attacker controlling the long-run majority of every relevant
  context is not solved by temporal consistency alone.
- Hidden oracle labels may score reported capability and realized future harm,
  but may not affect trust, candidate generation, replay, confirmation, or
  rollback decisions.
- A deterministic demo can validate mechanism semantics and falsify designs;
  it cannot establish public-model SOTA.

## Planned baselines and ablations

- Static prompting.
- Unverified Reflexion-style append-only memory.
- Full EvoShift.
- EvoShift without dynamic trust.
- EvoShift without future counterfactual audit.
- EvoShift without conflict supersession.

Each condition will run multiple seeds across clean, isolated-noise, burst
poison, and combined streams. All methods share sample order, provider, scorer,
and declared budgets.

## Adoption criteria

Adopt only if the full method:

- preserves clean changed-case success and invariant retention within a
  declared tolerance of the current verified method;
- lowers premature-update or harmful-promotion rate versus Reflexion and at
  least one verification ablation;
- demonstrates non-zero future-audit rollback on the constructed harmful
  candidate and later recovers when the real policy update arrives;
- reports no hidden-oracle access in online decision paths;
- keeps total request/token overhead bounded and explains any capability-cost
  tradeoff;
- passes unit, integration, formatting, strict type, config, and package gates.

Negative results and rejected variants will remain in this document.

## Pre-registered follow-up: asymmetric future-audit stopping

The first successful harmful-memory diagnostic showed that waiting for the
same minimum evidence count for both confirmation and rollback exposes users
to an avoidable second harmful application. The next variant therefore uses
asymmetric evidence requirements:

- a probationary memory may be rejected after one trusted paired future
  observation with strictly negative learner-visible utility;
- positive confirmation still requires the configured minimum number of
  disjoint observations and every normal promotion gate;
- an inconclusive decision at the minimum evidence count remains pending until
  the configured maximum evidence count instead of being rejected solely for
  low power;
- hidden-oracle deltas remain post-hoc metrics and never drive early stopping.

Adopt this variant only if it lowers rollback latency and premature-update
exposure under the burst-poison ablation without reducing clean changed-case
success, invariant retention, or final realized promotion precision. Report
false/unnecessary rollback rate separately so aggressive early rejection
cannot appear robust merely by deleting useful memories.

## Evidence checkpoint before asymmetric stopping

Run `20260804T174934Z-evoshift-e6ec2547` is the first deterministic diagnostic
that exercises the intended complete lifecycle. A locally plausible 14-day
rule passed six-example replay, entered probation, produced two future paired
deltas of `-1`, and was rejected. The same candidate was proposed again after
the real policy update and was confirmed; the later premium 30-day exception
was also confirmed. The run recorded one future-audit rollback, harmful
promotion rate `0.3333`, and mean audit latency `8.67`, but also exposed a
`0.1875` premature-update rate and `0.4286` post-attack poison-persistence error.

Five-seed deterministic baselines are stored in
`runs/sweeps/20260804T175032Z`; ablations are stored in
`runs/sweeps/20260804T175058Z`. Under a two-observation burst per affected
context with no ordinary noise, Reflexion and the replay-only ablation both
reached `0.75` premature
update. Static trust plus future audit reduced this to `0.25`, and the full
dynamic-trust method reduced it to `0.0`. This supports the layered design but
does not yet establish public-model quality or SOTA.

## Implemented mechanism

The final implementation contains three distinct defenses rather than one
opaque robustness score:

1. **Dynamic same-source trust.** Each observable source has a Beta reliability
   posterior. Each observable context has a committed label and a pending
   contradictory label. One contradiction is quarantined; repeated consistent
   contradiction commits a context change. Only source, context, and
   learner-visible feedback are read.
2. **Probation plus future counterfactual audit.** A replay-passing card is
   retrievable but remains `PROBATION`. On later relevant uses, the normal
   answer is paired with a control that excludes only that card. Learner-visible
   utility decides confirmation or rollback; hidden oracle utility is attached
   afterward for measurement only.
3. **Conflict lifecycle.** Confirmed cards may explicitly supersede older
   active rules. Superseded rules leave retrieval but remain versioned. If the
   successor later fails posterior-utility rollback, its superseded predecessor
   is reactivated.

Replay can be restricted to examples at or after the candidate's first
evidence, preventing a future-rule candidate from borrowing support from
irrelevant earlier history.

## Final single-run lifecycle evidence

The asymmetric-stopping diagnostic is
`runs/20260804T180757Z-evoshift-324e6d28`. It records the complete constructed
failure lifecycle:

- a premature 14-day memory passes recent replay and enters probation;
- its first trusted future counterfactual has negative learner-visible gain,
  so it is rolled back after one observation;
- the same semantic rule is proposed again after the genuine shift and is then
  confirmed;
- the later premium 30-day exception is also confirmed;
- invariant retention remains `1.0`.

Compared with the preregistered symmetric diagnostic
`20260804T174934Z-evoshift-e6ec2547`, asymmetric stopping changed:

| Metric | Symmetric minimum | Early harm stop |
|---|---:|---:|
| Mean score | 0.8611 | 0.8750 |
| Premature update | 0.1875 | 0.1250 |
| Poison persistence error | 0.4286 | 0.2857 |
| Harmful audit observations | 2 | 1 |
| Total requests | 196 | 194 |
| Total tokens | 58,718 | 57,560 |

This is causal mechanism evidence on a deterministic stream, not an estimate
of performance on arbitrary enterprise policies.

## Final repeated-seed baseline evidence

The final 60-run baseline sweep is
`runs/sweeps/20260804T180825Z`: three algorithms, five seeds, two noise levels,
and burst length zero or two. The primary burst rows are:

| Method | Noise | Score | Changed success | Premature update | Poison persistence | Requests | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full EvoShift | 0.0 | 0.9444 | 0.7143 | 0.0000 | 0.0000 | 112.0 | 34,579 |
| Reflexion | 0.0 | 0.8333 | 1.0000 | 0.7500 | 1.0000 | 84.0 | 30,396 |
| Full EvoShift | 0.1 | 0.9194 | 0.5857 | 0.0000 | 0.0000 | 133.8 | 40,849 |
| Reflexion | 0.1 | 0.8333 | 1.0000 | 0.7500 | 1.0000 | 85.2 | 31,468 |

Reflexion's perfect changed-case score is not an unqualified win: it accepts
the 14-day rule early, which also makes it adapt instantly after the real
change. Premature-update and poison-persistence slices expose that shortcut.
The full method removes the constructed attack but adapts more slowly.

## Final ablation evidence

The 140-run named ablation sweep is
`runs/sweeps/20260804T180837Z`. Disabling dynamic trust forces future audit to
act as the independent defense and isolates asymmetric stopping.

Burst only:

| Variant | Score | Changed | Premature | Poison | Rollback observations | Requests | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Early harm stop | 0.9306 | 0.8571 | 0.1875 | 0.3750 | 1.50 | 188.0 | 59,496 |
| Symmetric audit | 0.8750 | 0.7857 | 0.3750 | 0.7500 | 3.00 | 195.0 | 66,252 |

Noise plus burst:

| Variant | Score | Changed | Premature | Poison | False rollback | Rollback observations |
|---|---:|---:|---:|---:|---:|---:|
| Early harm stop | 0.8944 | 0.7286 | 0.2375 | 0.4750 | 0.2333 | 1.30 |
| Symmetric audit | 0.8417 | 0.6286 | 0.3875 | 0.7357 | 0.1667 | 2.73 |

Early stopping is adopted because clean behavior is unchanged while burst
exposure, first-pass error, requests, and tokens improve. Its higher false
rollback rate under noisy static trust is retained as a negative trade-off,
not hidden by the aggregate score.

## Metric correction discovered during analysis

An incomplete probation at stream end is `expired`, not a rollback. It must not
enter false-rollback rate or realized promotion precision. Reports now separate:

- `replay_estimated_promotion_precision`, based on recent replay gain;
- `realized_promotion_precision`, based only on completed future audits;
- `future_audit.realized_promotion_coverage`, the fraction of replay promotions
  that obtained a completed evidence audit.

The legacy `promotion_precision` field remains basis-labelled for compatibility.
For replay-only variants, realized precision is `N/A` and coverage is zero; the
replay estimate of `1.0` is not evidence of future utility.

The final code review also found a versioning race not exercised by the sweep:
a second version with the same `memory_id` could overwrite an older pending
audit, and completion could accidentally load the newest row. The adopted guard
allows one pending audit per logical memory, reloads the exact audited version,
and prevents posterior rollback from preempting probation. Unit tests cover the
guard; this fix does not change the reported deterministic sweep paths.

## Adoption decision and negative result

Adopt dynamic same-source trust, probationary future audit, explicit
supersession, stream-end expiration semantics, and asymmetric harm stopping.
The constructed burst—two observations in each of two future-rule contexts,
four attack-labelled episodes total—is fully suppressed by the complete method
across the five deterministic seeds.

The main negative result is adaptation latency. On the clean stream, disabling
dynamic trust raises changed-case success from `0.7143` to `0.8571`; under
noise plus burst, the full method reaches only `0.5857`. The current trust model
requires repeated per-context confirmation and is intentionally conservative.
The next algorithm iteration must reduce this delay without reopening
premature-update and poison-persistence failure modes.

## Claim boundary

These results justify the state-machine design and the next experiment. They do
not establish SOTA, real-model robustness, statistical generalization, or
production readiness. Required next evidence is a public or realistically
generated policy stream using a frozen API model, repeated stochastic runs,
sequentially valid promotion statistics, and an untouched held-out audit.

## Final quality gate

The adopted branch passed Ruff formatting/lint, strict mypy over 49 source
files, 109 tests, 84.77% branch-aware coverage, source/wheel build, and schema
loading for all 27 YAML files plus all 408 expanded sweep assignments.
