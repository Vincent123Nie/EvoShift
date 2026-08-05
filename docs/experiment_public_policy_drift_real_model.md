# Experiment: public-source policy drift with a live model

## Status

Pre-registered on branch `codex/public-policy-drift-eval` before implementation
or result inspection. This iteration is an engineering and evaluation upgrade,
not a claim that EvoShift is already state of the art.

## Motivation

The existing PolicyShift benchmark gives EvoShift a controlled oracle, noisy
learner-visible feedback, reversions, and selective-forgetting labels. Its
cases and policy rules are synthetic, and the strongest repeated-seed results
use the deterministic demo provider. A credible resume or research claim also
needs:

1. a non-scripted remote model;
2. a public, pinned policy source;
3. an explicit replay-only baseline;
4. repeated-seed paired uncertainty rather than mean and standard deviation
   alone; and
5. a trace showing the adaptation-versus-false-alarm trade-off.

## Public source and claim boundary

The source is the retail domain from
[`sierra-research/tau2-bench`](https://github.com/sierra-research/tau2-bench),
released as tau3-bench v1.0.1 under the MIT license.

| Item | Pinned value |
|---|---|
| Tag | `v1.0.1` |
| Peeled commit | `fc0055dc4e0a316c3f83133267fbd6faaa770992` |
| Retail policy path | `data/tau2/domains/retail/policy.md` |
| Retail policy SHA-256 | `2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467` |
| Retail tasks path | `data/tau2/domains/retail/tasks.json` |
| Retail tasks SHA-256 | `8e03ebce7901bd6218e7a7dc3105faa9324091a68058f7fe61c65262868812e8` |

The upstream policy is a static snapshot. EvoShift will derive structured
decision cases from its authentication, cancellation, return, exchange, and
payment constraints, then apply an explicit, versioned enterprise overlay to a
small subset of those constraints. Consequently the benchmark must be named
**Tau3Retail-PolicyDrift (public-source-derived)**. It is not an official
tau3-bench task, does not exercise the official interactive tool environment,
and must never be reported as an official tau3-bench or leaderboard score.

The source downloader must verify both revision and SHA-256. Generated samples
must record the source repository, tag, commit, source hash, rule family,
overlay version, and derivation status in their metadata.

## Stream design

The stream uses multiple independent rule families so that a learner cannot
solve the benchmark by memorizing one numeric threshold:

- cancellation reason eligibility;
- refund destination eligibility;
- exchange product compatibility;
- authentication method;
- gift-card balance sufficiency; and
- invariant confirmation and order-status requirements.

Policy versions form a revocation and recurrence schedule:

```text
v1 -> v2 -> v3 -> v2 -> v1 -> v2
```

Only a subset of rule families changes at each boundary. Unchanged families
are protected invariants. Every phase contains transition cases, future-change
cases, and invariant cases. The active version and oracle rule are hidden from
the solver. The online learner sees only its normal task feedback channel;
oracle validity/staleness labels remain post-hoc metrics.

The primary conditions are:

- clean persistent change;
- 10% incidental feedback noise; and
- a bounded same-source premature-update burst before a real change.

## Methods

All methods use the same remote model alias, prompt templates, sample order,
budget, and disabled shared cache.

| Method | Definition |
|---|---|
| Static | Fresh state, no candidate generation or update |
| Reflexion-style | Failure-derived append-only experience, no admission verification |
| Replay-only | Shadow candidates plus paired replay and protected slices; no dynamic trust, future audit, or active causal retirement |
| Full EvoShift | Dynamic feedback trust, paired replay, probation/future audit, conflict lifecycle, and active causal retirement |

`Reflexion-style` is a controlled repository baseline, not a reproduction of
the complete official Reflexion implementation. Replay-only must be a named,
auditable sweep variant rather than an informal interpretation of another row.

## Seeds and remote-model variability

The preregistered seeds are `[11, 22, 33, 44, 55]`. A seed controls case
generation, within-phase ordering, and corruption placement. It does not make
the remote model deterministic. The report must therefore distinguish stream
seed variability from provider/model variability and record the provider's
returned model identifier for every run.

Methods are paired by seed and sample identity. No failed seed may be silently
dropped. Provider failures remain explicit failed runs unless a preregistered
retry policy succeeds.

## Primary metrics

### Capability

- overall first-pass score;
- changed-case success;
- area under the adaptation curve;
- cumulative regret;
- mean recovery steps; and
- correct reacquisition after a recurring regime.

### Safety and memory governance

- old-rule leakage;
- premature-update rate;
- corrupted and attack feedback follow rate;
- invariant retention;
- harmful active-memory exposure;
- stale-memory retention;
- selective-forgetting precision/recall; and
- false-retirement rate.

### Resource use

- external requests;
- foreground and total input/output tokens;
- counterfactual-control requests; and
- nominal USD cost only when a dated, explicit price table is configured.

Proxy usage without a price table must be reported as token/request accounting,
not as zero monetary cost.

## Statistical analysis

Primary comparisons are paired deltas from Full EvoShift to each baseline.
The report must include:

- all per-seed rows;
- mean and standard deviation across seeds;
- a 95% hierarchical/cluster bootstrap interval that resamples seeds first and
  paired samples within each selected seed; and
- the number of paired seeds and paired samples contributing to each result.

The current flat single-run bootstrap is insufficient for repeated-seed
claims. Deterministic demo-provider results may validate the implementation but
cannot substitute for the live-model table.

## Failure-case trace

The primary negative case is a real persistent change in a low-frequency
context. The current dynamic trust model requires repeated consistent evidence
and shrinks source reliability toward a historical prior. This should reduce
false alarms from isolated contradictions but can quarantine the first correct
post-change feedback and increase recovery delay.

The trace must expose, per episode:

- learner-visible context and feedback label;
- trust before and after observation;
- pending contradiction count;
- committed context label;
- drift/candidate eligibility;
- active memory versions; and
- oracle score only in the post-hoc channel.

The next algorithmic candidate, after this trace is established, is a
Beta-Bernoulli change-point posterior over context run length. Its acceptance
criterion is lower post-change detection delay at matched or lower
premature-update and false-retirement rates. An anytime-valid confidence
sequence is the alternative if posterior thresholds prove poorly calibrated.

## Execution stages

1. Verify and cache the pinned public source.
2. Validate deterministic stream composition and hidden-oracle separation.
3. Run an 8--16 episode live provider mini diagnostic for each method.
4. Freeze model/config/prompt/budget and run the five-seed matrix.
5. Generate paired cluster intervals and a failure trace.
6. Adopt only changes that pass correctness gates and improve a preregistered
   metric without an unacceptable safety regression.

## Adoption gates

This iteration is accepted into `main` only if:

- source integrity failures are fail-closed;
- no source or overlay version leaks into solver prompts;
- all four method definitions are materialized in immutable configs;
- repeated-seed intervals are tested against hand-checkable examples;
- deterministic tests, Ruff, strict mypy, branch coverage, and builds pass;
- at least one live end-to-end policy-drift run completes on the current clean
  commit; and
- documentation states the public-source-derived and non-official claim
  boundary.

Full EvoShift need not win every capability metric. A safety improvement is
adoptable only when its capability and resource trade-offs are reported rather
than hidden.
