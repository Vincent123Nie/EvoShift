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

## Implemented engineering result

Commit `b0f965cb6483f155fcc96b65713b565e390cf13e` implements:

- fail-closed download, size, SHA-256, revision, and semantic validation of the
  two pinned tau3 retail files;
- a 7-rule-family, 18-case-template derived stream with transition,
  future-change, protected, noisy, attacked, revoked, and recurring regimes;
- learner-visible semantic memory tags without copying hidden oracle tags into
  the critic;
- a first-class `replay_only` algorithm mode;
- four-method live mini and five-seed formal sweep specifications; and
- seed-then-paired-sample hierarchical bootstrap reports in JSON, CSV, and
  Markdown.

The generated 144-episode formal stream at seed 11 has 90 `DENY` and 54
`ALLOW` labels, seven rule families, zero policy-version prompt leaks, and
dataset fingerprint
`a124f152e2228ff878c86069912ba8b344ede3ff71549ccc2e3ab36a530b7435`.

## Live four-method mini diagnostic

Artifact: `runs/sweeps/20260805T030609Z`. All four runs used the same remote
`gpt-5.6` alias, seed 11, 16 uncached foreground episodes, and clean
`v1 -> v2` stream. This is an engineering diagnostic, not the preregistered
five-seed result.

| Method | Score | Changed success | Invariant retention | Premature update | Total requests | Total tokens |
|---|---:|---:|---:|---:|---:|---:|
| Static | 0.6250 | 0.7500 | 0.7500 | 0.7500 | 16 | 9,447 |
| Reflexion-style | 0.7500 | 0.7500 | 0.7500 | 0.2500 | 20 | 18,139 |
| Replay-only | 0.6250 | 0.5000 | 0.7500 | 0.5000 | 78 | 58,568 |
| Full EvoShift | 0.5625 | 0.5000 | 0.6250 | 0.5000 | 27 | 18,083 |

The mini run rejects any capability claim for the current Full configuration.
Against Reflexion-style, Full changed mean score by `-0.1875`, with a one-seed
paired interval `[-0.5000, 0.1250]`. The interval is deliberately reported but
is not a repeated-seed inference.

## Established failure case

Full EvoShift quarantined 9 of 16 clean observations. Every unseen policy
context received cold-start trust `0.40`, below the `0.60` learning gates. A
second matching observation established the initial consensus, but sparse
invariant contexts appeared only once. At the real `v1 -> v2` change, the first
new label for each repeated context received conflict trust `0.10`; only the
second label confirmed the change.

This mechanism prevented immediate reactions to isolated contradictions, but
in the short public-source stream it also:

- suppressed useful failure extraction on first-seen contexts;
- delayed both changed-context updates by one observation;
- reduced invariant retention from `0.75` to `0.625` relative to the other
  adaptive methods; and
- produced one replay-passing probationary memory that expired at stream end
  without future-audit evidence.

Replay-only exposed the opposite cost problem: one memory passed replay, but
four memory/policy validations drove total usage to 78 requests and 58,568
tokens without improving score over Static. These two negative results make
the next priorities concrete: posterior change detection/cold-start handling
for Full, and replay scheduling/calibration for Replay-only.

## Iteration adoption decision

Adopt the evaluation infrastructure and first-class baseline because source
integrity, method isolation, real provider execution, resource accounting, and
paired reporting all passed. Do not adopt a claim that the present Full
algorithm is superior on the public-source stream. The next algorithm branch
must improve clean cold-start eligibility and change detection while retaining
the isolated-noise false-alarm defense.

## Five-seed same-model mini protocol

Artifact: `runs/sweeps/20260805T060325Z`. The combined report contains exactly
20 completed remote-model runs: Static, Reflexion-style, Replay-only, and Full
EvoShift over paired seeds `[11, 22, 33, 44, 55]`. Every run used the same
`gpt-5.6` alias, public-source-derived 16-episode clean `v1 -> v2` stream,
disabled cache, prompt family, and budget. An incomplete Replay-only artifact
from a terminal gateway `502` was excluded; seeds 44 and 55 were rerun under
the same configuration. Sweep-level per-run checkpoints were then added.

| Method | Score | Changed success | Invariant | Premature update | Requests | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Static | 0.5875±0.1439 | 0.5625±0.4270 (4 seeds) | 0.6402±0.1227 | 0.8333±0.2887 (3 seeds) | 16.0 | 9,576.8 |
| Reflexion-style | 0.7000±0.1027 | 0.8750±0.2500 (4 seeds) | 0.6962±0.0852 | 0.4167±0.2205 (3 seeds) | 20.8 | 19,805.4 |
| Replay-only | 0.6375±0.1618 | 0.5000±0.4564 (4 seeds) | 0.6965±0.1423 | 0.6944±0.0481 (3 seeds) | 69.4 | 51,395.2 |
| Full EvoShift | 0.5875±0.1439 | 0.5000±0.4082 (4 seeds) | 0.6412±0.1464 | 0.6944±0.0481 (3 seeds) | 20.4 | 13,566.6 |

Slice means exclude seeds with zero eligible examples. Seed 55 contained no
changed cases; only seeds 11, 22, 33, and 44 contribute to changed-case
intervals. Only seeds 11, 33, and 55 contained future-change cases and
contribute to premature-update intervals. The comparison artifact records the
exact `seed_list`, `n_seeds`, and `n_pairs` per metric.

Full versus Static produced score delta `0.0000` with 95% hierarchical
bootstrap CI `[-0.1000, 0.0875]`. Changed success delta was `-0.0625`
`[-0.2500, 0.0000]`; invariant delta was `0.0010`
`[-0.0533, 0.0571]`; premature-update delta was `-0.1389`
`[-0.5556, 0.2500]`. Full required 4.4 additional requests
`[2.0, 7.8]` and 3,989.8 additional tokens `[2,154.6, 6,360.6]`.

Full versus Reflexion-style produced score delta `-0.1125`
`[-0.2875, 0.0250]` and premature-update delta `0.2778`
`[0.0000, 0.6389]`, while using 6,238.8 fewer tokens
`[-8,962.0, -3,283.4]`. Replay-only used 49.0 more requests and 37,828.6 more
tokens than Full without a statistically resolved score advantage.

This five-seed result does not support a Full-EvoShift superiority claim.
Reflexion-style has the best mean score and the clearest premature-update
reduction on this short clean stream, although its score interval versus
Static remains wide. Full's present admission and trust stack buys no measured
first-pass gain over Static at a definite resource premium.

All 20 runs share the same recorded Git commit but were executed from the
preregistered experimental branch with a dirty worktree. They are valid
engineering evidence and an auditable interview result, but the matrix must be
rerun from a clean tagged commit before a paper, leaderboard, or formal SOTA
claim.
