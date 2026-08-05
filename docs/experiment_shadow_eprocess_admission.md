# Experiment: lane-isolated shadow e-process admission

## Status

Pre-registered on branch `codex/shadow-eprocess-admission` after the completed
quarantine-to-shadow experiment and before implementing or benchmarking this
candidate. The final fixed candidate passed its deterministic adoption gates,
the uncached live mini, and a five-seed common-response confirmation.

## Prior evidence and failure mechanism

The first shadow-admission candidate failed clean capability despite verified
replay and future audit. A low-trust future-policy hypothesis could fail replay
before a real transition, update the candidate's shared validation count and
cooldown, and then delay the first trusted transition observation. The clean
loss was identical in all five seeds: two additional changed-case errors.

The failure is therefore a shared scheduler-state bug in the algorithmic
design, not evidence that low-trust hypotheses must directly update policy.

## Hypothesis

Separate trusted and shadow scheduling state. A newly arriving trusted
observation must retain the current Full fast path regardless of earlier
shadow attempts. Shadow-only hypotheses should replay only after an
anytime-valid recurrence test crosses a fixed threshold.

For each unresolved shadow signature, candidate extractions form a
prequential binary stream: a matching extraction is `1`, and a different
shadow extraction is `0`. Under an explicit simple null with match probability
`q0`, and a coherent-recurrence alternative with probability `q1`, update the
likelihood-ratio e-process:

```text
E_t = E_(t-1) * (q1 / q0)^x_t * ((1 - q1) / (1 - q0))^(1 - x_t)
```

The first occurrence only creates the hypothesis and is not counted as a
match, avoiding selection bias from starting a process precisely when a match
has already occurred. Replay is permitted when `E_t >= 1 / alpha`. This is an
auditable sequential test under the declared recurrence null, not a claim that
arbitrary malicious feedback is statistically identifiable. The e-process
crossing is latched until the next shadow validation attempt, then the process
resets so repeated attempts require fresh evidence.

## Fixed candidate

| Parameter | Value |
|---|---:|
| Shadow admission | enabled |
| Minimum trust for shadow extraction | `0.10` |
| Minimum trusted candidate observations | `0` |
| E-process | enabled |
| Null recurrence probability `q0` | `0.25` |
| Alternative recurrence probability `q1` | `0.75` |
| Type-I threshold `alpha` | `0.05` |
| Trusted replay threshold | `0.60` |
| Paired replay | enabled |
| Future audit | enabled |

No parameter will be changed after inspecting the candidate matrix.

## Lane isolation invariants

- trusted readiness uses only the trusted validation clock and cannot be
  blocked by a shadow validation cooldown;
- trusted replay begins at the first trusted evidence index, not the first
  earlier shadow hypothesis index;
- shadow readiness uses only shadow observations, the e-process threshold, and
  the shadow validation clock;
- shadow extraction still cannot update drift, memory utility, active-memory
  audit labels, policy, or active memory directly;
- shadow replay and future audit still consume only feedback with trust at
  least `0.60`; and
- all accepted shadow-derived memories still require both replay and future
  counterfactual audit.

## Deterministic evaluation

Compare current Full and lane-isolated e-process admission over paired seeds
`[11, 22, 33, 44, 55]` under clean, 10% noise, burst, and noise-plus-burst
conditions. Report capability, safety, shadow funnel, e-process opportunities
and crossings, requests, and tokens with paired hierarchical intervals.

## Adoption gates

Adopt only if all conditions hold:

- clean score and changed-case success do not regress;
- burst premature-update and poison-persistence rates do not worsen by more
  than `0.05` absolute;
- corrupted-feedback follow and false-retirement rates do not worsen by more
  than `0.05` absolute;
- every shadow-derived activation has passing trusted replay and completed
  trusted future audit;
- at least one condition has a non-zero shadow-to-probation funnel;
- trusted observations are never deferred solely because of shadow cooldown;
- the e-process test has a unit-tested deterministic trace and resets only
  after a shadow validation attempt; and
- all tests, branch coverage, Ruff, strict mypy, and builds pass.

Only after deterministic adoption gates pass will the same-model balanced
Tau3-derived live mini be run. If clean equivalence is restored but the
e-process never creates a shadow-only replay, retain the implementation only
as a safety-preserving hypothesis cache and explicitly report the funnel as
inactive rather than claiming adaptation benefit.

## Pre-registered confirmatory live protocol

The first uncached live mini is the resource measurement. Because two
identical no-memory solver requests can still return different model answers
across sequential variants, a second capability-only confirmatory protocol is
pre-registered before running it:

- seeds `[11, 22, 33, 44, 55]`;
- the same eight-example `v1 -> v2` balanced phases and fixed candidate;
- one shared content-addressed response cache for current Full and the
  candidate, so byte-identical model requests receive the same previously
  sampled real-model response; and
- no resource-efficiency claim from the cached matrix, because external cache
  hits are intentionally not equivalent to logical algorithm calls.

This common-response design isolates changes caused by memory and evolution
state while the uncached mini remains the honest API request/token report.

## Deterministic result

Final artifact: `runs/sweeps/20260805T075926Z`. The matrix contains 40
completed demo runs over two variants, five paired seeds, two noise levels,
and two attack-burst lengths.

| Noise | Burst | Variant | Score | Changed | Invariant | Premature | Poison persistence | Requests | Tokens |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | current Full | 0.9167 | 0.6562 | 1.0000 | 0.0373 | 0.0000 | 288.0 | 83,938.0 |
| 0.00 | 0 | shadow e-process | 0.9167 | 0.6562 | 1.0000 | 0.0373 | 0.0000 | 293.0 | 86,914.8 |
| 0.00 | 2 | current Full | 0.9167 | 0.6562 | 1.0000 | 0.0373 | 0.0000 | 290.0 | 84,989.0 |
| 0.00 | 2 | shadow e-process | 0.9167 | 0.6562 | 1.0000 | 0.0373 | 0.0000 | 297.0 | 89,041.8 |
| 0.10 | 0 | current Full | 0.8958 | 0.5625 | 1.0000 | 0.0371 | 0.0000 | 289.6 | 87,087.8 |
| 0.10 | 0 | shadow e-process | 0.8958 | 0.5625 | 1.0000 | 0.0371 | 0.0000 | 317.8 | 101,175.8 |
| 0.10 | 2 | current Full | 0.8931 | 0.5500 | 1.0000 | 0.0371 | 0.0000 | 307.4 | 92,395.2 |
| 0.10 | 2 | shadow e-process | 0.8958 | 0.5563 | 1.0000 | 0.0294 | 0.0000 | 330.6 | 105,241.2 |

Clean, burst-only, and noise-only paired deltas were exactly zero for score,
changed success, invariant retention, and premature update. Under noise plus
burst, the paired hierarchical score delta was `+0.0028` with 95% interval
`[0.0000, 0.0111]`; changed success was `+0.0063 [0.0000, 0.0250]`; and
premature-update error was `-0.0077 [-0.0308, 0.0000]`. Poison persistence,
corrupted-feedback follow, and false retirement did not worsen.

The safety result is not achieved by disabling the new lane. Under noisy
conditions, the fixed e-process produced `0.6` shadow-only replay attempts per
seed on average after post-discovery recurrence crossings. The noise-only
condition also recorded `0.2` trusted fast-path shadow-cooldown bypasses per
seed. Clean hypotheses still joined later trusted evidence: five shadow-
derived replay attempts, four probations, and three activations per seed, all
through replay and completed future audit. No unverified promotion occurred in
any of the 20 candidate runs.

The cost is explicit. Relative to current Full, mean overhead ranged from five
requests and about 2,977 tokens in clean feedback to 28.2 requests and 14,088
tokens under isolated noise. Latency was not an optimization target for this
iteration.

## Uncached live mini

Artifact: `runs/sweeps/20260805T073245Z`. Both variants used the same remote
model and the balanced 16-episode seed-11 stream.

| Variant | Score | Changed | Invariant | Future-change success | Premature | Requests | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| current Full | 0.5000 | 0.5000 | 0.6000 | 0.2500 | 0.7500 | 20 | 13,732 |
| shadow e-process | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 24 | 17,715 |

The candidate met the pre-registered live score and changed-success
non-regression gate. It extracted five shadow hypotheses but had no recurrence
crossing or probation in this short stream. Two no-memory solver answers
differed across sequential API calls in opposite directions, leaving total
score unchanged while moving one success from the protected slice to the
future-change slice. These differences are model sampling variation, not a
memory effect.

## Five-seed common-response confirmation

Artifact: `runs/sweeps/20260805T074228Z`. The content-addressed cache was empty
before the run and stored only real remote-model responses. Across seeds
`11, 22, 33, 44, 55`, current Full and shadow e-process were episode-exact:
all 80 paired foreground answers and scores matched. Both variants had mean
score `0.6625`, changed success `0.8333`, invariant retention `0.6905`, future-
change success `0.4333`, and premature-update rate `0.5667`.

The candidate still extracted `3.2` shadow hypotheses per seed, but signature
recurrence never crossed the fixed e-process threshold and no live candidate
reached replay. This matrix is therefore capability/safety evidence for a
non-interfering hypothesis cache, not evidence of live adaptation benefit.
Cached external request counts are intentionally excluded from efficiency
claims; the uncached mini and deterministic matrix provide resource numbers.

## Decision

Adopt lane-isolated shadow e-process admission. It repairs the clean regression
that rejected the first shadow design, preserves every measured safety metric,
maintains the trusted fast path, and exposes a statistically explicit,
unit-tested route for repeated low-trust hypotheses. The result should be
described conservatively: deterministic noisy streams exercise shadow-only
replay, while the bounded live stream validates non-interference but does not
show an activated shadow memory.

The next research target is semantic signature clustering or a hierarchical
e-process across related candidate formulations. Exact signatures fragment
equivalent live hypotheses, but any clustering rule must be fixed before
evaluation and must not merge conflicting directives.

Final quality evidence: 149 tests passed with 84.43% branch-aware coverage;
Ruff lint and formatting, strict mypy over 51 source files, sdist/wheel builds,
and schema/loading validation for 43 YAML files with 908 expanded sweep
assignments all passed.
