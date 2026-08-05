# Experiment: quarantine-to-shadow verified admission

## Status

Pre-registered on branch `codex/quarantine-shadow-admission` after inspecting
the completed five-seed remote-model matrix and before implementation or
benchmark inspection for this candidate. The candidate was subsequently
rejected by the deterministic clean-capability gate and was not promoted to
the live API stage.

## Observed failure

Artifact `runs/sweeps/20260805T060325Z` shows that Full EvoShift ended every
seed with zero active memories. Clean feedback quarantine ranged from `0.5625`
to `0.8125`; Full generated only one to four failure records per seed, evaluated
at most one memory candidate, and confirmed no future audit. The resulting
mean score matched Static at `0.5875` while Reflexion-style reached `0.7000`.

The previous source-level change-point candidate proved that simply raising
cold-start trust is unsafe. A same-source attacker can present a false label on
a previously unseen context with exactly the same learner-visible fields as a
real new rule. No posterior over those fields alone can simultaneously provide
zero-delay learning and zero false acceptance.

## Hypothesis

Low-trust feedback should be allowed to propose a falsifiable shadow
hypothesis without being allowed to update drift, memory utility, replay
labels, or active policy state. A candidate derived from quarantine may advance
only if it passes paired replay over trusted episodes and then the existing
trusted future counterfactual audit.

This separates two decisions that the current runner conflates:

1. Is the observation safe enough to become a training or evaluation label?
2. Is it informative enough to formulate a hypothesis worth testing?

The shadow lane answers only the second question.

## Candidate pipeline

```text
low-trust visible failure
        |
        v
bounded critic extraction
        |
        v
shadow evidence pool -- no direct activation, drift update, or utility update
        |
        v
paired replay on feedback_trust >= 0.60 episodes only
        |
        v
probationary retrieval
        |
        v
trusted future counterfactual audit
        |
        +--> activate
        +--> reject / expire
```

Each candidate must persist total, trusted, and shadow observation counts plus
minimum, maximum, and mean provenance trust. Events and metrics must distinguish
shadow extraction, replay, probation, activation, rejection, and expiry.

## Learner-visible and safety boundary

Shadow extraction may read only the same learner-visible task, model answer,
feedback label, source, context, and current memory summaries already available
to the critic. It must not read oracle correctness, policy version, attack kind,
corruption flags, valid/stale tags, or phase boundaries.

Hard configuration invariants:

- shadow admission requires paired replay;
- shadow admission requires future audit;
- replay and future-audit examples retain their existing `0.60` trust gate;
- a shadow observation cannot directly increment memory utility, update drift,
  activate/supersede a memory, or mutate policy; and
- Static, Reflexion-style, and Replay-only behavior remains unchanged.

## Pre-registered candidate

| Parameter | Value |
|---|---:|
| Shadow admission | enabled |
| Minimum trust for shadow extraction | `0.10` |
| Minimum total observations | `1` |
| Minimum trusted candidate observations | `0` |
| Minimum replay-label trust | `0.60` |
| Paired replay | enabled |
| Future audit | enabled |
| Future-audit minimum observations | `2` |
| Future-audit maximum observations | `4` |

The current Full configuration is the control. Parameters are fixed before
running the candidate matrix and will not be relaxed after inspecting results.

## Coverage-balanced public-source benchmark

The 16-episode Tau3-derived mini protocol accidentally produced no changed
cases for seed 55 and no future-change cases for two seeds. Add an opt-in
coverage-balanced sampler that, for every phase and every non-empty oracle
slice, selects at least two transition, future-change, and protected examples
when the configured phase size permits it. Selection remains deterministic,
seeded, public-source-derived, and invisible in solver prompts.

The sampler must report per-phase slice counts and fail closed when a requested
coverage constraint is mathematically impossible. The existing rotated sampler
remains the backward-compatible default.

## Deterministic evaluation

Compare current Full and shadow admission over paired seeds
`[11, 22, 33, 44, 55]` under:

- clean;
- 10% incidental noise;
- premature-update burst; and
- noise plus burst.

Primary metrics are score, changed success, invariant retention, premature
update, poison persistence, corrupted-feedback follow/quarantine, false
retirement, harmful active-memory exposure, shadow candidate funnel counts,
requests, and tokens.

## Adoption gates

Adopt only if all conditions hold:

- deterministic clean score and changed success do not regress;
- burst premature-update and poison-persistence rates do not worsen by more
  than `0.05` absolute;
- corrupted-feedback follow and false-retirement rates do not worsen by more
  than `0.05` absolute;
- every shadow-derived promotion has both a passing trusted replay decision and
  a completed trusted future audit;
- at least one condition demonstrates a non-zero shadow-to-probation funnel;
- balanced sampling satisfies its declared slice coverage for all five seeds;
- live balanced mini score and changed success do not regress against the
  same-model current-Full control;
- capability, safety, and total API resources are all reported; and
- all tests, branch coverage, Ruff, strict mypy, and builds pass.

If this candidate creates hypotheses but none can pass trusted replay, retain
the control and next add a confidence-sequence evidence accumulator over shadow
hypotheses. If it promotes poisoned hypotheses, the next benchmark/algorithm
must add an independent delayed verification source because the single-source
problem is not identifiable from immediate labels alone.

## Deterministic result

Artifact: `runs/sweeps/20260805T071328Z`. The matrix contains 40 completed demo
runs: two variants, five paired seeds, two noise levels, and two attack-burst
lengths. All 20 shadow runs maintained the required replay-to-probation-to-
future-audit chain, and no shadow candidate used the unverified promotion path.

| Noise | Burst | Variant | Score | Changed | Invariant | Premature | Poison persistence | Requests | Tokens |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | current Full | 0.9167 | 0.6562 | 1.0000 | 0.0373 | 0.0000 | 288.0 | 83,938.0 |
| 0.00 | 0 | shadow admission | 0.9028 | 0.5938 | 1.0000 | 0.0373 | 0.0000 | 375.0 | 109,232.8 |
| 0.00 | 2 | current Full | 0.9167 | 0.6562 | 1.0000 | 0.0373 | 0.0000 | 290.0 | 84,989.0 |
| 0.00 | 2 | shadow admission | 0.9028 | 0.5938 | 1.0000 | 0.0373 | 0.0000 | 379.0 | 111,420.2 |
| 0.10 | 0 | current Full | 0.8958 | 0.5625 | 1.0000 | 0.0371 | 0.0000 | 289.6 | 87,087.8 |
| 0.10 | 0 | shadow admission | 0.8889 | 0.5250 | 1.0000 | 0.0294 | 0.0000 | 462.4 | 143,374.8 |
| 0.10 | 2 | current Full | 0.8931 | 0.5500 | 1.0000 | 0.0371 | 0.0000 | 307.4 | 92,395.2 |
| 0.10 | 2 | shadow admission | 0.8764 | 0.4813 | 1.0000 | 0.0445 | 0.0500 | 487.6 | 150,862.4 |

For clean feedback, the paired hierarchical bootstrap delta was `-0.0139`
with 95% interval `[-0.0236, -0.0069]` for score and `-0.0625` with interval
`[-0.1000, -0.0250]` for changed-case success. The regression occurred in all
five seeds. The candidate therefore fails both clean adoption gates despite
unchanged invariant retention, premature-update rate, poison persistence, and
false-retirement rate.

The shadow funnel was active rather than vacuous. Under clean feedback it
averaged eight shadow replay attempts, four probations, three activations, and
five replay/audit rejections per seed. Across the full matrix it consumed
`8,520` requests and `2,574,451` tokens versus `5,875` requests and `1,742,050`
tokens for the control.

## Failure analysis

The main failure is scheduling interference, not direct poisoned promotion.
The same candidate signature can first appear as a low-trust future-policy
hypothesis. Its early failed replay updates the shared validation observation
count and cooldown. When the first trusted transition observation later
arrives, that genuinely useful evidence is deferred by the earlier shadow
attempt. In the clean seed-11 trace, current Full had the `v2` refund memory
available for the changed cases at indices 30 and 126; shadow admission missed
both because the shared candidate clock delayed probation. This exactly
accounts for the two additional errors (`2 / 144 = 0.0139`) and the changed-
case loss (`2 / 32 = 0.0625`).

The next candidate must isolate shadow and trusted validation clocks. A shadow
hypothesis may accumulate anytime-valid confidence evidence, but a newly
arriving trusted observation must be able to trigger the normal trusted replay
path immediately. This preserves the baseline fast path while retaining the
shadow lane as a non-blocking source of hypotheses.

## Coverage-balanced sampler verification

The opt-in Tau3-derived sampler was checked for seeds `11, 22, 33, 44, 55`
with two eight-example phases. Every non-empty transition, future-change, and
protected slice contained at least two examples. Per-phase counts are now
written to run metrics and sweep `matrix.json`/CSV outputs. Construction fails
closed both when total phase capacity is insufficient and when a non-empty
slice contains fewer unique cases than requested.

## Decision

Reject quarantine-to-shadow admission as implemented and retain current Full.
The paid balanced live mini was intentionally not run because the pre-
registered deterministic clean gate had already failed. Preserve this branch
as an auditable negative result; do not merge it into `main`.

Final branch quality evidence: 147 tests passed with 84.47% branch-aware
coverage; Ruff lint and formatting, strict mypy over 51 source files, and both
sdist and wheel builds passed.
