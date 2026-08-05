# Experiment: source-level change-point feedback trust

## Status

Pre-registered on branch `codex/change-point-feedback-trust` after the live
Tau3Retail-PolicyDrift mini diagnostic and before implementation or benchmark
inspection.

## Observed failure

The Full EvoShift run `20260805T031458Z-evoshift-34f26485` quarantined 9 of 16
clean observations. The existing consistency model treats every unseen context
as untrusted until two matching labels arrive, and it requires two
contradictions independently inside every changed context. This is conservative
against isolated label noise but poorly matched to sparse enterprise policy
families.

The failure is structural rather than an LLM-only error:

- first observations receive trust `0.40`, below every `0.60` update gate;
- the first changed label receives trust `0.10` even when another context has
  just confirmed the same source-level policy change; and
- evidence about a shared regime change does not transfer across contexts.

## Hypothesis

A source-level Bayesian change posterior can share change evidence across
contexts while keeping isolated contradictions below the update threshold.
Separating initial context establishment from later change detection should
also remove unnecessary clean cold-start quarantine.

The proposed mode is not a claim of exact Bayesian online change-point
detection over an unrestricted generative process. It is an auditable
Beta-reliability, Bernoulli change/no-change filter with a configured hazard,
posterior threshold, and bounded post-change grace window.

## Algorithm

Each observable feedback source maintains:

- the existing Beta reliability posterior;
- a probability that a source-level regime change is currently in progress;
- a monotonically increasing regime index; and
- a bounded number of post-change grace observations.

For a context with an established label, let `r` be the clipped source
reliability and `h` the change hazard. Before an observation:

```text
p_prior = p_previous + (1 - p_previous) * h
```

For a contradictory label:

```text
p_change = p_prior * r /
           (p_prior * r + (1 - p_prior) * (1 - r))
```

For a label consistent with the committed regime, the two likelihoods are
swapped. An isolated contradiction remains pending and low-trust. When the
posterior crosses the threshold, all pending contradictions from the same
source are committed together, the source regime increments, and a bounded
grace window lets the first matching contradiction in another established
context transfer into the newly detected regime.

For a previously unseen context, the first label is committed immediately but
receives capped cold-start trust rather than full source trust. This enables
learning on sparse clean contexts while retaining paired replay and future
audit as downstream defenses. Because that first label has no within-context
comparison, it does not update source reliability or the change posterior.

## Learner-visible inputs and oracle boundary

The filter may read only:

- feedback source;
- configured observable context;
- learner-visible feedback label; and
- its own prior state.

It must not read `feedback_kind`, corruption annotations, attack goals, hidden
oracle labels, phase indices, policy versions, or valid/stale memory tags.
Change probability, source regime, grace observations, and decision reason
must be stored per episode for interview-grade traces.

## Configured candidate

Initial values to evaluate, not tune on final results:

| Parameter | Value |
|---|---:|
| Mode | `change_point` |
| Change hazard | `0.15` |
| Posterior threshold | `0.80` |
| New-context trust cap | `0.70` |
| Post-change grace observations | `8` |
| Reliability likelihood clip | `[0.55, 0.99]` |

The existing `consistency` mode remains the backward-compatible control.

## Deterministic tests

1. Sparse clean contexts: one observation per context is eligible in
   change-point mode but quarantined in the current mode.
2. Isolated contradiction: a single conflicting label does not cross the
   change threshold and is quarantined.
3. Recovered contradiction: return to the old label clears pending evidence
   and reduces the change posterior.
4. Shared change: contradictory labels across contexts confirm one source-level
   change sooner than independent two-observation context gates.
5. Grace transfer: after confirmation, one contradiction in another context is
   accepted and commits only that learner-visible label.
6. Oracle isolation: changing hidden annotations while holding observable
   fields fixed cannot change the assessment sequence.

## Benchmarks

### Deterministic PolicyShift

Compare `consistency` and `change_point` across five paired seeds under:

- clean;
- 10% incidental noise;
- premature-update burst only; and
- noise plus burst.

Primary metrics are score, changed success, premature update, corrupted
feedback quarantine, clean feedback quarantine, poison persistence, false
retirement, harmful exposure, requests, and tokens.

### Live Tau3Retail-PolicyDrift mini

Repeat the same 16-episode seed-11 stream from the established failure trace,
holding model alias, prompt, cache, and budget constant. This run is a bounded
engineering validation, not the five-seed final result.

## Adoption gates

Adopt change-point mode only if all conditions hold:

- clean feedback quarantine falls from `0.5625` to at most `0.25` in the live
  mini diagnostic;
- deterministic clean changed-case success and overall score do not regress;
- premature-update and poison-persistence rates do not worsen by more than
  `0.10` absolute under the burst conditions;
- false retirement does not increase by more than `0.05` absolute;
- the stored trace proves source-level evidence sharing without oracle access;
- all tests, Ruff, strict mypy, branch coverage, and builds pass; and
- resource changes are reported even when capability improves.

If the posterior candidate fails the safety gates, retain the current mode and
use an anytime-valid confidence sequence as the next candidate rather than
lowering thresholds after seeing final results.

## Deterministic five-seed result

Artifact: `runs/sweeps/20260805T033853Z`. The matrix contains 40 demo-provider
runs: two trust modes, five paired seeds, and the clean/noise/burst/
noise-plus-burst Cartesian product. These runs validate causal behavior and
safety plumbing; they are not remote-model capability evidence.

| Condition | Mode | Score | Changed | Premature | Poison persistence | Clean quarantine | Corrupt quarantine | Requests | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | consistency | 0.9167 | 0.6562 | 0.0373 | 0.0000 | 0.0694 | 0.0000 | 288.0 | 83,938.0 |
| clean | change-point | 0.9375 | 0.7500 | 0.0373 | 0.0000 | 0.0069 | 0.0000 | 274.0 | 81,360.8 |
| burst | consistency | 0.9167 | 0.6562 | 0.0373 | 0.0000 | 0.0714 | 0.5000 | 290.0 | 84,989.0 |
| burst | change-point | 0.9167 | 0.7500 | 0.1494 | 0.3750 | 0.0143 | 0.0000 | 348.0 | 105,472.2 |
| noise | consistency | 0.8958 | 0.5625 | 0.0371 | 0.0000 | 0.0949 | 0.8125 | 289.6 | 87,087.8 |
| noise | change-point | 0.9014 | 0.6188 | 0.0750 | 0.0000 | 0.0378 | 0.2400 | 483.2 | 145,432.8 |
| noise + burst | consistency | 0.8931 | 0.5500 | 0.0371 | 0.0000 | 0.0979 | 0.7415 | 307.4 | 92,395.2 |
| noise + burst | change-point | 0.8792 | 0.5813 | 0.1493 | 0.3714 | 0.0519 | 0.2857 | 481.4 | 147,052.0 |

The clean hypothesis passed: score and changed success improved while clean
quarantine fell. The safety hypothesis failed. Under burst-only feedback,
premature update worsened by `0.1120`, above the preregistered `0.10` limit,
and poison persistence worsened from `0.0000` to `0.3750`. Lower corrupted
quarantine in these rows is not a benefit: the filter accepted poisoned labels
as a shared source-level regime change.

## Live same-model mini result

Artifact: `runs/sweeps/20260805T034017Z`. Both runs used the remote `gpt-5.6`
alias, the same seed-11 16-episode public-source-derived stream, disabled
cache, and identical budgets.

| Mode | Score | Changed | Invariant | Premature | Clean quarantine | Requests | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| consistency | 0.6250 | 0.5000 | 0.7500 | 0.5000 | 0.5625 | 27 | 17,942 |
| change-point | 0.5625 | 0.5000 | 0.7500 | 0.7500 | 0.0625 | 63 | 48,118 |

The live clean-quarantine gate passed, but capability, premature-update, and
resource behavior did not. The candidate reduced quarantine by `0.50`
absolute while reducing score by `0.0625`, worsening premature update by
`0.25`, adding 36 requests, and adding 30,176 tokens.

The stored trace is auditable. Episode 12 recorded one pending contradiction
with change probability `0.6073` and trust `0.10`. Episode 13 confirmed source
regime 1 and opened an eight-observation grace window. Episode 14 used that
grace to transfer the learner-visible new label into another context. No
oracle phase, corruption, or attack field was read by the filter.

## Adoption decision

Reject `change_point` as the default trust mode and retain `consistency`.
Keep the implementation behind an explicit experimental configuration because
it is a useful, reproducible failure case: source-level evidence sharing fixes
sparse cold start but collapses isolated-noise protection when a coordinated
or repeated false label resembles a regime shift.

The next preregistered candidate is an anytime-valid confidence sequence or a
source/context hierarchical detector that requires cross-context diversity,
not merely repeated contradictions. It must explicitly control false regime
commit probability under burst feedback before optimizing detection delay.
