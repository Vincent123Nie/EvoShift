# Trust-weighted candidate evidence

Status: rejected after the pre-registered targeted diagnostic; not eligible for
`main`.

## Motivation

The one-mature gate did not change outcomes because a provisional candidate was
released after one later mature observation. Requiring two full mature
observations improved burst slices but delayed genuine clean updates. A hard
integer count cannot express that initial consensus is informative but less
reliable than evidence following a committed regime.

## Frozen candidate

Use a bounded evidence score in the candidate pool:

- `dynamic_consistent`, `dynamic_confirmed_change`, and static provenance each
  contribute `1.0`;
- `dynamic_cold_start`, `dynamic_initial_consensus`, and unresolved pending
  changes contribute `0.5` by default;
- replay requires weighted evidence at least `2.5`.

The weights are learner-visible trust-state categories, not hidden labels. The
candidate does not read policy version, attack/noise annotations, oracle score,
or benchmark phase. The default threshold remains `0.0` and preserves current
behavior.

## Targeted matrix and gates

Compare the current contradiction-aware fast-retirement stack with
`candidate_min_weighted_mature_evidence: 2.5` on seeds
`[133, 155, 233, 277, 333]`, 10% noise, and attack bursts `0`, `2`, and `4`
(30 paired deterministic runs).

Proceed to fresh confirmation only if:

- score and changed-case success do not regress on burst `0`;
- score improves on burst `2` or `4` with no invariant-retention regression;
- realized promotion precision or harmful-promotion rate strictly improves on
  a burst slice;
- at least one initial-consensus-only replay is blocked; and
- every staged candidate event reports its weighted evidence value.

Requests/tokens are secondary diagnostics. This is mechanism evidence, not a
public-model SOTA claim.

## Reproduction

```bash
.venv/bin/evoshift sweep \
  --spec configs/sweeps/policy_shift_trust_weighted_candidate_evidence_targeted.yaml
```

## Targeted result and decision

Artifact: `runs/sweeps/20260806T050442Z` (30 paired runs).

| Burst | Score delta | Changed delta | Realized-promotion-precision delta | Harmful-exposure delta |
|---:|---:|---:|---:|---:|
| 0 | `-0.0250` | `-0.1063` | `-0.0800` | `0.0000` |
| 2 | `+0.0069` | `-0.0125` | `+0.1300` | `-0.2000` |
| 4 | `+0.0097` | `-0.0063` | `0.0000` | `-0.0400` |

The weighted implementation was active and wrote the evidence value into
candidate, staging, and future-audit events. It reproduced the two-mature
candidate's burst behavior but regressed clean adaptation more strongly. The
candidate is rejected and the branch must not be merged.

This is a constructive negative result: when the same observable source can
emit either a new policy label or a coordinated burst with identical timing,
any purely online maturity threshold must trade adaptation delay for safety.
The next principled improvement is an independent delayed verification lane
(for example a second source, held-out user outcome, or future counterfactual
query), not another fixed admission threshold.
