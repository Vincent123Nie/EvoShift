# Context-scoped candidate admission

Status: rejected after the pre-registered targeted gate; not eligible for
`main`.

## Failure being addressed

The current candidate pool aggregates identical memory proposals by textual
signature only. That is useful for deduplication, but it can fuse evidence from
different observable feedback contexts. In the 10% noise + attack-burst
diagnostic, seed `133` produced a `premium_exception` proposal from an attack
context and then added a protected/noise observation from a different context.
The fused candidate passed replay and entered future probation; its later
oracle delta was negative. This is an evidence-association error, not a
retrieval error.

## Hypothesis

When dynamic feedback trust is enabled, candidate evidence should be aggregated
by the pair `(candidate_signature, feedback_source, feedback_context)`. This
prevents an attack/noise observation in one context from satisfying the
minimum-evidence gate for an otherwise identical proposal in another context.
The memory itself remains globally versioned and the lifecycle/retirement
transaction is unchanged. Only candidate-evidence aggregation is scoped.

The baseline keeps the existing global signature pool. The candidate variant
sets:

```yaml
evolution:
  candidate_evidence_context_scoped: true
```

The online path uses only the typed learner-visible source/context already
used by the trust model. Hidden policy versions, phase labels, attack/noise
annotations, stale/valid tags, and oracle scores are not read.

## Pre-registered gates

The candidate must satisfy all of the following on every condition before
adoption:

- realized promotion precision is no lower than global aggregation, with a
  strict improvement on at least one noisy/burst condition;
- harmful promotion rate and false-retirement rate do not increase;
- invariant retention and changed-case success do not decrease;
- overall score has a non-negative paired 95% CI on clean, noisy, and burst
  slices;
- candidate evidence contexts are present in every context-scoped admission
  event; no hidden-label firewall check changes.

The benchmark is two variants × 20 fixed seeds × noise rates `0.0`, `0.10`,
and `0.25` × attack bursts `0`, `2`, and `4` (360 paired runs). A 12-run
targeted slice on seeds `233` and `255` is run first. Bootstrap comparisons
resample seed clusters and paired episodes; run-level promotion metrics are
reported separately because they are not episode-level quantities.

## Reproduction

```bash
.venv/bin/evoshift sweep \
  --spec configs/sweeps/policy_shift_context_scoped_candidate_admission_targeted.yaml
.venv/bin/evoshift sweep \
  --spec configs/sweeps/policy_shift_context_scoped_candidate_admission.yaml
```

This is a deterministic mechanism benchmark. It is not a public-model
leaderboard claim. A real API-model confirmation must use the same seed list,
sample order, and cost accounting after the deterministic gates pass.

## Targeted result and decision

Artifact: `runs/sweeps/20260806T043759Z` (12 runs: two variants, seeds `233`
and `255`, 10% noise, and attack-burst lengths `0`, `2`, and `4`).

| Burst | Score delta | Promotion-precision delta | Requests delta | Tokens delta |
|---:|---:|---:|---:|---:|
| 0 | `0.0000` | `0.0000` | `+8` | `+2,429` |
| 2 | `0.0000` | `0.0000` | `+8` | `+2,430` |
| 4 | `0.0000` | `0.0000` | `+8` | `+2,427` |

The candidate failed the pre-registered requirement for a strict
promotion-precision improvement on at least one noisy/burst condition. The
full 360-run matrix and live-model confirmation were therefore not run.

Event inspection confirmed that the option was active rather than vacuous.
Identical signatures were split into separate source/context evidence buckets,
which caused additional validation attempts. However, the configured minimum
candidate evidence remained one observation, so each bucket could still enter
replay independently. The downstream replay and future-audit decisions were
therefore unchanged while resource usage increased.

The next candidate should distinguish evidence maturity rather than only its
context key. In particular, a proposal supported solely by a dynamic initial
consensus should remain provisional until at least one learner-visible
observation is consistent with an already committed context, or confirms a
change from a previously committed state.
