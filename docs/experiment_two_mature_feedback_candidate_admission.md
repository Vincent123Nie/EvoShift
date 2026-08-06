# Two-mature-feedback candidate admission

Status: rejected after the pre-registered targeted diagnostic; not eligible for
`main`.

This is a follow-up to the rejected one-mature-observation candidate in
`experiment_mature_feedback_candidate_admission.md`. The first gate blocked a
candidate supported only by initial consensus, but a single subsequent mature
observation still released it with unchanged promotion precision.

## Frozen hypothesis

Require two mature learner-visible observations before a candidate can enter
replay:

```yaml
evolution:
  candidate_min_mature_feedback_observations: 2
```

The existing evidence pool still records provisional observations and uses the
same textual candidate signature. Mature evidence remains limited to
`dynamic_consistent`, `dynamic_confirmed_change`, or static provenance mode.
No hidden benchmark metadata is available to admission.

## Targeted matrix and gates

Compare the current contradiction-aware fast-retirement stack against the
two-mature gate on seeds `[133, 155, 233, 277, 333]`, 10% noise, and bursts
`0`, `2`, and `4` (30 paired deterministic runs). The candidate may proceed to
fresh seeds only if:

- realized promotion precision strictly improves on burst `2` or `4`;
- score, changed-case success, and invariant retention do not decrease in any
  condition mean;
- harmful-promotion, false-retirement, and harmful active-memory exposure do
  not increase;
- at least one initial-consensus-only candidate is prevented from replay; and
- every admitted candidate has two mature observations in its event trace.

This is a mechanism diagnostic. It does not support a public-model SOTA claim.

## Reproduction

```bash
.venv/bin/evoshift sweep \
  --spec configs/sweeps/policy_shift_two_mature_feedback_candidate_admission_targeted.yaml
```

## Targeted result and decision

Artifact: `runs/sweeps/20260806T045750Z` (30 paired runs).

| Burst | Score delta | Changed delta | Realized-promotion-precision delta | Harmful-promotion delta |
|---:|---:|---:|---:|---:|
| 0 | `-0.0139` | `-0.0563` | `-0.0800` | `+0.0400` |
| 2 | `+0.0069` | `-0.0125` | `+0.1300` | `-0.2000` |
| 4 | `+0.0097` | `-0.0063` | `0.0000` | `-0.0400` |

The strict gate materially reduced burst exposure, but clean capability
regressed with a paired 95% score interval `[-0.0264, -0.0014]` and changed
success interval `[-0.1062, -0.0063]`. It therefore fails the all-condition
capability gate and is not eligible for fresh seeds or merge. The next
candidate replaces the hard count with a weighted evidence threshold.
