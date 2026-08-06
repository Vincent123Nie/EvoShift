# Contradiction-aware fast retirement confirmation

## Question

Can an agent retire a stale active memory one audit earlier after a real policy
change, while refusing to treat a noisy or adversarial observation as proof?
The experiment targets the evolution problem of *selective forgetting*: remove
the obsolete successor quickly, preserve protected rules, and leave enough
lineage information to recover a valid predecessor later.

The implementation is deliberately model-agnostic. It changes only the
learner-visible retirement transaction and the dormant-memory eligibility gate;
the task model, prompts, sample order, scorer, and feedback stream are held
constant.

## Hypothesis and variants

The baseline (`sequential`) requires two temporally separated,
learner-visible current-over-old observations before confirming retirement.
The candidate (`contradiction_aware_fast_confirmation`) adds a narrow fast
path:

1. the observation is current-over-old and has trust at least `0.92`;
2. it is separated from registration by `min_evidence_span` episodes;
3. no veto or contradiction has been recorded for the pending retirement.

The first qualifying observation then confirms retirement. The normal path is
unchanged when any gate fails.

Two safeguards make the shortcut reversible rather than blind:

- A low-trust observation that favors the old memory is recorded in
  `contradiction_indices` and blocks the fast path. It is not silently thrown
  away, so a later audit can still resolve the pending transaction.
- A fast-confirmed retirement is tagged by exact memory version. If that card
  is a lineage successor (`supersedes_memory_ids` is non-empty), dormant
  revival requires an additional 13 episodes. Base memories keep the ordinary
  revival age, and a new lifecycle clears the tag.

The feature is off by default. The final candidate settings are:

```yaml
evolution:
  active_audit_retirement_probation_fast_confirm_enabled: true
  active_audit_retirement_probation_fast_confirm_min_trust: 0.92
  active_audit_retirement_probation_fast_revival_cooldown_episodes: 13
```

All online decisions use learner-visible feedback. Oracle stale/valid tags are
used only for post-hoc metric attribution.

## Parameter selection

The threshold grid used 20 seeds on the noisy and attack-burst slices. The
`0.90` candidate had the highest point score, but its burst-slice promotion
precision fell below the sequential control (`0.9258` versus `0.9300`) and
harmful promotion rose (`0.0642` versus `0.0600`). It was therefore rejected.
The `0.94` and `0.96` candidates were safe but more conservative and had lower
score/resource gains. The preregistered choice is `0.92`, the lowest threshold
that kept promotion precision and harmful promotion at least as good as the
control on the stress slice. The five recurring precision-failure seeds in the
burst slice were `133, 155, 233, 277, 333`; they occur in the sequential
control as well and are not hidden by threshold selection.

## Final paired comparator

Artifact: `runs/sweeps/20260806T041158Z` (160 runs: two variants × 20 seeds ×
two feedback-noise levels × two attack-burst lengths). Intervals are the
runner's 2,000-replicate 95% seed-then-paired-sample bootstrap; deltas are
candidate minus sequential control.

| Noise / burst | Score delta | Invariant-retention delta | Old-leakage delta | Requests delta | Tokens delta |
|---|---:|---:|---:|---:|---:|
| clean / 0 | `+0.00694 [0.00417, 0.01007]` | `0.00000 [0.00000, 0.00000]` | `-0.03125 [-0.04531, -0.01875]` | `-19.00 [-19.00, -19.00]` | `-5,247 [-5,247, -5,247]` |
| noisy / 0 | `+0.00486 [0.00139, 0.00972]` | `+0.00348 [0.00000, 0.00984]` | `-0.01250 [-0.02504, -0.00313]` | `-9.20 [-14.35, -4.50]` | `-2,568 [-4,037, -1,204]` |
| clean / 2 | `+0.00694 [0.00417, 0.01007]` | `0.00000 [0.00000, 0.00000]` | `-0.03125 [-0.04531, -0.01875]` | `-19.00 [-19.00, -19.00]` | `-5,247 [-5,247, -5,247]` |
| noisy / 2 | `+0.00417 [0.00104, 0.00868]` | `+0.00290 [0.00000, 0.00928]` | `-0.01094 [-0.02344, -0.00156]` | `-7.35 [-11.60, -3.45]` | `-2,103 [-3,392, -899]` |

At the aggregate level, the candidate score is `0.9514` on clean streams,
`0.9354` on noisy/no-burst streams, and `0.9274` on noisy/burst streams,
versus `0.9444`, `0.9306`, and `0.9233` for sequential. Invariant retention
is `1.0000` clean and `0.9971--0.9976` under noise. Retirement-probation
false-confirmation is `0.0000` in all 160 runs. Replay-estimated promotion
precision remains `1.0000`; realized precision and harmful-promotion rate are
unchanged from the control on every paired condition (the stress slice is
`0.9300` and `0.0600`, respectively).

The main gain is therefore a capability/safety improvement with fewer control
audits, not a claim that the agent has solved adversarial feedback. The attack
burst result is a safety non-regression, not a superiority result.

## Reproduction

```bash
.venv/bin/evoshift sweep \
  --spec configs/sweeps/policy_shift_fast_retirement_comparator.yaml
.venv/bin/pytest -q
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/mypy src
```

The targeted and diagnostic specifications are retained for a quick smoke and
the 20-seed mechanism check. Generated `runs/` artifacts are intentionally
ignored by Git; the run IDs above are the provenance anchors for this report.

## Known limitations and next test

The fast path still relies on a fixed trust floor and a fixed evidence span.
It does not estimate a posterior over change points, model correlated attack
bursts, or produce an anytime-valid confidence sequence. The five burst-slice
failure seeds show why promotion precision is only tied with sequential. The
next principled extension is to replace the fixed one-observation shortcut
with a change-point posterior (or confidence-sequence boundary) and to report
the latency/precision frontier across burst lengths 0, 2, and 4. Any such
extension must pass the same paired safety gates before adoption.

This is deterministic PolicyShift mechanism evidence, not a public-model
leaderboard or SOTA claim. A real OpenAI-compatible model run is a separate
confirmatory experiment and must use environment-provided credentials only.
