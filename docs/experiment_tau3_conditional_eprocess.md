# Pre-registration: Tau3 conditional shadow e-process

Status: pre-registered before development or confirmation API execution.

## Question

Can shadow evidence be conditioned on a learner-visible feedback family so that
unrelated business contexts stop diluting recurrence evidence, while competing
signals inside the same context remain negative evidence?

The family key is a hash of `(domain, feedback_source, feedback_context)`. The
cluster key additionally includes `feedback_signal`. For an existing cluster:

- the same family and same signal is positive evidence;
- the same family and a different signal is negative evidence;
- a different family is skipped as unrelated.

The key excludes policy version, reference answer, corruption labels, valid or
stale memory tags, and all other hidden evaluator metadata. Exact and historical
hierarchical behavior remain available as unchanged comparators. The conditional
path is default-off and requires both exact and hierarchical e-process flags.

## Frozen protocol

- Public source: pinned `tau3-bench` retail policy/tasks revision `v1.0.1`,
  transformed by the checked-in PolicyShift adapter.
- Stream: `v1 -> v2 -> v3 -> v2`, 24 balanced examples per phase, 96 episodes.
- Feedback: clean capability control, no attack burst, shared support-portal
  source, six consistent observations required for source confirmation.
- Model: `gpt-5.6` through the configured OpenAI-compatible Responses gateway.
- Variants: Exact, historical Hierarchical, and Conditional Hierarchical.
- Fixed e-process parameters: `q0=0.25`, `q1=0.75`, `alpha=0.05`. They are not
  tuned using the earlier Tau3 matrix or either new stage.
- Development seeds: `[66, 77]`, using a fresh dedicated shared cache.
- Held-out confirmation seeds: `[88, 99, 111]`, using a second fresh cache.
- Identical logical requests within a stage share cached model responses.
  Request and token counts are therefore not efficiency claims.

Development is used only to catch implementation failure. The confirmation
configuration, seeds, metrics, and gates are frozen before development starts.
No threshold or family-key change may be adopted after reading development
results without a new pre-registration and new held-out seeds.

## Gates

The protocol is valid only when all variant pairs have aligned samples, matching
dataset hashes, one clean Git commit and model, the expected shared cache, and
resolved configurations that differ only in the two e-process mode flags.

Conditional evidence passes a stage only if:

1. it produces at least one cluster crossing and one shadow-only replay;
2. mean overall score and changed-case success are no lower than both Exact and
   historical Hierarchical;
3. paired 95% intervals show invariant retention no worse than `-0.02`, while
   old-rule leakage, premature update, harmful active-memory exposure, and false
   retirement worsen by no more than `+0.02` against both comparators;
4. harmful active-memory exposure is zero, and every activation is covered by a
   confirmed future audit.

A zero-crossing, neutral, or negative result is retained. Poison persistence is
not applicable because this is a clean-feedback capability control.

Run:

```bash
rm -f data/llm_cache_tau3_conditional_eprocess_development.sqlite3
evoshift sweep --spec configs/sweeps/tau3_retail_conditional_eprocess_development.yaml
python scripts/audit_tau3_conditional_eprocess.py runs/sweeps/<id> --stage development

rm -f data/llm_cache_tau3_conditional_eprocess_confirmation.sqlite3
evoshift sweep --spec configs/sweeps/tau3_retail_conditional_eprocess_confirmation.yaml
python scripts/audit_tau3_conditional_eprocess.py runs/sweeps/<id> --stage confirmation
```

Interrupted sweeps resume without rerunning completed assignments:

```bash
evoshift sweep --spec <same-spec> --resume runs/sweeps/<id>
```
