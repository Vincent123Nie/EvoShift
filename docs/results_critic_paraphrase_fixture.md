# Result: deterministic critic-paraphrase fixture

Status: the preregistered deterministic mechanism gate passed. The hierarchical
feature remains default-off and is not claimed as SOTA or adopted production behavior.

## Provenance

- Preregistration commit: `d953c72`.
- Clean implementation commit: `56e0bb8`.
- Sweep spec: `configs/sweeps/critic_paraphrase_fixture.yaml`.
- Fixture: `examples/critic_paraphrase_fixture.jsonl`.
- Clean local sweep: `runs/sweeps/20260810T092735Z`.
- Dataset hash:
  `4ad6d52c7e6b19c65d65210a4d7c4e737ceb7bc3bb51bdbd23bc22f94112c18f`.

Both manifests recorded commit `56e0bb8`, `git_dirty=false`, the same dataset hash,
model, budget, and episode order. Reproduce and verify with:

```bash
evoshift sweep --spec configs/sweeps/critic_paraphrase_fixture.yaml
python scripts/verify_critic_paraphrase_fixture.py runs/sweeps/<sweep-id>
```

## Fixture

The 19-episode stream contains 11 protected day-1-to-7 refund cases and eight
day-8-to-14 policy-change cases. Six initial protected observations establish a
trusted learner-visible context. Five failing policy-change observations then remain
in the low-trust shadow lane while the deterministic critic emits five distinct but
semantically equivalent memory wordings. Trusted protected observations are
interleaved as a neutral safety replay buffer. The sixth change observation confirms
the new feedback context and later observations provide disjoint memory-on/off future
audit evidence.

Only `evolution.shadow_hierarchical_eprocess_enabled` differs between variants.

## Results

| Metric | Exact e-process | Hierarchical e-process | Delta |
|---|---:|---:|---:|
| Overall score | 0.6842 | 0.7895 | +0.1053 |
| Policy-change score | 0.2500 | 0.5000 | +0.2500 |
| Protected-anchor score | 1.0000 | 1.0000 | 0.0000 |
| Exact e-process crossings | 0 | 0 | 0 |
| Cluster e-process crossings | N/A | 1 | +1 |
| Shadow-only replay attempts | 0 | 1 | +1 |
| Shadow-derived probations | 0 | 1 | +1 |
| Shadow-derived activations | 0 | 1 | +1 |
| Future-audit confirmations | 1 | 1 | 0 |
| Harmful promotion rate | 0.0000 | 0.0000 | 0.0000 |
| Harmful active-memory exposure | 0 | 0 | 0 |
| Total requests | 35 | 33 | -2 |
| Total tokens | 10,454 | 9,709 | -745 |

The paired per-episode bootstrap score interval was `[0.0000, 0.2632]`; it includes
zero and must not be presented as population-level significance. The deterministic
gain is exactly two additional correct episodes out of 19. The hierarchical path
entered neutral, protected-safe probation before feedback trust reached the ordinary
candidate threshold, then obtained a `+1.0` learner-visible and oracle future-audit
delta. The exact path learned later through the unchanged trusted lane.

All preregistered deterministic gates passed through
`scripts/verify_critic_paraphrase_fixture.py`. The hierarchical run used fewer total
requests and tokens because earlier probation prevented two later failure-attribution
cycles.

## Interpretation

This result validates the intended mechanism that exact candidate identity can
fragment under critic wording drift while a code-derived learner-visible anchor can
accumulate evidence and safely reduce adaptation delay. It does not establish that
the anchor is well calibrated for arbitrary model paraphrases, noisy feedback,
adversarial sources, or public workloads.

The result is deliberately narrower than the earlier 40-run confirmation in
`results_hierarchical_shadow_eprocess.md`, which had nonzero opportunities but no
cluster crossings and zero uplift. Together they show both the mechanism's reachable
case and its coverage limitation; the negative result is not superseded.

## Next gate

A bounded real-model mini is now eligible, but it must be separately preregistered.
It should replace the deterministic critic with an OpenAI-compatible model while
freezing the same samples, trust path, solver behavior, cluster-key code, and exact
versus hierarchical comparison. It must report observed critic-signature diversity,
crossing coverage, safety, requests, tokens, and failures without post-hoc threshold
tuning.

