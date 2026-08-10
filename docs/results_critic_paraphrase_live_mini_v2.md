# Result: real-model critic-paraphrase mini v2

Status: completed as a one-seed diagnostic. The upstream hierarchical mechanism
was reached, but shadow admission did not succeed and capability did not improve.
The feature remains default-off.

## Provenance

- Budget-amendment preregistration commit: `1e2b02a`.
- Sweep: `runs/sweeps/20260810T095703Z`.
- Exact run: `20260810T095703Z-evoshift-1e69c693`.
- Hierarchical run: `20260810T100417Z-evoshift-d202dac2`.
- Dataset hash:
  `4ad6d52c7e6b19c65d65210a4d7c4e737ceb7bc3bb51bdbd23bc22f94112c18f`.
- Model: `gpt-5.6`; seed 42; cache disabled.

Both manifests record the same clean Git commit, dataset, model, seed, and
experiment settings. Only
`evolution.shadow_hierarchical_eprocess_enabled` differs.

## Results

| Metric | Exact | Hierarchical | Delta |
|---|---:|---:|---:|
| Overall score | 0.6842 | 0.6842 | 0.0000 |
| Policy-change score | 0.2500 | 0.3750 | +0.1250 |
| Protected-anchor score | 1.0000 | 0.9091 | -0.0909 |
| Critic records | 6 | 6 | 0 |
| Unique critic signatures | 6 | 6 | 0 |
| Unique memory triggers | 6 | 6 | 0 |
| Unique memory directives | 6 | 6 | 0 |
| Structured critic fallbacks | 0 | 0 | 0 |
| Exact e-process crossings | 0 | 0 | 0 |
| Cluster e-process opportunities | N/A | 3 | +3 |
| Cluster e-process crossings | N/A | 1 | +1 |
| Shadow-only replay attempts | 0 | 1 | +1 |
| Shadow-derived probations | 0 | 0 | 0 |
| Shadow-derived activations | 0 | 0 | 0 |
| Shadow-derived rejections | 0 | 1 | +1 |
| Harmful promotion rate | 0.0000 | 0.0000 | 0.0000 |
| Harmful active-memory exposure | 0 | 0 | 0 |
| Requests | 41 | 50 | +9 |
| Total tokens | 299,461 | 286,290 | -13,171 |

The score-slice differences cannot be attributed to the hierarchical flag: the
two uncached runs made independent model calls, so ordinary generation variation
is a confound. The equal overall score is descriptive, not a paired causal
estimate. Token differences are likewise not an efficiency claim.

## Mechanism and safety outcome

The real critic produced six distinct signatures, triggers, and directives in
both variants. Exact candidate identity therefore fragmented and never crossed.
The observable cluster accumulated enough same-context evidence to cross once
and authorize one shadow-only replay.

That shadow candidate failed the unchanged replay gate with mean delta `-0.5`,
CI `[-1.0, 0.0]`, regression rate `0.5`, and protected-slice regression `0.5`.
It was rejected before probation, so no harmful memory became active. A later
trusted candidate passed a neutral replay, but its future audit ended with only
one positive observation, below the fixed minimum of two, and expired without
activation.

This is a partial positive result: live wording diversity and hierarchical
crossing are demonstrated, while downstream admission correctly refuses an
unsafe candidate. It is not evidence of adaptation uplift.

## Next research gate

The next valid comparison should record common model responses for the shared
solver/critic prefix, then replay those responses across exact and hierarchical
variants. After that mechanism-controlled run, use multiple seeds and a public
policy-drift stream. Do not tune the e-process threshold from this one-seed
result; the immediate algorithmic issue is semantic candidate quality and
protected replay coverage, not crossing sensitivity.
