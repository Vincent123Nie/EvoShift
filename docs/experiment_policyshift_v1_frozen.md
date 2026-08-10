# PolicyShift v1 frozen live-model comparison

## Status and claim boundary

This is the v1 engineering freeze for the public-source-derived
`Tau3Retail-PolicyDrift` stream. It is not an official tau3-bench score: the
stream is derived from tau2-bench retail policy/tasks and uses an EvoShift
policy overlay rather than the official interactive tool environment.

The matrix is retained as evidence, including its negative result. It is a
same-model, five-seed paired comparison, but its raw run manifests were
created with `git_dirty=true` on historical commit
`5520a9a0deb9fd6e4adfe6d2c5442e7edbc047e9`. It therefore supports an honest
engineering comparison, not a clean-current-commit reproducibility claim.

## Frozen protocol

Artifact: `runs/sweeps/20260805T060325Z` (`matrix.json`, `matrix.csv`,
`report.md`). The matrix has exactly 20 runs: Static, Reflexion-style,
Replay-only, and Full EvoShift for seeds `[11, 22, 33, 44, 55]`. All rows use
the returned model identifier `gpt-5.6`, the same source-derived dataset
family, disabled cache, the same prompt/budget policy, and the same ordered
`v1 -> v2` two-phase stream with `phase_size=8`.

This is the short live-model freeze, not the larger six-phase stress stream in
`configs/experiments/tau3_retail_policy_shift_live.yaml`. The latter remains a
future paid evaluation because it is substantially more expensive.

The public source is pinned in
`docs/experiment_public_policy_drift_real_model.md`:

- tau3-bench tag `v1.0.1`, peeled commit
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`;
- retail policy SHA-256
  `2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467`;
- retail tasks SHA-256
  `8e03ebce7901bd6218e7a7dc3105faa9324091a68058f7fe61c65262868812e8`.

Every raw run directory listed by `matrix.json` contains `manifest.json` with
the seed, algorithm, model, config hash, generated-stream dataset hash, Git
commit/dirty flag, and the exact run path. `matrix.json` additionally records
the seed list, pair counts, and all bootstrap inputs; it is the authoritative
provenance index for the table below.

## Results

Values are mean across the five seeds. Recovery is mean recovery steps on
eligible changed contexts. `N/A` in a slice means that no eligible example was
present for that seed; it is not zero.

| Method | Score | Changed success | Old-rule leakage | Invariant retention | Future-change success | Premature update | Recovery steps | Requests | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static | 0.5875 | 0.5625 | 0.4375 | 0.6402 | 0.1667 | 0.8333 | 5.80 | 16.0 | 9,576.8 |
| Reflexion-style | 0.7000 | 0.8750 | 0.1250 | 0.6962 | 0.5833 | 0.4167 | 5.50 | 20.8 | 19,805.4 |
| Replay-only | 0.6375 | 0.5000 | 0.5000 | 0.6965 | 0.3056 | 0.6944 | 5.75 | 69.4 | 51,395.2 |
| Full EvoShift | 0.5875 | 0.5000 | 0.5000 | 0.6412 | 0.3056 | 0.6944 | 6.20 | 20.4 | 13,566.6 |

The primary Full-vs-baseline paired 95% hierarchical intervals (seed clusters,
then paired samples) are:

| Baseline | Score delta | Changed delta | Invariant delta | Old-leakage delta | Premature delta | Request delta | Token delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reflexion-style | `-0.1125 [-0.2875, 0.0250]` | `-0.3750 [-0.8125, 0.0625]` | `-0.0550 [-0.2578, 0.1367]` | `+0.3750 [-0.0625, 0.8125]` | `+0.2778 [0.0000, 0.6389]` | `-0.4 [-2.8, 2.6]` | `-6238.8 [-8962.0, -3283.4]` |
| Replay-only | `-0.0500 [-0.1625, 0.0500]` | `0.0000 [-0.2500, 0.2500]` | `-0.0554 [-0.1734, 0.0489]` | `0.0000 [-0.2500, 0.2500]` | `0.0000 [0.0000, 0.0000]` | `-49.0 [-76.4, -20.2]` | `-37828.6 [-58565.8, -15639.6]` |
| Static | `0.0000 [-0.1000, 0.0875]` | `-0.0625 [-0.2500, 0.0000]` | `+0.0010 [-0.0533, 0.0571]` | `+0.0625 [0.0000, 0.2500]` | `-0.1389 [-0.5556, 0.2500]` | `+4.4 [2.0, 7.8]` | `+3989.8 [2154.6, 6360.6]` |

The live result does not support adopting Full EvoShift as a capability
winner. Reflexion is stronger on this short stream; Replay-only spends about
3.4x the requests of Full without improving its score; Full ties Static on
score while improving request/token cost relative to Replay-only. The result
is retained as a failure case rather than tuning the method to this test.

## Mechanism trace and decision

The failure trace is low-frequency context cold start: unseen contexts begin
at trust `0.40`, below learning gates. The first post-shift contradiction has
trust `0.10` and remains pending until a second consistent observation. This
protects against isolated feedback noise but delays real changes, producing
slower recovery and missed updates in the short stream. Replay-only shows the
opposite failure: validated candidates consume 69.4 requests and 51,395 tokens
per seed without a capability gain.

Adopt the four-method benchmark harness, provenance format, paired bootstrap,
and failure trace. Do not adopt a Full-vs-Reflexion capability claim. The next
research candidate is a pre-registered change-point/confidence-sequence gate
for cold-start confirmation; it must be evaluated on a new branch with the
same safety and resource gates.
