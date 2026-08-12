# Result: Tau3 hierarchical common-response confirmation

Status: completed negative result. The public-source Tau3 stream preserved
paired capability and safety exactly, but the hierarchical evidence process
never crossed, replayed, or changed an answer. The candidate fails the
pre-registered capability and mechanism gates on this stream.

## Provenance

- Pre-registration commit: `fe17326`.
- Execution commit: `e7b697f` (clean).
- Sweep: `runs/sweeps/20260812T020823Z`.
- Seeds: `11, 22, 33, 44, 55`.
- Model: `gpt-5.6` through the configured Responses gateway.
- Source: pinned Tau3 retail revision `v1.0.1`, schedule
  `v1 -> v2 -> v3 -> v2`, 96 episodes per run.
- Shared cache:
  `data/llm_cache_tau3_hierarchical_common_response.sqlite3`.

All five Exact/Hierarchical pairs used aligned samples, matching non-empty
within-pair dataset hashes, the same clean commit/model/cache, and configurations
that differed only in `evolution.shadow_hierarchical_eprocess_enabled`. Different
seeds intentionally have different dataset hashes because each seed selects a
different balanced task stream.

The first post-run audit incorrectly required one dataset hash across all seeds.
The corrected audit requires a matching non-empty hash within each paired seed,
which is the condition stated by the seeded-stream design. This correction does
not change any capability, safety, or mechanism value.

## Results

| Metric | Exact | Hierarchical | Delta | Paired 95% interval |
|---|---:|---:|---:|---:|
| Overall score | 0.6146 | 0.6146 | 0.0000 | [0.0000, 0.0000] |
| Changed-case success | 0.4893 | 0.4893 | 0.0000 | [0.0000, 0.0000] |
| Invariant retention | 0.8090 | 0.8090 | 0.0000 | [0.0000, 0.0000] |
| Old-rule leakage | 0.5107 | 0.5107 | 0.0000 | [0.0000, 0.0000] |
| Premature update | 0.8379 | 0.8379 | 0.0000 | [0.0000, 0.0000] |
| Harmful active-memory exposure | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| False retirement | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |

All 480 paired foreground outputs and scores were identical. Every hierarchical
foreground request was served from the shared content-addressed cache. Request
and token differences are therefore cache effects and are not resource claims.

## Mechanism diagnosis

The hierarchical path was active, not disconnected:

- 170 shadow failure extractions;
- 2,817 exact-signature e-process opportunities;
- 873 hierarchical cluster opportunities across 30 seed-local clusters;
- 0 exact or cluster crossings;
- 0 shadow-only replays, probations, activations, or harmful exposures.

With the frozen `q0=0.25`, `q1=0.75`, and `alpha=0.05`, a match multiplies the
e-value by 3 and a mismatch divides it by 3; crossing requires `E >= 20`.
Reconstructing the binary traces from opportunity counts and final e-values
gives 140 matches in 873 opportunities, a 16.0% match rate. Seed-local cluster
rates ranged from 10.3% to 28.6%, with a 15.2% median. The largest final cluster
e-value was only `1.88e-6`.

The public Tau3 stream therefore violates the fixed recurrence model in the
opposite direction: observations assigned to a broad context cluster are mostly
other signals, so negative evidence overwhelms recurring same-signal evidence.
Hierarchical clustering removes critic-wording fragmentation, but the current
Bernoulli likelihood treats every different signal in that cluster as evidence
against the target hypothesis. No downstream adaptation can occur because the
admission gate never opens.

## Decision

Reject the candidate as a public Tau3 capability improvement. It is
non-interfering and safety-preserving on this matrix, but it provides no measured
adaptation benefit and fails the required non-zero crossing/replay gate.

Do not tune `q0`, `q1`, `alpha`, the cluster key, or replay thresholds on this
confirmation matrix. The next independent candidate should replace the simple
per-cluster Bernoulli recurrence assumption with a pre-registered competing-risk
or per-signal conditional process that does not count unrelated signals in the
same context as direct negative evidence. It requires a new development stream
and a fresh held-out confirmation, not reuse of this matrix as an adoption test.
