# Result: observable-anchor hierarchical shadow e-process

## Status

Rejected for adoption on the pre-registered deterministic gate. The code path
is retained on branch `codex/hierarchical-shadow-eprocess`; it is not a claim of
capability improvement and was not promoted to `main`.

## Run

Artifact: `runs/sweeps/20260810T040713Z`

The run used the six-regime, 144-episode deterministic policy-shift stream,
seeds `[11, 22, 33, 44, 55]`, and the four clean/noise/burst/combined
conditions. The only candidate change between `shadow_eprocess` and
`hierarchical_shadow_eprocess` was the new default-off flag.

Across every condition, the paired deltas were exactly zero:

| Metric | Candidate minus exact baseline |
|---|---:|
| Score | `0.0000` |
| Changed-context success | `0.0000` |
| Invariant retention | `0.0000` |
| Old-rule leakage | `0.0000` |
| Requests | `0.0000` |
| Tokens | `0.0000` |

The plumbing counters were live: cluster opportunities occurred, and shadow
replay/probation occurred. However, `shadow_cluster_eprocess_crossings` was
`0` in all 40 runs, so the non-zero-crossing adoption gate failed. No paid live
confirmation was run.

## Interpretation

The deterministic demo critic emits the same proposal for a repeated
observable anchor. In this stream, those repetitions become trusted or are
handled by the exact candidate lane before enough low-trust observations can
cross the cluster e-process. The implementation therefore has not yet shown
the intended advantage: semantically equivalent but textually different
critic proposals reaching shadow replay.

This is an informative negative result, not evidence to tune the cluster key,
threshold, or trust model after looking at the confirmation stream. The next
valid test is a pre-registered common-response live mini or a deterministic
critic-paraphrase fixture that holds the observable anchor fixed while varying
only critic wording. Adoption still requires a non-zero cluster crossing,
shadow-only replay/probation, no safety regression, and the original paired
confidence/resource gates.
