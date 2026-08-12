# Result: Tau3 conditional shadow e-process

Status: development and held-out confirmation completed. The conditional
evidence change passes the pre-registered mechanism and safety/non-inferiority
gates, but it does not improve foreground score on this 96-episode stream.

## Provenance

- Implementation and pre-registration commit: `44bb020`.
- Development sweep: `runs/sweeps/20260812T031846Z`, seeds `[66, 77]`.
- Held-out confirmation sweep: `runs/sweeps/20260812T040344Z`, seeds
  `[88, 99, 111]`.
- Model: `gpt-5.6` through the configured Responses gateway.
- Each stage used a fresh, dedicated content-addressed cache. Cache-derived
  request and token counts are not efficiency evidence.
- All 15 runs used the same clean commit within their stage. Every three-way
  seed comparison had aligned sample IDs, matching dataset hashes and resolved
  configurations that differed only in the hierarchical and conditional mode
  flags.

## Results

The held-out aggregate result is:

| Metric | Exact | Hierarchical | Conditional |
|---|---:|---:|---:|
| Overall score | 0.6632 | 0.6632 | 0.6632 |
| Changed-case success | 0.5749 | 0.5749 | 0.5749 |
| Cluster opportunities per seed | N/A | 137.7 | 21.3 |
| Unrelated updates skipped per seed | N/A | 0.0 | 112.3 |
| Cluster crossings per seed | N/A | 0.0 | 3.7 |
| Shadow-only replays per seed | 0.0 | 0.0 | 1.0 |
| Probations / activations | 0 / 0 | 0 / 0 | 0 / 0 |
| Harmful active-memory exposure | 0 | 0 | 0 |

Across 288 paired confirmation episodes, Conditional and both comparators had
identical outputs and scores. The paired score, changed-case, leakage,
invariant-retention, and premature-update deltas were exactly `0.0`, with
interval `[0.0, 0.0]`. The three confirmation seeds produced 11 conditional
cluster crossings and 3 shadow-only replay attempts; historical Hierarchical
produced no crossings. Development showed the same pattern: 8 crossings and 2
shadow-only replays, with zero foreground delta and zero harmful exposure.

## What improved

The previous public Tau3 result observed 873 historical hierarchical
opportunities, no crossing, and a 16.0% inferred recurrence rate because every
other cluster was treated as negative evidence. Conditional evidence removed
that dilution: a cluster is now updated only by the same learner-visible
`(domain, feedback_source, feedback_context)` family. A competing signal inside
the family remains negative evidence, while unrelated contexts are skipped.

This is a real mechanism improvement: the held-out run skipped 337 unrelated
cluster updates, reduced relevant opportunities to 64, and reached the replay
eligibility threshold 11 times. Three crossings scheduled shadow-only replay,
without weakening the existing replay and future-audit safety barriers.

## Why score did not improve

Evidence recurrence and memory utility are separate questions. The downstream
paired replay correctly rejected or deferred the crossed candidates:

- seed 88 replayed two shadow candidates. One had mean delta `-0.1667`, CI
  `[-0.5, 0.0]`, and regression rate `0.1667`; the other had delta `0.0`.
- seeds 99 and 111 had replay-passing memory candidates, but they arose on
  trusted paths or reached future audit too near stream end. Four probationary
  candidates expired before a future observation could confirm them.
- no candidate became active, so no learner-visible prompt changed and all
  foreground outputs remained identical.

Lowering replay gain or future-audit gates after reading these results would be
post-hoc tuning and would weaken the project's safety story. The defensible
claim is therefore narrower than score uplift: Conditional fixes cross-context
evidence dilution and safely makes the intended mechanism reachable on unseen
public-source streams; this stream does not establish capability improvement.

## Practical conclusion

This is an appropriate stopping point for the project scope. The system now has
a tested conditional sequential-evidence mechanism, real-model public-source
validation, negative-result analysis, and resumable experiment infrastructure.
A future score-oriented experiment would need a longer stream or a protocol
that places post-replay observations after crossing, pre-registered with new
seeds; it should not reuse these matrices for tuning.
