# Result: five-seed common-response critic-paraphrase confirmation

Status: completed. The common-response fixture shows a reproducible causal
capability gain from the hierarchical evidence path, with no protected-slice
regression or harmful exposure. It is not a public-benchmark or independent-model
sampling result.

## Provenance

- Pre-registration commit: `4cc811e`.
- Sweep: `runs/sweeps/20260811T034059Z`.
- Seeds: `11, 22, 33, 44, 55`.
- Model: `gpt-5.6` through the configured Responses gateway.
- Dataset hash:
  `4ad6d52c7e6b19c65d65210a4d7c4e737ceb7bc3bb51bdbd23bc22f94112c18f`.
- Shared cache:
  `data/llm_cache_critic_paraphrase_common_response.sqlite3`, created empty for
  this run.

All ten runs used the same clean commit, dataset, model, cache path, and paired
seed. Within each pair, only
`evolution.shadow_hierarchical_eprocess_enabled` differed.

## Results

Every seed produced the same controlled result:

| Metric | Exact | Hierarchical | Delta |
|---|---:|---:|---:|
| Overall score | 0.6842 | 0.7368 | +0.0526 |
| Policy-change score | 0.3750 | 0.5000 | +0.1250 |
| Protected-anchor score | 0.9091 | 0.9091 | 0.0000 |
| Cluster crossings per seed | N/A | 1 | +1 |
| Shadow-only replays per seed | 0 | 1 | +1 |
| Shadow probations per seed | 0 | 1 | +1 |
| Shadow activations per seed | 0 | 0 | 0 |
| Harmful active-memory exposure | 0 | 0 | 0 |

Across 95 paired foreground episodes, 75 outputs and scores were identical.
Each pair matched exactly through episode index 14; the first divergence was
index 15, after the hierarchical cluster crossing, replay, and probation. The
probationary memory improved one of the final four policy-change episodes, so
the controlled gain is one correct episode out of 19 (`+5.26` percentage
points overall and `+12.5` points on the eight change cases).

The hierarchical path crossed, replayed, and entered probation in all five
runs. It never exposed a harmful active memory. Future audit expired at stream
end before reaching two observations, so the memory was not activated and no
long-lived state remained.

## Interpretation

This resolves the independent-call confound in the preceding live mini: the
first 15 foreground outputs are byte-equivalent parsed responses, and the score
gain appears only after the algorithm changes its learner-visible memory state.
The result therefore supports a causal mechanism claim on this fixture.

The five seeds are not five independent model samples. They intentionally reuse
the same content-addressed responses on the same fixture and produce identical
deltas. A seed-level confidence interval would be degenerate and must not be
presented as population-level evidence. Request/token comparisons are also
invalid because cache hits replace external calls.

The next required gate is a five-seed public-source Tau3 policy-drift stream,
where seeds select different cases while each exact/hierarchical pair shares
responses for identical requests.
