# Pre-registration: five-seed common-response confirmation

Status: pre-registered before this confirmation run. This protocol isolates
capability effects from independent API sampling; it does not make resource or
latency claims because cache hits intentionally replace external calls.

## Frozen comparison

- Fixture: `examples/critic_paraphrase_fixture.jsonl`, all 19 samples in fixed order.
- Variants: `exact_shadow_eprocess` and `hierarchical_shadow_eprocess`.
- Seeds: `11, 22, 33, 44, 55`.
- Model: `gpt-5.6` through the configured OpenAI-compatible Responses gateway.
- A single fresh content-addressed cache is shared by all ten runs at
  `data/llm_cache_critic_paraphrase_common_response.sqlite3`.
- Only `evolution.shadow_hierarchical_eprocess_enabled` differs between variants;
  cache settings are identical and explicit in the sweep grid.
- The cache is deleted before the first run. No API key is stored in the cache,
  run artifacts, or Git.

The sweep order runs all exact seeds before all hierarchical seeds. Identical
logical requests therefore reuse the first sampled real-model response. Once an
algorithm changes its learner-visible memory state, a different request is
allowed and is reported as an algorithmic divergence rather than silently
treated as a common response.

## Primary audit fields

The audit must report, per seed and pooled:

- completed exact/hierarchical pairs and matching dataset/model/Git provenance;
- sample-id alignment and exact output/score matches per foreground episode;
- cache-hit counts for each variant and common-response coverage;
- overall/protected/change scores and paired score deltas;
- exact/cluster e-process opportunities and crossings, shadow replay,
  probation/activation/rejection, future-audit safety, and harmful exposure.

This is a capability-control experiment. Do not quote token/request differences
as efficiency, because the shared cache intentionally removes repeated external
calls. Do not tune thresholds after inspecting the matrix.

## Interpretation gates

The protocol is valid only if all ten runs complete with the same clean commit,
dataset, model, and seed assignments, cache is enabled with the expected path,
and the two variants differ only in the hierarchical flag. A high common-output
match rate with zero score delta is a valid non-interference result. A crossing
without safe replay admission is a mechanism reachability result, not capability
uplift. Any output divergence must be listed with its first episode and selected
memory state.

Run:

```bash
rm -f data/llm_cache_critic_paraphrase_common_response.sqlite3
evoshift sweep --spec configs/sweeps/critic_paraphrase_common_response.yaml
python scripts/audit_critic_paraphrase_common_response.py runs/sweeps/<sweep-id>
```
