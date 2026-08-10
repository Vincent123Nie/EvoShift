# Result: real-model critic-paraphrase mini (initial protocol)

Status: **aborted by the pre-registered resource stop rule**. This artifact is
not a capability result and must not be compared with the deterministic fixture
or treated as evidence for hierarchical e-process uplift.

## Provenance

- Protocol commit: `af371d3`.
- Config: `configs/experiments/critic_paraphrase_live_mini.yaml`.
- Sweep: `configs/sweeps/critic_paraphrase_live_mini.yaml`.
- Gateway model: `gpt-5.6` through the configured OpenAI-compatible `/responses`
  endpoint.
- Cache: disabled.

## What happened

The one-request provider smoke passed with a typed `OK` response. The first
(`exact_shadow_eprocess`) run then reached the configured token ceiling during
the 19-sample fixture. The budget ledger stopped the run at the first attempted
reservation whose estimate would have taken total usage to `182,022 > 180,000`.
The sweep therefore never produced a completed exact/hierarchical pair.

The partial run directory is retained under `runs/` for forensic inspection but
has no `costs.json`, `metrics.json`, or valid comparison matrix. Its partial
foreground stream contains 14 predictions and four typed critic records. No
credential material was written to the run; the gateway key remains an
environment-only input.

## Interpretation

The failure is an engineering sizing result: `gpt-5.6` reasoning usage makes the
initial 180k-token bound too small for this fixture's solver, critic, and replay
calls. It says nothing about exact versus hierarchical adaptation. The original
stop rule was honored; no threshold or algorithm parameter was changed after a
score inspection.

A separate amended protocol is required before another paid run.
