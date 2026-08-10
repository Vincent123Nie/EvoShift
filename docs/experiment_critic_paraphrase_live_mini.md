# Pre-registration: real-model critic-paraphrase mini

Status: pre-registered before the first live run. This is a bounded diagnostic,
not a SOTA benchmark or a claim about the general capability of `gpt-5.6`.

## Question

When the same policy feedback is expressed by a real model with wording
variation, does the learner-visible hierarchical shadow e-process accumulate
evidence more reliably than exact candidate identity alone, without unsafe
promotion or protected-slice regression?

## Frozen protocol

- Fixture: `examples/critic_paraphrase_fixture.jsonl`, all 19 samples, fixed order.
- Provider: OpenAI-compatible `/responses` gateway, model `gpt-5.6`.
- Variants: `exact_shadow_eprocess` and `hierarchical_shadow_eprocess`.
- Seed: 42; cache disabled; one run per variant.
- The only algorithmic difference is
  `evolution.shadow_hierarchical_eprocess_enabled`.
- Trust, e-process (`p0=0.25`, `p1=0.75`, `alpha=0.05`), replay, probation,
  future-audit, solver, and memory retrieval settings are fixed.
- Maximum budget: 160 external requests and 180,000 total tokens per run.
- Credentials are read only from `LOCAL_OPENAI_GATEWAY_API_KEY`; they are never
  written to config, logs, manifests, or Git.

The live critic is instructed to return the existing typed JSON schema. The
protocol records its observed signature/trigger/directive diversity; it does not
pretend that a model's semantic equivalence is established by hidden labels.
The deterministic fixture remains the causal mechanism gate.

## Reporting and stop rules

Run:

```bash
EVOSHIFT_OPENAI_BASE_URL=https://easyai123.shop/v1 \
  evoshift sweep --spec configs/sweeps/critic_paraphrase_live_mini.yaml
python scripts/audit_critic_paraphrase_live.py runs/sweeps/<sweep-id>
```

Stop the mini if either run reaches its request/token budget, the provider
returns repeated non-retryable errors, or the gateway cannot produce valid
typed critic JSON. Report failures and missing diversity as coverage limits,
not as evidence for changing thresholds.

Primary report fields are score, policy-change score, observed critic diversity,
exact/cluster e-process opportunities and crossings, shadow-only replay,
probation/activation, harmful promotion/exposure, requests, tokens, and errors.
Paired bootstrap intervals are descriptive only because this mini has one seed.

## Interpretation

A cluster crossing with safe downstream audit is evidence that the mechanism can
be reached by a live model on this fixture. Zero crossings or low wording
diversity are valid negative/coverage results. Neither outcome supports a SOTA
claim without repeated seeds and a public policy-drift benchmark.
