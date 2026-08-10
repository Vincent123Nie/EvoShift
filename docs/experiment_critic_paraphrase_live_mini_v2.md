# Amendment: real-model critic-paraphrase mini v2

Status: pre-registered after the initial protocol stopped at its explicit
resource bound. This amendment is a separate diagnostic and does not rewrite
the initial result.

The only amendment is the resource ceiling: `max_requests=240` and
`max_total_tokens=360000`, sized from the observed `182,022` reservation at the
initial stop. Samples, seed, model, cache policy, trust/e-process parameters,
replay, future audit, and exact versus hierarchical flag comparison are
unchanged. The amended config is
`configs/experiments/critic_paraphrase_live_mini_v2.yaml` and the sweep is
`configs/sweeps/critic_paraphrase_live_mini_v2.yaml`.

The same stop rules apply: abort on the amended budget, repeated non-retryable
provider errors, or invalid typed critic output. Report v2 only as a one-seed
diagnostic; use `scripts/audit_critic_paraphrase_live.py` for the machine-readable
protocol and safety audit.
