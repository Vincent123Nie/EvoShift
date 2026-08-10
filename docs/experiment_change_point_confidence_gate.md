# Pre-registration: source-aware change-point posterior gate

## Question

Can a calibrated, source-aware change-point gate reduce cold-start recovery
delay on persistent policy changes without admitting isolated feedback noise?

The current dynamic trust model assigns every unseen `(source, context)` a
fixed `0.40` trust and requires repeated evidence before the first context
label is committed. The public PolicyShift trace showed that this delayed
learning for legitimate low-frequency contexts. This branch tests one
explicit candidate; it does not change the default behavior until the gates
below pass.

## Candidate

The candidate is enabled only by
`evolution.dynamic_feedback_posterior_gate_enabled=true` and keeps the
predict-before-feedback firewall.

For a source Beta posterior `Beta(alpha_s, beta_s)`, use a conservative normal
lower bound for a cold-start decision:

```text
mu_s = alpha_s / (alpha_s + beta_s)
var_s = alpha_s * beta_s /
        ((alpha_s + beta_s)^2 * (alpha_s + beta_s + 1))
lcb_s = max(0, mu_s - z * sqrt(var_s))
```

The frozen candidate uses `z=1.2815515655` (a one-sided 90% bound). For an
unseen context, the post-score trust may use `max(cold_start_trust, lcb_s)`;
the context label is still not committed until normal evidence confirmation.
Unknown or low-prior sources therefore remain below the ordinary learning
threshold.

For a committed context and a pending alternative label, maintain a
Beta-Bernoulli change-point posterior. If the pending label repeats `n` times,

```text
odds_change = pi / (1 - pi) * (p1 / p0)^n
P_change = odds_change / (1 + odds_change)
```

where `pi=0.20`, `p0=0.20`, and `p1=0.80`. A change can be committed only when
`P_change >= 0.80` and the configured temporal span is satisfied. Returning to
the old label cancels the pending run and records transient evidence. The
posterior is an evidence gate, not an oracle or semantic truth detector.

## Baselines and protocol

The primary comparison is current Full EvoShift versus the candidate variant
on the same source-derived Tau3Retail-PolicyDrift stream, model, prompt,
budget, cache policy, ordered samples, and seeds `[11, 22, 33, 44, 55]`.
The existing Static, Reflexion-style, and Replay-only rows remain the external
capability context. Deterministic demo streams are used only for mechanism and
oracle-firewall tests.

Primary metrics are changed-context success and mean recovery steps. Safety
metrics are old-rule leakage, invariant retention, premature update,
corrupted-feedback follow rate, future-audit false rollback, and active-memory
false retirement. Resource metrics are requests, foreground tokens, total
tokens, and reported cost.

## Adoption gates

The candidate is adopted only if all gates pass:

1. On deterministic clean/noisy/attack fixtures, no oracle or hidden metadata
   changes online predictions, trust, state, events, or candidate decisions.
2. On the fixed safety fixtures, corrupted/attack follow rate and false
   retirement do not increase, invariant retention does not fall by more than
   `0.05`, and the candidate does not bypass paired replay or future audit.
3. On the clean public-derived stream, changed-context success improves by at
   least `0.10` or mean recovery steps decreases by at least `1.0`, with a
   non-negative paired 95% interval for the improved primary metric.
4. The candidate does not increase mean requests or total tokens by more than
   `25%` versus current Full EvoShift.
5. Full tests, branch coverage, Ruff, strict mypy, source/wheel build, and all
   configuration-loading checks pass.

Failure is retained as a documented result and the candidate stays disabled.
No public SOTA or production claim follows from passing this mechanism gate.

## Live mini result

Artifact: `runs/sweeps/20260810T033713Z`. The uncached mini ran current Full
and the frozen candidate on seed `11`, the same 16-example `v1 -> v2` stream,
and remote model `gpt-5.6`. Both manifests are clean at commit
`5f0d70dc0239fb5c0e83b0b81b1bc55a3abf5434`, share dataset hash
`a08569648e3b45ba552ca3ab17d09b318fbca55b4c2ee21653c660026c3332c4`,
and use distinct recorded config hashes. No API secret appears in the
artifacts.

| Variant | Score | Changed | Invariant | Future change | Premature | Recovery | Requests | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current Full | 0.5625 | 0.5000 | 0.7500 | 0.2500 | 0.7500 | 5.0 | 32 | 214,808 |
| Change-point posterior | 0.6875 | 0.5000 | 0.8750 | 0.5000 | 0.5000 | 5.0 | 50 | 324,179 |

The candidate reduced clean-feedback quarantine from `0.5625` to `0.1250`
and produced the intended trace: a configured `0.90` source gave unseen
contexts a lower-bound trust between roughly `0.67` and `0.79`, while the
first conflicting label remained quarantined at trust `0.10` with change
posterior `0.50`. Two repeated conflicts crossed the frozen posterior
threshold and were recorded separately from ordinary confirmations.

The primary gate did not pass. Changed-context success and mean recovery were
unchanged. The `+0.1250` score point difference is not a causal memory claim:
sequential uncached calls produced different answers on byte-identical
no-memory prompts before the variants' states diverged. The one-seed interval
is not repeated-seed evidence.

The resource gate also failed. The candidate staged two replay-passing
probationary memories instead of one, but both expired without realized future
audit. The extra eight-example replay increased external requests by `56.25%`
and total tokens by `50.91%`, above the frozen `25%` limits.

## Decision

Reject this candidate and do not run the ten-run confirmatory matrix. The
implementation remains default-off on its experiment branch for auditability;
it is not adopted into `main`. Raising cold-start trust also preserves the
known identifiability problem: a same-source attacker can make the first label
of a previously unseen context observationally identical to a legitimate new
rule.

The next candidate should leave trust unchanged and aggregate low-trust shadow
hypotheses with a hierarchical e-process over a predeclared learner-visible
cluster key. That targets live critic signature fragmentation without allowing
one observation to update drift, utility, or active memory.
