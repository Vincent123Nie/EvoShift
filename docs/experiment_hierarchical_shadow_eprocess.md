# Pre-registration: observable-anchor hierarchical shadow e-process

## Question

Can low-trust feedback hypotheses recur across different LLM critic wordings
and reach verified replay without raising feedback trust or weakening the
existing trusted fast path?

The adopted shadow e-process hashes the complete proposed memory text. In the
five-seed live confirmation it preserved safety but produced no e-process
crossing because semantically related critic outputs had different scopes,
triggers, directives, and signatures. A rejected source-lower-bound candidate
showed that directly raising cold-start trust increased score on one seed but
did not improve changed-context success or recovery and exceeded its resource
gate. This candidate leaves trust unchanged.

## Candidate

The candidate is default-off behind
`evolution.shadow_hierarchical_eprocess_enabled=true` and requires the existing
shadow candidate lane and exact-signature e-process.

After scoring and observing learner-visible feedback, the critic path derives
an opaque cluster key from only:

```text
normalized domain
normalized feedback source
normalized feedback context
canonical learner-visible feedback signal
```

The key is a SHA-256 digest. It excludes the oracle reference, phase, policy
version, corruption/attack annotations, protected labels, valid/stale memory
tags, model-generated tags, and model-generated prose. Different signals for
the same context therefore form different clusters.

Every exact candidate keeps its own text signature and trusted/shadow clocks.
For low-trust observations, an additional cluster-level binary e-process uses
the already adopted frozen parameters:

```text
q0 = 0.25
q1 = 0.75
alpha = 0.05
E_t = E_(t-1) * (q1/q0)^x_t * ((1-q1)/(1-q0))^(1-x_t)
```

The discovery observation creates a cluster but is not counted as a match.
Each later low-trust observation is a match only for the same opaque cluster.
At `E_t >= 1/alpha`, the current exact candidate may enter the ordinary shadow
replay path. The exact candidate, not a merged text synthesis, is replayed.
After a shadow validation attempt, the cluster e-process resets and needs fresh
evidence. A trusted observation always uses the existing trusted clock and is
never delayed by cluster state.

Cluster evidence cannot update drift, memory utility, source trust, active
audit, policy, or active memory directly. Replay labels and future audit still
require the existing trust threshold. A passing replay produces only a
probationary memory; activation still requires later trusted memory-on/off
evidence.

The e-process is anytime-valid for each declared simple cluster stream. This
iteration does not claim family-wise error control over an unbounded number of
clusters or semantic truth identification.

## Protocol

Development uses the 144-episode, six-regime Tau3-derived stream with seeds
`[11, 22, 33, 44, 55]` under clean, `0.10` incidental-noise, attack-burst, and
combined conditions. Compare the adopted exact shadow e-process with the
hierarchical candidate, changing only the new flag.

Primary metrics are score, changed-context success, recovery steps, old-rule
leakage, invariant retention, premature update, poison persistence,
corrupted-feedback follow, false retirement, shadow cluster opportunities,
crossings, replay/probation/activation funnel, requests, and tokens.

A live API mini and five-seed common-response confirmation may run only after
the deterministic gates pass. The uncached mini is the resource result; a
shared content-addressed cache may be used only to isolate capability from
remote model sampling and cannot support a cost claim.

## Adoption gates

Adopt only if all conditions hold:

1. Hidden/oracle metadata mutations cannot change the cluster key, online
   predictions, trust, state, events, replay sample selection, or decisions.
2. Different learner-visible feedback signals for one context never share a
   cluster, and model-generated tags/prose cannot choose the cluster key.
3. The deterministic candidate has a non-zero cluster crossing and
   shadow-only replay/probation funnel; every activation has passing trusted
   replay and completed trusted future audit.
4. Macro changed-context success improves by at least `+0.03` with paired 95%
   interval lower bound no lower than `0`, while clean score does not regress.
5. Invariant retention, premature update, poison persistence,
   corrupted-feedback follow, false retirement, and harmful exposure do not
   worsen by more than `0.02` absolute in any condition.
6. Mean requests and tokens increase by no more than `25%` in every condition.
7. Full tests, branch coverage, Ruff, strict mypy, package builds, and all
   shipped configuration checks pass.

A gate failure is retained as a negative result. It does not authorize tuning
the cluster key or e-process threshold on the same confirmation stream.
