# Critic-paraphrase hierarchical e-process fixture

Status: preregistered before implementation and execution.

## Question

Does the learner-visible hierarchical shadow e-process accumulate repeated semantic
feedback when a deterministic critic expresses the same policy lesson with different
wording, while the exact-text shadow e-process remains fragmented?

This is a mechanism fixture, not an end-to-end SOTA result. It isolates the causal
claim behind hierarchical evidence sharing before any paid-model confirmation.

## Frozen comparison

The comparison has two variants:

- `exact_shadow_eprocess`: exact candidate identity only.
- `hierarchical_shadow_eprocess`: exact candidate identity plus the existing opaque
  cluster digest computed from learner-visible domain, feedback source, feedback
  context, and feedback signal.

The following inputs must be identical between variants:

- benchmark samples, hidden oracle, stream order, and seed;
- learner-visible feedback source, context, reference, and signal;
- trust configuration and trust-state transitions;
- solver and memory retrieval behavior;
- replay, probation, future-audit, and safety configuration;
- exact e-process null, alternative, and alpha parameters;
- cluster-key function;
- deterministic critic paraphrase assignment.

The only algorithmic difference is
`evolution.shadow_hierarchical_eprocess_enabled`. The provider fixture may vary critic
wording, but it must do so from a stable hash of learner-visible task content rather
than call order, algorithm state, hidden labels, or run variant.

## Fixture contract

The demo provider receives a default-off critic-paraphrase mode. In fixture mode it
emits semantically equivalent refund-policy memories whose trigger, directive,
anti-pattern, and evidence wording vary deterministically. Tags, policy meaning,
feedback metadata, solver behavior, and oracle behavior remain unchanged.

At least five paraphrase identities must occur inside one observable evidence cluster.
With preregistered e-process parameters `p0=0.25`, `p1=0.75`, and `alpha=0.05`, five
consecutive cluster matches are sufficient to cross the threshold because
`(p1 / p0)^4 = 81 > 20`; the discovery observation is excluded by design.

No production trust, admission, replay, or safety rule may be weakened solely to make
the fixture pass. The stream may use an explicit deterministic trust transition: first
collect low-trust shadow evidence, then expose later trusted observations required by
the unchanged future-audit gate.

## Primary gates

The deterministic fixture passes only if all of the following hold:

1. The exact variant has zero exact e-process crossings for the paraphrased lesson.
2. The hierarchical variant has at least one cluster e-process crossing.
3. The hierarchical crossing authorizes a shadow-only replay attempt.
4. At least one shadow candidate enters probation through that path.
5. No shadow memory is promoted without completing the existing replay and future-audit
   gates.
6. There is no increase in protected-slice regression or harmful-memory exposure.
7. Candidate and baseline consume the same benchmark samples and hidden oracle.
8. The hierarchical variant's total requests and tokens remain within the configured
   resource gate relative to the exact variant.
9. Oracle-firewall tests show that hidden policy labels cannot affect paraphrase choice
   or cluster identity.

## Secondary observations

Report exact and cluster e-process opportunities/crossings, shadow-only replay
attempts, probation outcomes, future-audit outcomes, score, protected-slice metrics,
harmful exposure, requests, and tokens. Capability score need not improve in this
mechanism fixture; a crossing without safe downstream action is a failed fixture.

## Stop rules

- Do not change trust thresholds, e-process probabilities, alpha, cluster-key fields,
  or paraphrase assignment after inspecting experiment outcomes.
- A negative deterministic result is recorded as negative; it is not repaired by
  post-hoc threshold tuning.
- Do not run a paid OpenAI API mini until this deterministic fixture passes every
  primary gate.
- Any paid mini must be separately preregistered with a bounded request/token budget.

