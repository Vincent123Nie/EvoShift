# Limitations and non-goals

This document is intentionally explicit. A strong interview answer should state
which risks the system controls, which it merely measures, and which remain
open research problems.

## Research validity

### Replay is not an independent test

Memory and policy candidates are generated from observed stream failures and
validated on a recent/protected replay buffer. Paired replay reduces variance
and catches immediate regressions, but repeated testing can overfit that buffer.

The online probation audit now closes part of this gap by pairing later relevant
interactions with and without each replay-passing card. The separate
`evoshift audit` evaluates the entire evolved state on a distinct,
mutation-free stream. The system still does **not**:

- reserve a disjoint promotion buffer during online learning;
- guarantee that every probation receives enough later evidence before stream
  end;
- run per-card counterfactuals on a fully untouched public held-out dataset;
- compute anytime-valid confidence under repeated sequential looks.

Within-stream candidate attribution and whole-state held-out transfer are
measurable. Candidate-level **held-out** causal generalization remains future
work.

### Statistical power and dependence

The promotion gate uses a percentile bootstrap over paired examples. Small
windows produce coarse intervals, stream examples may be autocorrelated, and
repeated candidate testing creates a multiple-comparison problem. The current
comparison command does not implement clustered/block bootstrap, sequential
testing, Holm correction, or exact McNemar analysis.

### Baseline scope

Static, Self-Refine, and Reflexion are controlled local modes, not official
project reproductions. They are useful for isolating EvoShift's verification
mechanism under one solver/provider implementation, but external paper numbers
are contextual references only.

### Provider nondeterminism

An API model alias can change without notice. Even with identical prompts,
provider-side sampling, batching, safety systems, or infrastructure may alter
outputs. Idempotency and caching reduce duplicate work; they do not make a
remote model an immutable experimental object.

## Algorithmic limitations

### Lexical retrieval

BM25 and token-Jaccard MMR are transparent and CPU-only, but they miss
paraphrases, multilingual semantic matches, and deeper contradiction. Dense or
hybrid retrieval should be compared behind the same cost and promotion
protocol rather than assumed superior.

The default provenance-domain gate prevents unrelated memories from being
retrieved solely because of generic word overlap or a high utility prior. It
depends on stable, meaningful domain metadata and can suppress useful transfer
between related task families. `allow_cross_domain_transfer=true` exists for a
preregistered ablation; it should not be enabled after inspecting test results.

### Imperfect memory credit

The solver reports `applied_memory_ids`, filtered against retrieved IDs. When it
reports none, the runner credits the retrieved set. This does not identify
interactions among memories or separate base-model competence from memory
effects. Leave-one-out, influence-function, or Shapley-style estimates would be
more causal and much more expensive.

### Heuristic drift detection

Page-Hinkley plus retrieval-novelty EWMA is constant-memory and auditable, but
it can trigger on a cluster of hard examples, miss conditional/covariate shifts,
or react too late. Its thresholds are benchmark-dependent and require a
calibration split. ADWIN, CUSUM variants, learned detectors, and explicit
change-point models remain ablations.

### Narrow policy search

The slow-loop proposer is deterministic and mutates only allowlisted retrieval
and write hyperparameters. This makes behavior easy to audit but explores a
small search space. GEPA-style textual evolution, contextual bandits, Bayesian
optimization, or MemSkill-style memory operators could increase capability at
the cost of more evaluations and a larger safety surface.

### Memory poisoning and semantic safety

Schema bounds, shadow staging, replay, protected examples, and solver
instructions reduce poisoning risk. They cannot prove that a seemingly useful
directive is universally safe, free of hidden prompt injection, or robust to
semantic paraphrases. The shadow recurrence e-process is anytime-valid only
under its declared bound on exact-signature recurrence probability. Exact
signatures can fragment equivalent hypotheses, while semantic clustering would
introduce a new false-merge attack surface. It also cannot identify a
coordinated source whose repeated false feedback satisfies the alternative
recurrence model. Independent delayed evidence remains necessary for that case.
No production deployment should auto-promote untrusted memory without
domain-specific policy checks and adversarial evaluation.

### Dynamic trust can be slow or captured

The dynamic trust model treats an isolated contradiction as suspect and waits
for repeated per-context agreement. This reduces premature updates but delays
legitimate change when contexts repeat slowly. A sufficiently persistent
attacker can also become the committed label because the model measures
temporal consistency, not semantic truth or signer authenticity.

### Future audit is not sequentially calibrated

Asymmetric early rollback limits exposure after negative future utility, but
the current bootstrap gate was designed for a fixed replay window. Repeated
looks up to `future_audit_max_observations` are not backed by an anytime-valid
confidence sequence. Ordinary noise can therefore increase unnecessary
rollback, as observed in the static-trust ablation.

### Active causal audit is selective, not complete causal identification

The active auditor runs only after a trusted failure and only for applied IDs
reported by the solver. This makes cost bounded, but it does not audit silent
harm, successful-but-unnecessary memories, or interactions among several cards.
Its ledger accumulates negative observations across the memory version rather
than using an anytime-valid or context-windowed test. At 25% noise the adopted
dynamic two-evidence rule still shows up to 0.10 false retirement in the
deterministic stress sweep.

Oracle valid/stale memory tags exist only for the synthetic PolicyShift metric.
A real deployment needs domain policy provenance, executable counterfactuals,
or human review to know whether a retirement was correct. The deterministic
demo validates lifecycle semantics, not general causal identifiability.

## Dataset limitations

### BBH is a proxy for changing task distributions

Ordered BBH task families create clear domain shifts and exact scoring, but BBH
does not directly test long-term conversational memory, tool execution, user
policy compliance, or production state. Public benchmark contamination is also
possible for frontier models.

The held-out task families are unseen by the EvoShift run, not necessarily
unseen by the base model. The audit measures transfer of external state, not
zero-shot novelty of the underlying LLM.

### Memory-specific datasets need dedicated readers

The pinned LongMemEval session-retrieval adapter is implemented and validated
against the cleaned 500-entry file, including the 470/30 scored/abstention
split and official retrieval metrics. It does not yet implement the downstream
reader QA/temporal grader or LongMemEval-V2. LoCoMo still needs its own pinned
adapter and scorer. Flattening either dataset through the generic Hugging Face
adapter would erase the memory problem being evaluated. BFCL and tau-bench
likewise need structured tool calls and executable environments.

## Systems limitations

- SQLite is appropriate for a local single-writer research runtime, not a
  high-QPS multi-writer or multi-region service.
- The cache namespace includes provider kind and model but is not a tenant
  security boundary.
- Run artifacts contain benchmark prompts, references, and model outputs and
  are not encrypted at rest.
- There is no complete dependency lockfile; bounded version ranges can resolve
  differently across dates.
- Run manifests record configured model identity but not a provider-side model
  checksum or API deployment version.
- Dollar gates are meaningful only when accurate provider prices are supplied;
  otherwise token count is used as a resource proxy.
- The live workflow assumes an OpenAI-compatible `/responses` endpoint. A
  gateway may ignore idempotency, report incomplete usage, or reject optional
  fields differently.
- Generated run directories and downloaded datasets are local and Git-ignored;
  formal experiments require a separate artifact retention policy.

## Non-goals

EvoShift currently does not attempt to:

- train or fine-tune model weights;
- generate and execute arbitrary tools or source-code patches;
- provide an enterprise multi-tenant Agent platform;
- reproduce every published baseline implementation;
- claim leaderboard SOTA from a synthetic demo or smoke run;
- guarantee monotonic improvement under every distribution shift.

## Prioritized roadmap

1. Validate causal active-memory governance with a frozen public API model and
   a realistic policy/tool benchmark.
2. Replace cumulative failure-triggered evidence with context-windowed,
   sequentially valid retirement tests and bounded multi-memory attribution.
3. Run per-card counterfactual audit on an untouched public or realistic policy
   stream with a frozen API model.
4. Implement exact McNemar plus block/cluster bootstrap for repeated seeds.
5. Add the LongMemEval-V2 adapter, temporal reader scorer, and same-reader QA
   transfer evaluation; keep the current retrieval-only adapter as a separate
   track.
6. Add dense/hybrid retrieval and contradiction-aware memory admission.
7. Add BFCL structured tool-call state and official executable scoring.
8. Add tenant-aware storage/cache isolation and encrypted artifact export.

Until those items are complete, the strongest honest claim is a tested,
auditable framework for studying API-only test-time adaptation—not a proven
production service or state-of-the-art algorithm.
