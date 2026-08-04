# Limitations and non-goals

This document is intentionally explicit. A strong interview answer should state
which risks the system controls, which it merely measures, and which remain
open research problems.

## Research validity

### Replay is not an independent test

Memory and policy candidates are generated from observed stream failures and
validated on a recent/protected replay buffer. Paired replay reduces variance
and catches immediate regressions, but repeated testing can overfit that buffer.

The implemented `evoshift audit` closes one part of this gap by evaluating the
entire evolved state on a distinct, mutation-free stream. It does **not** yet:

- reserve a disjoint promotion buffer during online learning;
- run leave-one-memory-out counterfactuals for every promoted card;
- map every replay decision to a future realized gain;
- compute held-out promotion precision automatically.

Whole-state forward transfer is therefore measurable; candidate-level causal
generalization remains future work.

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
adversarial tasks. No production deployment should auto-promote untrusted
memory without domain-specific policy checks and adversarial evaluation.

## Dataset limitations

### BBH is a proxy for changing task distributions

Ordered BBH task families create clear domain shifts and exact scoring, but BBH
does not directly test long-term conversational memory, tool execution, user
policy compliance, or production state. Public benchmark contamination is also
possible for frontier models.

The held-out task families are unseen by the EvoShift run, not necessarily
unseen by the base model. The audit measures transfer of external state, not
zero-shot novelty of the underlying LLM.

### Memory-specific datasets need dedicated adapters

LongMemEval and LoCoMo contain nested histories, temporal structure, and
specialized graders. Flattening them through the generic Hugging Face adapter
would erase the memory problem being evaluated. BFCL and tau-bench likewise
need structured tool calls and executable environments.

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

1. Add a disjoint online promotion buffer and candidate-to-future-outcome audit.
2. Implement exact McNemar plus block/cluster bootstrap for repeated seeds.
3. Add a dedicated LongMemEval-V2 adapter and temporal scorers.
4. Add dense/hybrid retrieval and contradiction-aware memory admission.
5. Add BFCL structured tool-call state and official executable scoring.
6. Add tenant-aware storage/cache isolation and encrypted artifact export.

Until those items are complete, the strongest honest claim is a tested,
auditable framework for studying API-only test-time adaptation—not a proven
production service or state-of-the-art algorithm.
