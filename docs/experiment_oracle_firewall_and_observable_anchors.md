# Oracle firewall and provenance-context replay anchors

## Why this iteration was necessary

The first 16-run context-bound revival screen at
`runs/sweeps/20260805T163314Z` passed its mechanism gates and produced a
four-condition candidate mean score of `0.9236`. A later decision-path audit
showed that the inherited replay verifier and future counterfactual auditor
read the benchmark-only per-example `metadata["protected"]` flag online. That
flag identifies invariant cases after benchmark construction; it is not
learner-visible deployment evidence.

The old sweep therefore remains useful for debugging the revival transaction,
but it is not adoption evidence. The correction had to satisfy two goals at
once:

1. remove the oracle marker from every promotion and rollback path;
2. retain an observable compatibility test so real policy changes do not
   erase unrelated historical capabilities.

## First oracle-free attempt and negative result

The first replacement reserved one third of replay for the latest trusted
example from each historical `(feedback_source, feedback_context)` pair and
backfilled any unused replay slots from recent trusted history. It did not use
extra model calls or hidden labels.

That implementation was safe from `metadata["protected"]`, but the existing
PolicyShift integration cases fell from the former `0.9722` score to:

- clean: `0.8750`;
- 10% noise plus 10% attack: `0.8333`.

The failure was informative. A candidate learned from
`refund:any:days_8_14` was replayed against old-regime examples from the same
context. Those examples were not stability tests: they encoded the exact rule
being superseded. Repeated same-context backfill produced artificial
regressions, delayed promotion for several candidate versions, and reduced
changed-case adaptation while invariant retention remained high.

## Adopted observable rule

Historical replay now distinguishes a candidate's change context from
unrelated compatibility contexts:

- derive the candidate context only from its provenance episode IDs;
- map provenance episodes to the typed learner-visible
  `(feedback_source, feedback_context)` key;
- exclude matching pre-boundary contexts from historical anchors and backfill;
- keep current-regime examples from that context, so the candidate must still
  demonstrate positive present-regime utility;
- reserve the fixed historical quota for other observable contexts;
- treat anchors as ordinary global replay examples, never as protected cases;
- persist replay sample IDs, episode indices, and the configured protected mask
  beside every replay decision for later audit.

This is not an invariant detector. It encodes a narrower causal claim: old
labels from the candidate's observed change context are unsuitable evidence
against that change, while unrelated observable contexts remain reasonable
non-interference probes.

The quota uses exact `floor(window * fraction)` semantics. Configurations that
enable historical anchors but round to zero slots are rejected rather than
silently exceeding the registered fraction.

After provenance-context exclusion, both previously failing PolicyShift
integration cases returned to `0.9722`, with `0.8571` changed-case success,
`0.1429` old-rule leakage, and `1.0` invariant retention.

## Oracle firewall

The integration firewall runs the deterministic hard PolicyShift stream four
times while changing only `metadata["protected"]`:

- original labels;
- all false;
- complement;
- deterministic shuffle.

For every perturbation it requires exact equality of:

- ordered promotion and rollback decisions;
- replay sample IDs and episode indices, equivalent to replay Jaccard `1.0`;
- replay protected masks, which remain all false when no explicit protected
  phase contract is configured;
- the complete online evolution-event sequence after replacing run-local UUIDs
  and timestamps with stable semantic identities;
- canonical SQLite memory and policy state;
- all metrics except the dataset/config hashes and post-hoc invariant slice
  annotations;
- the new cross-run semantic final-state hash.

The scenario is non-trivial: it must contain at least one promoted memory and
at least one future-audit rollback. The current firewall passes all four
perturbations, so its measured decision flip rate is `0.0` and online event,
replay, and semantic state matches are all `1.0`.

## Additional decision-input hardening

Two adjacent paths were hardened before the larger matrices:

- solver requests no longer contain `sample_id`; PolicyShift and other
  benchmark IDs encode phase or position and must not reach a real model;
- `dynamic_feedback_context_field` is restricted at configuration and runtime
  to the typed learner-visible `feedback_context` field. It can no longer be
  pointed at `protected`, policy version, phase index, valid/stale tags,
  corruption labels, or similar benchmark metadata.

Explicit `benchmark.protected_phases` remains a separate deployment contract.
It must represent a known canary group, not a hidden benchmark phase. The
per-example benchmark marker remains available only for post-hoc metrics.

## Preliminary oracle-safe targeted result

The dirty-worktree verification sweep at
`runs/sweeps/20260806T020049Z` completed all 16 assignments. It is a pre-commit
check, not the final archived result.

Candidate scores were:

| Seed | Attack burst | Candidate score | Invariant retention |
|---:|---:|---:|---:|
| 233 | 0 | 0.9097 | 1.0000 |
| 255 | 0 | 0.9375 | 1.0000 |
| 233 | 2 | 0.8819 | 1.0000 |
| 255 | 2 | 0.9375 | 1.0000 |

The candidate four-condition mean was `0.9167`, above both
`semantic_revival` and `transactional_retirement` at about `0.9115`. The known
seed-233 same-context revival was preserved; seed-255 wrong-context versions
were excluded before a revival model intervention; context-mismatch
confirmations, post-confirmation tag-associated harm, and unconfirmed
persistent transitions were all zero. Context-record coverage was `1.0`.

Retirement registration precision remained only `0.667`, `0.5`, `1.0`, and
`0.5` across the four candidate runs. Sequential confirmation prevented false
persistent retirement, but these numbers do not support a claim that the
retirement detector itself is accurate. Provisional exposure counts and
failure episodes also use different units and must not be divided into a
causal rate.

## Remaining adoption evidence

Before merge, the implementation still requires clean-commit evidence from:

1. the same 16-run targeted matrix;
2. the four-way oracle firewall;
3. a 400-run development matrix on the inspected seeds
   `[122, ..., 333]` and five variants;
4. a 320-run confirmation matrix on the unused seeds
   `[344, ..., 555]` and four frozen variants;
5. full Ruff, formatting, strict Mypy, package build, data-manifest audit, and
   branch-aware test coverage gates.

Passing the mechanism screen is not equivalent to SOTA. The defensible claim
for this iteration is narrower: EvoShift can preserve context-bound recurrence
and measurable safety under policy drift without using the hidden protected
label in online replay or future-audit decisions. Real-model and public-data
results remain separate evidence requirements.
