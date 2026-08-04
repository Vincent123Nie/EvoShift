# Live API engineering validation

This document records an end-to-end engineering validation performed on
2026-08-04 against a private OpenAI-compatible `/responses` gateway. The
endpoint and credential are intentionally omitted. Generated runs, downloaded
BBH payloads, and the LLM cache remain Git-ignored.

These numbers are **not research results or resume improvement claims**. The
runs used one mutable model alias (`gpt-5.6`), one seed, zero configured dollar
prices, and a dirty worktree while defects were being fixed. Regenerate clean,
repeated-seed artifacts from a single commit before reporting performance.

## Scope exercised

- provider doctor and one-request live smoke;
- pinned BBH download and SHA-256 validation;
- Static, Self-Refine, Reflexion, and EvoShift execution;
- failure attribution, shadow memory, paired replay, promotion rejection and
  admission;
- Page-Hinkley-triggered bounded `PolicyPatch` replay and rejection;
- paired run comparison with cache and provenance guards;
- frozen held-out audit with state fingerprint verification;
- Ruff, strict mypy, branch-aware coverage, and package builds.

## Live observations

### Provider and cheap smoke

The provider smoke returned exactly `OK` from model alias `gpt-5.6`. On the
16-example BBH smoke stream all four methods scored `1.0`, so the stream was
saturated and useful only for protocol validation.

The login shell exposed the gateway key but not its base-URL override. A smoke
using the repository's safe official default therefore did not exercise the
private reverse proxy. Supplying both `EVOSHIFT_OPENAI_API_KEY` and
`EVOSHIFT_OPENAI_BASE_URL` at runtime made the compatible gateway path pass;
neither value is stored in run artifacts or Git.

With response caching disabled, the external request counts were:

| Method | Foreground examples | External requests |
|---|---:|---:|
| Static | 16 | 16 |
| Self-Refine | 16 | 32 |
| Reflexion | 16 | 16 |
| EvoShift | 16 | 16 |

### Medium source stream

The eight-task `evoshift_bbh_full.yaml` stream was run with
`benchmark.phase_size=8`, producing 64 first-pass examples.

| Method | Score | External requests | Total tokens |
|---|---:|---:|---:|
| Static | 0.9375 | 64 | 43,379 |
| EvoShift | 0.9375 | 228 | 189,558 |

EvoShift evaluated five state candidates: four memories and one policy patch.
One memory passed all replay gates; the other memories and the policy patch
were rejected. Relative to Static, one later answer improved and one regressed,
so the paired mean gain was zero. This is evidence that the adaptation state
machine executed, not evidence that adaptation improved this stream.

### Frozen held-out audit

The final source state contained one active memory. It was evaluated on 64
examples from four disjoint task families: `formal_fallacies`,
`geometric_shapes`, `hyperbaton`, and `web_of_lies`.

| Method | Score | External requests | Total tokens |
|---|---:|---:|---:|
| Static held-out control | 0.953125 | 64 | 43,958 |
| Frozen evolved state | 0.953125 | 64 | 52,372 |

The memory was retrieved for all 64 audit examples. Accuracy was unchanged and
token use increased by 8,414. The source and final state fingerprints were
identical:

```text
d249a1602f0a58846d7d0dd5084f22b725cd1ab4995dea4401edb2bf76ea5da4
```

The audit store recorded zero validations, zero promotions, zero rollbacks,
and `promotion_precision = null`/N/A.

### Provenance-scope regression validation

The first audit exposed a serving defect: with only one active memory, any
nonzero BM25 overlap normalized to relevance `1.0`, and utility/UCB could also
select a zero-overlap memory. Generic task wording therefore made the causal
card appear on every unseen domain.

The fix records the creating episode's domain as deterministic
`source_domains`, gates retrieval to matching provenance domains by default,
filters zero-overlap unscoped candidates, and exposes
`allow_cross_domain_transfer=true` only as an explicit ablation. For the legacy
source run used here, `source_domains` was backfilled from the episode/domain
trace referenced by the memory's provenance ID.

An offline retrieval replay over the same source and held-out samples produced:

| Retrieval policy | Source selections | Held-out selections |
|---|---:|---:|
| Provenance scoped | 8/64, all `causal_judgement` | 0/64 |
| Cross-domain ablation | 56/64 | 64/64 |

The scoped path therefore retained every target-domain sample while removing
all observed unrelated retrieval. This is a retrieval separation result, not
an accuracy result.

The fix was then re-evaluated with fresh provider calls from clean commit
`01aaf69c494218a2d4c921951c08d07ce4f8ad97`:

| Method | Run | Score | Input tokens | Total tokens | Retrieval episodes |
|---|---|---:|---:|---:|---:|
| Static control | `20260804T135712Z-static-73988af6` | 0.953125 | 36,354 | 43,860 | 0/64 |
| Frozen scoped state | `20260804T140229Z-audit-d6345c63` | 0.953125 | 36,354 | 43,777 | 0/64 |

The paired mean gain and 95% bootstrap interval were both exactly zero. Input
tokens were identical, proving that the earlier 8,414-token memory-context
overhead was removed; the 83-token total reduction came only from ordinary
output-length variation. Both runs used the same model, dataset hash, Git
commit, clean worktree, disabled cache policy, and 64 uncached requests, so
`evoshift compare` reported `comparable_for_claims=true` with no warnings.

The frozen state hash was unchanged before and after evaluation:

```text
10af9513a16f0a26e5b3bd6e736e44bb8746bdc2d422a583827034f8c75fae4d
```

This validates removal of measured negative transfer and cost overhead. It
does not show that the learned memory improves unseen-domain accuracy.

## Defects found by the live run

1. Shared cache hits could make a later method appear to use zero provider
   requests and zero latency while `compare` still marked it claim-comparable.
   Formal BBH configs now disable caching, cache-hit episode metrics are
   reported, and comparison emits separate accuracy/resource warnings.
2. Three large BBH files had embedded hashes calculated from truncated
   payloads. Their full pinned-revision sizes and SHA-256 values were corrected.
   `scripts/verify_bbh_manifest.py` now audits all 27 upstream files; the live
   audit passed 27/27.
3. The multiple-choice scorer missed an unambiguous option label placed at the
   end of an answer, such as `09/09/1908 (B)`. A restricted trailing
   parenthesized-label rule was added.
4. Yes/No BBH tasks used normalized exact match, incorrectly rejecting answers
   such as `Yes, Christie tells the truth.` A dedicated leading binary-label
   scorer now handles Yes/No and True/False targets.
5. Dirty-worktree runs were not excluded from formal comparison claims.
   Comparison now checks Git commit equality and `git_dirty=false` separately
   from score alignment.

## Reproduction order

```bash
python scripts/verify_bbh_manifest.py
evoshift data pull bbh --config configs/experiments/evoshift_bbh_full.yaml
evoshift run --config configs/experiments/evoshift_bbh_full.yaml \
  --set algorithm=static --set evolution.enabled=false
evoshift run --config configs/experiments/evoshift_bbh_full.yaml
evoshift run --config configs/experiments/static_bbh_heldout.yaml
evoshift audit --source-run runs/<evoshift-run> \
  --config configs/experiments/audit_bbh_heldout.yaml
```

For a formal result, run the preregistered repeated-seed sweep from one clean
commit and archive the ignored run directories in a separate artifact store.
