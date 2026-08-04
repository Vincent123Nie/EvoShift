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
