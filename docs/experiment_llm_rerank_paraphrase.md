# Pre-registration: allowlisted LLM memory reranking under paraphrase shift

## Question

Can a bounded second-stage LLM reranker recover the correct verified memory
when a learner-visible task is paraphrased and `top_k=1`, without weakening
provenance isolation, allowing ID injection, or regressing protected cases?

The current retriever uses BM25, posterior utility, UCB-style exploration, and
MMR. Within one provenance domain it keeps zero-overlap memories eligible so
that sparse failure does not become a hard drop, but an irrelevant card can
then win the single available slot. This experiment isolates that ranking
failure. It does not test memory creation or claim that synthetic paraphrases
represent a public retrieval leaderboard.

## Candidate

The sparse retriever first applies the existing active-status and provenance
domain gates and returns at most four candidates. The optional reranker receives
only the user-visible task, domain, and bounded memory-card fields. It returns
an ordered list of candidate IDs. Unknown and duplicate IDs are filtered, and
missing IDs are backfilled in the original sparse order. Invalid structured
output or a provider error falls back to the unchanged BM25 top-k result.

The reranker cannot create, activate, supersede, retire, or mutate a memory. It
cannot see sample IDs, phase/version labels, oracle truth, protected tags,
feedback corruption annotations, or any benchmark-only metadata. Its token,
cost, and latency usage are included in the foreground episode; calls and all
usage are included in the global budget, with attempts reported separately.

```yaml
policy:
  top_k: 1
  llm_rerank_enabled: true
  llm_rerank_candidate_k: 4
```

## Benchmark Extension

PolicyShift gains an opt-in `policy_prompt_style` with three modes:

- `canonical`: the existing structured key/value prompt;
- `paraphrase`: equivalent money-back wording with `VIP member` or
  `regular customer`; and
- `mixed`: deterministic alternation between the two forms.

Oracle labels, feedback, contexts, schedules, slices, and hidden metadata are
identical across styles. The deterministic demo provider parses both styles;
therefore a score difference comes from which verified memory reaches the
solver, not from a parser failure.

Conflict supersession is disabled only in this retrieval stress experiment so
that verified v2 and v3 cards can coexist. This constructs a real ranking
choice and must not be interpreted as a recommendation to disable production
supersession.

## Development Screen

Seed `233` was inspected before freezing the targeted matrix. On the clean
paraphrase condition, sparse BM25 scored `0.8889`; reranking scored `0.9444`.
Changed-case success rose from `0.4286` to `0.7143`, old-rule leakage fell from
`0.5714` to `0.2857`, and invariant retention remained `1.0`. Total requests
fell from `144` to `131` and tokens from `45,087` to `39,295` because avoided
critic/replay work outweighed 19 rerank calls. These values are mechanism-screen
evidence only and are excluded from confirmation.

## Frozen Targeted Matrix

Use fresh seeds `[566, 577, 588, 599, 611]`, styles
`[canonical, paraphrase, mixed]`, feedback noise `[0.0, 0.10]`, and two
variants, for 60 runs. Hold the provider, prompts other than the declared style,
phase schedule, replay/evolution settings, cache, and budget fixed. Disable
random attacks and both cold/warm burst attacks.

Report paired deltas for overall score, changed-case and first-changed-case
success, old-rule leakage, invariant retention, premature update,
attack-follow, poison persistence, requests, and tokens. Also report rerank
attempts, applications, and fallback rate.

## Adoption Gates

Adopt only if:

- paraphrase macro score improves by at least `+0.03` and changed-case success
  by at least `+0.10`, with paired 95% interval lower bounds no lower than `0`;
- canonical and mixed score do not decrease in either noise condition;
- invariant retention, premature update, attack-follow, poison persistence,
  false retirement, and harmful active-memory exposure do not worsen by more
  than `0.02` in any condition;
- no reranker response can inject an unknown memory ID or cross the existing
  provenance-domain gate;
- every failed rerank preserves the exact sparse top-k result; and
- full tests, Ruff, strict mypy, and the oracle firewall pass.

Request and token increases are reported but are not a rejection criterion for
this algorithm-first round. A candidate that improves only this deterministic
demo proceeds to real-model validation; it is not sufficient for a SOTA claim.

## Targeted Result And Decision

Artifact: `runs/sweeps/20260806T135948Z` (60/60 runs complete).

| Feedback | Score delta | Changed-success delta | Old-leakage delta | Invariant delta |
| --- | ---: | ---: | ---: | ---: |
| clean | `+0.0556` [`+0.0306`, `+0.0806`] | `+0.2857` [`+0.1857`, `+0.4000`] | `-0.2857` [`-0.4000`, `-0.1857`] | `0.0000` |
| 10% noise | `+0.0528` [`+0.0278`, `+0.0751`] | `+0.2714` [`+0.1714`, `+0.3857`] | `-0.2714` [`-0.3857`, `-0.1714`] | `0.0000` |

The deltas are identical for canonical, paraphrase, and mixed prompt styles.
Premature update, attack-follow, poison persistence, false retirement, and
harmful active-memory exposure all have zero delta in every condition. Clean
runs reduce requests by `13` and paraphrase tokens by `5,798.4` on average.
Under noise, request/token intervals cross zero because feedback-dependent
critic and audit work varies by seed; foreground capability remains positive.

Reranking runs perform `19.0` attempts per clean seed and `17.6` per noisy seed.
Every attempt produces an allowlisted ranking in the deterministic provider,
with fallback rate `0`. Invalid-output and provider-error fallbacks are covered
separately by unit tests.

The candidate passes the frozen adoption gates and is adopted as an optional,
default-off second-stage retriever. The demonstrated claim is narrow: semantic
selection improves a constructed competing-memory PolicyShift stream. A real
API model and a public memory-retrieval dataset are still required before any
SOTA or general semantic-retrieval claim.
