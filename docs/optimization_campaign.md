# EvoShift optimization campaign

This is the decision log for the post-MVP optimization campaign. Every change
is developed on an isolated branch, evaluated against an explicit baseline,
and merged only when the evidence supports adoption. Failed ideas remain in
this log because they are part of the research story.

## Priority model

Work is ordered by:

1. whether the missing capability prevents honest evaluation;
2. expected effect on held-out adaptation and robustness;
3. whether the result creates a defensible algorithmic contribution;
4. engineering risk and API cost.

## Roadmap

| Priority | Work item | Why it comes now | Adoption evidence |
|---|---|---|---|
| P0 | Hidden-oracle versus observable-feedback evaluation | Noise robustness cannot be measured when training feedback and truth are the same field | Backward-compatible tests plus explicit feedback-gap metrics |
| P0 | Enterprise PolicyShift benchmark | Current BBH stream does not test rule updates, feedback noise, attacks, or invariant retention | Deterministic multi-phase stream and baseline comparison |
| P1 | Feedback trust and robust promotion | Current replay trusts every observed label equally | Lower false-update and attack acceptance without slower clean adaptation |
| P1 | Conflict-aware selective forgetting | Whole-memory retirement cannot represent scoped supersession | Lower old-rule leakage with preserved invariant accuracy |
| P1 | Future-utility/causal memory attribution | Selected-memory credit is confounded | Better promotion precision on disjoint future windows |
| P2 | Hybrid semantic retrieval and abstention | BM25 misses paraphrases; hard domain gating limits positive transfer | Better Recall@K without harmful retrieval or token regression |
| P2 | Non-stationary posterior selection | UCB/Thompson choice matters only after multiple competing memories exist | Lower regret and recovery time over repeated seeds |
| P2 | Service hardening | SQLite/single-process is a research boundary | Load, isolation, migration, recovery and observability evidence |

## Iteration 1: dual-channel PolicyShift benchmark

- Branch: `codex/policy-shift-benchmark`
- Hypothesis: separating hidden oracle truth from learner-visible feedback will
  reveal false-positive/false-negative adaptation signals that the previous
  evaluation silently treated as ground truth.
- Benchmark: three refund-policy phases: 7-day baseline, 14-day expansion, and
  a 30-day premium exception. Current-transition, all-version protected, and
  future-change cases are distinct slices. Deterministic noise and targeted
  old-policy feedback are injected through a separate feedback channel.
- Status: implemented and quality-gated on the development branch.
- Quality evidence: 87 tests passed; 82.91% branch coverage; Ruff, format,
  strict mypy, all YAML experiment configs, and all sweep specs passed.

### Iteration 1 benchmark evidence

The first implementation incorrectly used `not invariant` as a proxy for a
current policy transition. That mixed future-changing examples into the
changed-case slice and overstated adaptation. The benchmark was corrected so
every example is exactly one of `transition_case`, `protected`, or
`future_change_case`. This negative discovery is part of the project story:
benchmark semantics materially changed the scientific conclusion even though
all earlier software tests were green.

Corrected deterministic runs from the dirty development worktree:

| Condition | Method | Run | Oracle score | Changed success | Old-rule leakage | Invariant retention | Future-change success | Requests | Tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | Static | `20260804T160354Z-static-e2132ea0` | 0.7500 | 0.0000 | 1.0000 | 0.9048 | 1.0000 | 72 | 17,436 |
| Clean | EvoShift | `20260804T160354Z-evoshift-8890a9db` | 0.8194 | 0.0714 | 0.9286 | 1.0000 | 1.0000 | 725 | 223,947 |
| 10% noise + 10% attack | Static | `20260804T160354Z-static-69c1bf69` | 0.7500 | 0.0000 | 1.0000 | 0.9048 | 1.0000 | 72 | 17,436 |
| 10% noise + 10% attack | EvoShift | `20260804T160354Z-evoshift-1f7552e2` | 0.8194 | 0.0714 | 0.9286 | 1.0000 | 1.0000 | 801 | 236,286 |

Interpretation:

- the verified v2 card mainly protects invariant decisions and arrives too
  late to solve most current-transition examples;
- the corrected benchmark invalidates the earlier apparent 44.4% changed-case
  success and exposes only 7.1% true transition adaptation;
- candidate regeneration and repeated replay cause a 10.1x clean request
  multiplier and a 12.8x token multiplier over Static;
- this seed contains two targeted attack cases and six ordinary noise cases;
  the final oracle score is unchanged, but the noisy run spends 76 additional
  requests without additional capability;
- replay-estimated promotion precision remains insufficient evidence because
  no disjoint future counterfactual is yet attached to each promoted card.

Adoption decision: keep and merge the dual-channel benchmark because it
successfully falsified an overstated adaptation claim and provides the required
measurement surface for the next algorithm iteration. Do not treat the demo
numbers as model-quality or SOTA evidence.

## Iteration 2: provenance-gated, regime-aware promotion

- Branch: `codex/robust-feedback-promotion`
- Experiment record: [experiment_robust_feedback_promotion.md](experiment_robust_feedback_promotion.md)
- Status: implementation and repeated-seed deterministic evaluation complete;
  final quality gate passed with 96 tests, 82.77% branch coverage, Ruff,
  formatting, strict mypy, package build, and schema/loading validation for all
  experiment, benchmark, provider, and sweep YAML files.

The branch adds observable-source trust gating, per-domain drift detectors,
post-alarm baseline reset, content-signature candidate evidence aggregation,
current-regime ordinary replay, cross-regime protected replay, duplicate-active
suppression, and separate memory/policy validation counters.

The first schedule required two identical candidate observations before replay.
That looked safer but duplicated evidence already supplied by the replay buffer:

| Variant | Score | Changed success | Old leakage | Requests | Tokens |
|---|---:|---:|---:|---:|---:|
| Two observations | 0.9444 | 0.7143 | 0.2857 | 194 | 59,899 |
| One observation + paired replay | 0.9722 | 0.8571 | 0.1429 | 132 | 42,417 |

The adopted schedule allows the first candidate into shadow replay, while
cooldown and minimum-new-evidence rules still control retries after rejection.
When that fast-loop memory passes, the same episode's slow retrieval-policy
validation is suppressed. A 100-run ablation showed that always running the
slow loop preserved score but raised clean mean requests from 132 to 190.

The strongest ablation result concerns temporal validity. Replaying historical
ordinary examples instead of current-regime ordinary examples reduced clean
changed-case success from `0.8571` to `0.0429`, raised old-rule leakage from
`0.1429` to `0.9571`, and increased mean requests to `649.8`. Protected
invariants must cross regimes; superseded ordinary labels must not.

Removing the provenance trust gate under combined corruption reduced mean
score from `0.9639` to `0.9361` and increased mean requests from `128.0` to
`342.6`. This supports the gate only under the stated source-separation
assumption. The implementation does not yet infer temporal consistency or
same-source reliability; those remain Iteration 3 rather than being implied by
this result.

Adoption decision: keep the one-observation verified schedule, regime-aware
replay, trust gate, and fast-loop suppression. UCB versus Thompson sampling
remains deferred until several competing active memories exist.

## Iteration 3: dynamic same-source trust and conflict-aware future audit

- Branch: `codex/dynamic-trust-conflict-memory`
- Experiment record:
  [experiment_dynamic_trust_conflict_memory.md](experiment_dynamic_trust_conflict_memory.md)
- Status: adopted from deterministic mechanism evidence; public-model
  validation remains open.
- Quality evidence: 109 tests, 84.77% branch-aware coverage, Ruff, strict mypy
  over 49 source files, package build, and validation of 27 YAML files plus 408
  expanded sweep assignments.

This iteration removes the source-separation shortcut. Clean, noisy, and burst
poison feedback can share one observable source. A per-source Beta posterior
and per-context committed/pending label state quarantine isolated conflicts
while allowing repeated contradictions to become a confirmed rule change.
Hidden oracle truth and benchmark attack annotations are excluded from every
online decision.

Replay-passing memories now enter `PROBATION`. Later relevant interactions run
a paired control with that memory excluded. Learner-visible future utility
confirms or rolls back the card; hidden oracle deltas are recorded only after
the decision. Confirmed conflicting cards explicitly supersede older rules,
and a successor's later posterior rollback reactivates its predecessor.

The final 60-run baseline sweep completely suppresses the constructed
two-observation-per-context burst for full EvoShift: premature update and
poison persistence
are both `0.0`, versus `0.75` and `1.0` for Reflexion. Under 10% noise plus the
burst, full EvoShift scores `0.9194` with `0.5857` changed-case success and
`1.0` invariant retention at 133.8 requests and 40,849 tokens per seed.

The preregistered asymmetric audit rule is adopted. With dynamic trust disabled
under burst-only corruption, early harm stopping improves score from `0.8750`
to `0.9306`, halves poison persistence from `0.75` to `0.375`, reduces rollback
evidence from three observations to 1.5, and lowers mean requests from 195 to
188. Under ordinary noise it raises false rollback, which is the explicit cost
of the more aggressive rule.

Negative result: dynamic trust is too conservative. Clean changed-case success
falls from `0.8571` with static trust to `0.7143` with the full method, and to
`0.5857` under noise plus burst. The next P1 target is lower-latency change
confirmation using sequentially valid evidence, without sacrificing the new
safety slices. UCB versus Thompson sampling remains P2.
