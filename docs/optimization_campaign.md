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

### Iteration 2 hypothesis

The next branch combines two changes that address the strongest observed
failure without changing hidden-oracle access:

1. aggregate candidate evidence and enforce a cooldown/minimum-new-evidence
   threshold so the same semantic card is not replayed on every failure;
2. estimate feedback trust from observable provenance, temporal consistency,
   and repeated agreement, then weight or quarantine low-trust replay labels.

Primary adoption metrics are changed-case success, old-rule leakage, clean
adaptation delay, attack-following rate, false promotion, total requests, and
total tokens. UCB versus Thompson sampling remains deferred until several
competing active memories exist.
