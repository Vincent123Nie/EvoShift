# EvoShift results template

Replace every `TBD` only from immutable run artifacts. Use `N/A` when an audit
was not performed. Never convert a missing result into zero.

## 1. Experiment identity

| Field | Value |
|---|---|
| Evaluation date | TBD |
| Git commit / dirty | TBD |
| Provider and model identifier | TBD |
| Provider model snapshot, if available | TBD |
| BBH revision | `9ee07bd481feebf959a6b59d61ea57bdcf30964d` |
| Dataset hash | TBD |
| Main stream subsets | TBD |
| Phase size | TBD |
| Seed list | TBD |
| Feedback mode | TBD |
| Scorer | TBD |
| Bootstrap resamples / confidence | TBD |
| Price table or token-only accounting | TBD |
| Held-out audit status | TBD |

## 2. Primary same-backbone comparison

Report mean and 95% interval across all preregistered paired seeds. Values in
parentheses may contain the across-seed standard deviation.

| Method | Mean score ↑ | Success rate ↑ | Post-shift gain vs static ↑ | AUAC ↑ | Recovery steps ↓ | Regret to 1 ↓ | Total provider tokens ↓ | Total provider cost ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static | TBD | TBD | — | TBD | TBD | TBD | TBD | TBD |
| Self-Refine | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Reflexion | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VERA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Primary paired effect: `VERA − strongest baseline = TBD` percentage points,
95% paired bootstrap CI `[TBD, TBD]`, exact McNemar `p = TBD`, Holm-adjusted
`p = TBD`.

Interpretation: TBD.

## 3. Per-seed results

| Seed | Method | Dataset hash | Mean score | Success rate | Post-shift gain | Total requests | Total tokens | Total cost | Completed without exclusion? |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| TBD | Static | TBD | TBD | TBD | — | TBD | TBD | TBD | TBD |
| TBD | Self-Refine | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | Reflexion | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | VERA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Excluded or failed runs, with preregistered reason: TBD / none.

## 4. Performance by BBH phase

| Phase | Subset | N per seed | Static | Self-Refine | Reflexion | VERA | VERA − strongest baseline |
|---:|---|---:|---:|---:|---:|---:|---:|
| 00 | `boolean_expressions` | TBD | TBD | TBD | TBD | TBD | TBD |
| 01 | `date_understanding` | TBD | TBD | TBD | TBD | TBD | TBD |
| 02 | `disambiguation_qa` | TBD | TBD | TBD | TBD | TBD | TBD |
| 03 | `multistep_arithmetic_two` | TBD | TBD | TBD | TBD | TBD | TBD |
| 04 | `causal_judgement` | TBD | TBD | TBD | TBD | TBD | TBD |
| 05 | `navigate` | TBD | TBD | TBD | TBD | TBD | TBD |
| 06 | `tracking_shuffled_objects_five_objects` | TBD | TBD | TBD | TBD | TBD | TBD |
| 07 | `word_sorting` | TBD | TBD | TBD | TBD | TBD | TBD |

Worst-domain delta: TBD. Protected-phase maximum regression: TBD.

## 5. Shift adaptation

Use known phase boundaries for the primary table. Detector-triggered results
belong in a separate diagnostic table.

| Boundary | From → to | Window | Static score | VERA score | Paired gain | Recovery steps | Cumulative regret after shift |
|---:|---|---:|---:|---:|---:|---:|---:|
| TBD | TBD | 20 | TBD | TBD | TBD | TBD | TBD |

| Detector diagnostic | Precision | Recall | False alarms / 100 episodes | Median detection delay |
|---|---:|---:|---:|---:|
| Page-Hinkley + novelty EWMA | TBD | TBD | TBD | TBD |

Detector diagnostics are optional until a ground-truth boundary evaluator is
implemented; mark them `N/A` rather than estimating manually.

## 6. Paired statistical tests

| Contrast | N paired samples | Mean delta | 95% paired bootstrap CI | VERA-only wins | Baseline-only wins | Exact McNemar p | Holm-adjusted p | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| VERA vs Static | TBD | TBD | [TBD, TBD] | TBD | TBD | TBD | TBD | TBD |
| VERA vs Self-Refine | TBD | TBD | [TBD, TBD] | TBD | TBD | TBD | TBD | TBD |
| VERA vs Reflexion | TBD | TBD | [TBD, TBD] | TBD | TBD | TBD | TBD | TBD |

Bootstrap level: task / clustered by seed and sample ID / other: TBD.

## 7. Promotion gate and held-out audit

### 7.1 Replay decisions

| Candidate type | Evaluated | Promoted | Rejected | Replay-estimated promotion precision | Mean validation gain | Mean regression rate | Worst protected regression | Mean cost delta | Rollbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Memory cards | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Policy patches | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 7.2 Independent frozen-memory audit

| Audit domain | Samples | Static frozen score | VERA frozen score | Transfer gain | Harm rate | Token delta | No audit feedback consumed? |
|---|---:|---:|---:|---:|---:|---:|---|
| `formal_fallacies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `geometric_shapes` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `hyperbaton` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `web_of_lies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Held-out promotion precision: TBD. Negative-transfer rate: TBD.

The whole-state audit is implemented. If per-candidate counterfactual audit has
not been run, write: **N/A — whole-state transfer measured; candidate-level
held-out promotion precision not evaluated.**

## 8. Continual-learning matrix

`A[i,j]` is performance on a fixed, no-feedback probe for domain `j` after
adapting through domain `i`.

| After phase ↓ / Probe domain → | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | TBD | — | — | — | — | — | — | — |
| D1 | TBD | TBD | — | — | — | — | — | — |
| D2 | TBD | TBD | TBD | — | — | — | — | — |
| D3 | TBD | TBD | TBD | TBD | — | — | — | — |
| D4 | TBD | TBD | TBD | TBD | TBD | — | — | — |
| D5 | TBD | TBD | TBD | TBD | TBD | TBD | — | — |
| D6 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | — |
| D7 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Backward transfer: TBD. Forgetting: TBD. Frozen held-out forward transfer: TBD.

## 9. Ablations

| Variant | Mean score | Post-shift gain | AUAC | Regret | Protected regression | Total tokens | Held-out transfer | Key conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Full VERA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Reference |
| No slow policy evolution | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Historical/unpaired control | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| No protected replay | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| No UCB exploration | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| No utility term | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| No MMR diversity | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| BM25-only retrieval | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Only the named component may differ from Full VERA. Record the resolved config
hash for every row.

## 10. Resource and latency breakdown

| Method | Foreground logical requests | External provider requests | Foreground input tokens | Foreground output tokens | External provider tokens | USD | Tokens / success | Foreground p50 ms | Foreground p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Self-Refine | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Reflexion | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VERA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Cache policy: TBD. Final cost/latency runs used disabled or per-run-isolated
cache: TBD. Cache hit rate is not currently persisted in run artifacts and
must be `N/A` unless collected separately. Provider pricing source/date: TBD.

## 11. External comparison and SOTA eligibility

| External result | Model | Dataset/split | Online adaptation allowed? | Prompt/tool match? | Compute match? | Reported score | Directly comparable? |
|---|---|---|---|---|---|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | No / Yes, because TBD |

SOTA claim status: **Not evaluated / Not eligible / Eligible with evidence:** TBD.

Mandatory caveats:

- The EvoShift BBH stream is an ordered eight-subset prequential protocol, not
  the official static full-BBH leaderboard aggregate.
- BBH is public and may be present in model pretraining data.
- Replay-estimated promotion precision is not held-out evidence.
- Foreground resource metrics exclude critic and replay calls; total cost comes
  from `costs.json`.
- A stable closed-model name may not identify an immutable model snapshot.
- LongMemEval and LoCoMo results are `N/A` until dedicated pinned adapters and
  official evaluation are implemented.

## 12. Final claim text

Use a claim no stronger than the evidence supports:

> Across TBD paired seeds on the pinned EvoShift-BBH prequential stream, VERA
> changed first-pass mean score by TBD percentage points versus TBD using the
> same base model. The paired 95% interval was [TBD, TBD]. Total provider usage
> changed by TBD, protected-phase regression was TBD, and frozen held-out
> transfer was TBD. These results do/do not support the preregistered claim.

Do not replace “EvoShift-BBH prequential stream” with “BBH SOTA” unless every
eligibility condition in `docs/evaluation.md` has independently been met.
