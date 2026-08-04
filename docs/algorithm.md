# VERA algorithm

VERA stands for **Verified Experience Replay and Adaptation**. It is EvoShift's
two-time-scale test-time learning algorithm. The language model remains fixed;
the evolving state consists of external procedural memories and an explicitly
bounded retrieval/write policy.

The central hypothesis is falsifiable:

> Under a non-stationary task stream, selectively promoted experience and
> shift-triggered policy mutation should improve post-shift utility faster than
> static prompting or unverified memory, without unacceptable regression on
> protected earlier domains or resource cost.

The implementation does not yet establish that hypothesis on a public
benchmark. Synthetic/demo behavior is a pipeline check only. A valid result
requires same-model baselines, repeated trials where meaningful, held-out
evaluation, uncertainty estimates, and cost reporting.

## Notation

At episode `t`:

- `x_t` is the task prompt and metadata;
- `y_t` is the benchmark reference or environment outcome;
- `r_t` in `[0, 1]` is the primary score;
- `M_t` is the set of active versioned memories;
- `pi_t` is the active `PolicyGenome`;
- `R(x_t, M_t, pi_t)` retrieves experience cards;
- `f_theta` is the fixed API model;
- `c_t` is measured token or dollar cost.

The first-pass prediction is:

```text
y_hat_t = f_theta(system_prompt, x_t, R(x_t, M_t, pi_t))
```

Only after `y_hat_t` is scored may feedback influence `M_(t+1)` or `pi_(t+1)`.

## End-to-end loop

```text
for sample in stream:
    memories = retrieve(sample.prompt, active_memory, champion_policy)
    prediction = solve(sample, memories)
    reward = evaluator(sample.reference, prediction.answer)
    update_memory_posteriors(prediction.applied_memory_ids, reward)
    shift = detector.update(1 - reward, retrieval_novelty)

    if eligible_for_experience_extraction:
        failure = critic(sample, prediction, reward, active_memory, shift)
        candidate = stage(failure.proposed_memory)
        if candidate is not rejected:
            decision = paired_replay(candidate, champion)
            activate(candidate) if decision.promote else reject(candidate)

    if shift.detected and recent_failures:
        patch = deterministic_bounded_mutation(champion_policy, recent_failures)
        challenger = apply_and_validate(patch)
        decision = paired_replay(challenger, champion_policy)
        champion_policy = challenger if decision.promote else champion_policy
```

Static, Self-Refine, Reflexion-style, and EvoShift modes share the same runner
but enable different behaviors. This is preferable to separate scripts because
the scorer, sample order, artifact format, and provider accounting remain
comparable.

## Retrieval: BM25 + Beta utility/UCB bonus + MMR

Each memory document concatenates its trigger, scope, tags, directive, and
anti-pattern. The tokenizer lowercases alphanumeric terms and adds CJK
bigrams. For query term `q`, document `d`, corpus size `N`, document frequency
`n_q`, term frequency `f(q,d)`, average document length `avgdl`, and policy
parameters `k1` and `b`, the sparse score is:

```text
IDF(q) = ln(1 + (N - n_q + 0.5) / (n_q + 0.5))

BM25(d, q) = sum_q IDF(q) * f(q,d) * (k1 + 1)
              / (f(q,d) + k1 * (1 - b + b * |d| / avgdl))
```

Raw BM25 is normalized by the largest candidate score for the current query:

```text
relevance_i = raw_i / max_j(raw_j)
```

Each memory maintains a Beta-Bernoulli success posterior initialized with
`alpha=1`, `beta=1`. When a credited memory contributes to a successful answer,
`alpha` increments; otherwise `beta` increments. Its posterior utility is:

```text
utility_i = alpha_i / (alpha_i + beta_i)
```

The exploration term is UCB-like:

```text
exploration_i = min(
    1,
    sqrt(2 * ln(total_memory_uses + 2) / (use_count_i + 1)) / 2
)
```

The pre-diversification score is:

```text
score_i = w_r * relevance_i
        + w_u * utility_i
        + w_e * exploration_i
```

This is often summarized as “Beta-UCB,” but the exact implementation is a Beta
posterior mean plus a UCB-style exploration bonus; it is not a posterior
quantile or Thompson sample. Stating that distinction is important when
defending the algorithm technically.

MMR then chooses up to `top_k` memories. If `S` is the selected set and `J` is
Jaccard similarity between token sets:

```text
MMR(i) = lambda * score_i
       - (1 - lambda) * max_{j in S} J(document_i, document_j)
```

This balances high utility with context diversity. The rendered memory context
is bounded by a conservative four-characters-per-token budget.

### Retrieval novelty

For an established memory bank, novelty is derived from the strongest lexical
match:

```text
novelty_t = 1 - max_i relevance_i
```

The runner suppresses cold-start novelty when there is no active/retrieved
memory, avoiding an automatic shift alarm simply because the system has not
learned anything yet. The tradeoff is slower detection of the first genuine
domain transition if task reward remains high.

## Online drift detection

VERA applies Page-Hinkley to bounded loss and augments it with an EWMA of
retrieval novelty.

```text
loss_t = 1 - clip(reward_t, 0, 1)
mean_t = mean_(t-1) + (loss_t - mean_(t-1)) / t
cum_t  = cum_(t-1) + loss_t - mean_t - delta
min_t  = min(min_(t-1), cum_t)
PH_t   = cum_t - min_t
```

A performance shift is eligible when:

```text
t >= min_instances
and cooldown == 0
and PH_t > threshold
```

Novelty uses:

```text
novelty_ewma_t = a * novelty_t + (1 - a) * novelty_ewma_(t-1)
```

and triggers when its configured threshold is crossed under the same
eligibility conditions. On detection, Page-Hinkley cumulative state resets and
a cooldown prevents immediate repeated policy mutations.

Why Page-Hinkley in the MVP:

- `O(1)` memory and time per episode;
- an interpretable threshold and tolerance `delta`;
- appropriate for an online mean-loss increase;
- simple enough to test deterministically.

It does not identify which feature shifted, assumes a relatively stable loss
process between changes, and can fire on clusters of hard but in-distribution
examples. ADWIN, CUSUM variants, embedding-distribution tests, and slice-aware
detectors are valid ablations rather than assumed improvements.

## Failure attribution and experience distillation

After an eligible failure, the critic receives a bounded structured view:

- task, domain, and stream phase;
- agent answer, confidence, and short rationale summary;
- reward and allowed feedback;
- selected memory IDs and summaries of a few active memories;
- the current shift report.

The critic must return a JSON object with a failure type, signature, evidence,
confidence, and procedural-memory fields. Supported failure categories include
write miss, retrieval miss, ranking error, conflict error, reasoning error,
format error, knowledge gap, tool error, and unknown.

Feedback modes have different scientific meanings:

- `reward_only`: the critic sees only success/failure and is the safest default;
- `grader_feedback`: it sees an external grader's text;
- `reference_upper_bound`: it sees the reference and is an oracle upper bound,
  not a fair online setting unless the environment truly reveals it.

The critic is LLM-driven, but promotion is not. Invalid critic JSON falls back
to a low-confidence generic candidate. Pydantic limits length, kind, confidence,
and status fields.

## Candidate staging and deduplication

A candidate below `write_confidence_threshold` is immediately rejected. An
eligible candidate is compared lexically against stored memories. If its token
Jaccard similarity exceeds `dedup_similarity_threshold`, it reuses the nearest
memory ID, increments the version, merges provenance, and retains accumulated
utility counts. Otherwise it receives a stable SHA-256-derived ID.

Every staged candidate remains `SHADOW`; normal retrieval only sees `ACTIVE`
items. This keeps model suggestions from becoming behavior before validation.

The current deduplication is intentionally simple and can both under-merge
paraphrases and over-merge lexically similar but semantically different rules.
An embedding or entailment-based deduplicator is a future comparison, not an
assumed replacement.

## Paired replay

The replay buffer mixes adaptation relevance with backward-compatibility
protection:

- roughly two thirds are recent ordinary episodes ranked by lexical similarity
  to the candidate trigger/scope, breaking ties by recency;
- up to roughly one third come from configured protected phases or explicitly
  protected examples;
- remaining slots are filled by recent episodes.

For a memory candidate, the control uses the current active memory set and the
challenger receives the shadow candidate as a forced extra card. With
`paired_replay=true`, both are freshly solved. If it is disabled, memory control
can reuse the recorded first-pass score while the challenger is fresh; this is
cheaper but more exposed to API nondeterminism. Policy verification always
reruns both champion and challenger policies.

For replay sample `i`:

```text
delta_i = score(challenger_i) - score(control_i)
```

The same sample pairing removes much of the variance due to task difficulty.
It does not remove provider nondeterminism, order effects, or temporal
dependence.

Resource comparison uses dollar cost when configured and nonzero. If private
gateway prices are unknown, total tokens become the cost proxy instead of
treating calls as free.

## Paired bootstrap and promotion gate

VERA resamples the vector of paired deltas with replacement and computes a
percentile confidence interval for their mean. Pairing occurs before
resampling; independently resampling champion and challenger would destroy the
within-task control.

A candidate is promoted only if all configured gates pass:

1. replay sample count is at least `min_validation_examples`;
2. mean paired gain is at least `min_mean_gain`;
3. bootstrap lower bound is at least `min_ci_lower_bound`;
4. the fraction of examples with negative delta is no more than
   `max_regression_rate`;
5. mean regression on protected examples is no more than
   `max_protected_slice_regression`;
6. relative cost increase is no more than `max_cost_increase_ratio`.

The decision stores every check, means, interval, deltas, regression rates, and
cost ratio. A positive average alone is therefore insufficient.

Limitations of the current statistical gate:

- a small replay window produces coarse, unstable intervals;
- percentile bootstrap assumes examples are exchangeable, while stream samples
  may be autocorrelated;
- repeated candidate testing creates a multiple-comparison problem;
- the buffer includes recently observed examples and is not a held-out test;
- replay-time gain can overestimate realized future utility.

Block bootstrap, sequential tests, false-discovery control, and a separate
promotion/audit split are natural research extensions.

## Slow-loop policy mutation

The MVP policy proposer is deterministic, not LLM-generated. It summarizes
recent `FailureType` counts and applies an allowlisted mutation:

- write/retrieval misses increase `top_k` and relevance weight;
- ranking errors increase utility weight and MMR lambda;
- conflict errors reduce `top_k` and lower the dedup threshold;
- otherwise, novelty controls a small increase/decrease in exploration weight.

Every value is clamped by the `PolicyGenome` schema. The patch includes its
base version, hypothesis, supporting failure IDs, and a stable fingerprint.
Applying a stale patch or changing a non-evolvable field raises an error.

This design trades search breadth for auditability. A later implementation can
add Bayesian optimization, bandits, or an LLM proposer while retaining the
same typed mutation and replay gate.

## Online utility and rollback

The solver returns `applied_memory_ids`; hallucinated IDs are filtered against
the retrieved allowlist. If the solver returns no credited IDs, the runner
conservatively credits the retrieved set.

For each credited active memory:

```text
success: alpha <- alpha + 1
failure: beta  <- beta  + 1
utility <- alpha / (alpha + beta)
```

After at least `rollback_min_uses`, an item is retired when utility is below
`rollback_utility_threshold`. Rollback is local to the harmful memory rather
than reverting the entire run. A newer active version also retires the prior
version.

This mechanism can misattribute success when several cards are retrieved. The
`applied_memory_ids` contract reduces but does not eliminate the credit
assignment problem. Counterfactual leave-one-memory-out evaluation would be
more accurate and substantially more expensive.

## Metrics and claims

The canonical report supports:

- mean score and success rate by phase;
- area under the adaptation curve;
- cumulative regret, by default against score `1.0`;
- shift indices and recovery steps;
- post-shift gain against a sample-aligned baseline;
- backward transfer and forgetting when a performance matrix is supplied;
- promotion precision;
- tokens, cost, and p50/p95 latency.

Two caveats matter:

1. `promotion_precision` without later realized gains uses replay-estimated
   mean delta and is optimistic; the run summary explicitly labels it as such.
2. Foreground resource metrics cover task-solving episodes, whereas
   `costs.json` from the budget ledger covers adaptation calls as well.

## Frozen held-out state audit

`evoshift audit` evaluates the final external state without allowing the audit
stream to influence that state. The source loader accepts only a completed
prequential run, parses the final `PolicyGenome` and active `MemoryItem`s, and
computes a canonical SHA-256 over their JSON representation.

The target run then enforces:

```text
target_model == source_model
target_dataset_hash != source_dataset_hash
critic/write/utility/policy mutation/promotion/rollback == disabled
final_state_hash == source_state_hash
```

The runner still records first-pass scores, resource usage, and drift reports;
the detector output is observational and cannot trigger a state transition in
audit mode. If any policy or memory field changes, finalization raises instead
of emitting a successful audit.

This protocol estimates forward transfer of the complete evolved state. It
does not identify which individual memory caused the transfer. Per-card
leave-one-out audit is a separate, more expensive experiment.

A SOTA claim requires identical sample IDs, model/snapshot, prompts, feedback
access, token budget, and evaluator. Published values from another model are
context, not an apples-to-apples win.

## Complexity and cost envelope

For active-memory count `N`, replay window `W`, bootstrap samples `B`, and
generated token budget `G`:

- foreground solve: one API call and retrieval roughly linear in the current
  memory corpus;
- critic on an eligible failure: one additional API call;
- memory verification: up to `2W` solver calls with paired replay;
- policy verification: `2W` solver calls;
- bootstrap gate: `O(BW)` CPU time;
- memory state: `O(N)` plus all retained versions in SQLite.

The expensive term is replayed API inference, not local statistics. Practical
controls are `validation_window`, `min_validation_examples`, shift cooldown,
candidate count, cache, maximum requests/tokens/cost, and provider concurrency.

## Required ablations

To establish where gains come from, compare at least:

- static memory versus no memory;
- Self-Refine versus one-pass solve;
- unverified Reflexion-style activation versus verified activation;
- VERA without the drift gate;
- VERA without policy evolution;
- relevance only versus relevance + utility + exploration;
- no MMR;
- no protected replay;
- recorded control versus freshly paired control;
- different replay windows and promotion thresholds.

Hyperparameters must be selected on a calibration split or earlier stream,
then frozen before the final test. Tuning on reported test outcomes invalidates
the confidence interval regardless of how sophisticated the bootstrap is.
