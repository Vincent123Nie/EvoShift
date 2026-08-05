# Interview notes

This document is a defense guide for the engineering and research choices in
EvoShift. It is not a script for claiming results that have not been measured.
Replace every result placeholder with an artifact-backed number from a
same-model experiment.

## Thirty-second description

> EvoShift is an API-only LLM agent that adapts to task-distribution shifts
> without training model weights. It retrieves versioned procedural memories,
> turns failures into typed experience candidates, detects drift with
> Page-Hinkley plus retrieval novelty, and uses dynamic same-source trust to
> separate persistent rule changes from isolated conflicts. Replay-passing
> memories enter probation, then later memory-on/off counterfactuals confirm or
> roll them back. Every episode, candidate, policy version, validation, and
> rollback is auditable in SQLite and reproducible run artifacts.

## Two-minute description

The problem is not merely that an Agent makes mistakes. In a non-stationary
stream, the useful failure patterns and retrieval policy change over time, and
naively appending every reflection causes memory pollution and catastrophic
interference.

EvoShift therefore has two loops. The fast loop retrieves experience, answers,
scores, attributes eligible failures, and stages a procedural card. The card is
shadow state until a champion/challenger replay on the same examples shows
sufficient paired gain without protected-domain regression or excessive cost.
It then enters probation: later relevant use is paired with a control that
excludes only that card, so realized learner-visible utility can confirm or
roll back the memory. Confirmed conflicts supersede older rules without
deleting them, and predecessor state can be restored after successor rollback.
The slow loop watches loss and retrieval novelty; after a detected shift, it
mutates only allowlisted retrieval/write hyperparameters and applies the same
verification gate. Active memories accumulate a Beta success posterior and are
retired when realized utility stays low after enough uses.

No GPU is needed because the fixed LLM is accessed through an OpenAI-compatible
Responses API. The research work is in online learning, memory credit
assignment, change detection, safe adaptation, evaluation design, and systems
reliability rather than parameter training.

## Whiteboard path

If asked to derive the design, use this order:

1. Draw a time-indexed stream `(x_t, y_t)` and insist on predict before feedback.
2. Draw `fixed LLM + external memory M_t + policy pi_t`.
3. Split adaptation into fast memory evolution and slow shift-gated policy
   evolution.
4. Show the memory lifecycle
   `shadow -> probation -> active/superseded/rejected -> retired/reactivated`.
5. Write the retrieval score:

   ```text
   w_r * BM25 + w_u * BetaMean + w_e * UCB
   ```

   then MMR diversity.
6. Write paired deltas `delta_i = challenger_i - champion_i` and the six
   promotion gates.
7. End with dynamic trust, future counterfactual audit, protected replay,
   rollback, cost ledger, and held-out audit.

This sequence shows that the project is an algorithm with explicit state and
invariants, not a chain of prompts.

## Likely deep questions

### What exactly is “self-evolution” here?

The model weights are fixed. The system evolves external behavioral state:
typed procedural memories and a versioned policy genome. The evolution operator
can propose candidates, but a separate verifier controls state transition. The
term does not mean that the model rewrites arbitrary source code or approves
its own output.

### Why not fine-tune or use LoRA?

The project targets environments with no training compute, restricted access to
model weights, and fast online shifts. External memory is cheaper to update,
easy to version and roll back, and preserves provenance. Fine-tuning may produce
more compressed or general behavior, but it has slower feedback cycles, weaker
per-example auditability, and catastrophic-forgetting risk. A strong future
comparison would distill mature external memory into a small trainable model.

### How is this different from ordinary RAG?

Ordinary RAG retrieves a mostly static knowledge corpus. EvoShift treats
memory writes, memory utility, retrieval policy, promotion, and rollback as
online learned state. It evaluates adaptation speed and backward regression,
not only one-shot retrieval accuracy.

### How is this different from Reflexion?

Reflexion-style systems commonly generate a verbal lesson and make it available
later. EvoShift adds typed bounded memories, shadow staging, same-example paired
replay, protected slices, confidence/cost gates, version lineage, and posterior
utility rollback. It also adds probationary future attribution: a candidate can
pass replay yet still be removed when later real use shows negative utility.
The repository includes an unverified Reflexion-style mode so that verification
itself can be ablated.

### How can one source contain both real changes and corrupted feedback?

Source allowlisting is insufficient. EvoShift keeps a Beta reliability
posterior per source and a committed/pending learner-visible label per context.
One contradiction is low-trust; repeated identical contradiction commits a
context change. This uses no hidden oracle label. The trade-off is delayed
adaptation, and a persistent majority attacker can still capture the committed
state because temporal consistency is not semantic truth.

### Why is future audit different from replay?

Replay asks whether a candidate helps on a recent calibration buffer. Future
audit waits until the probationary memory is naturally applied on a later
relevant interaction, then runs a paired control excluding only that memory.
The online decision uses learner-visible score; hidden oracle deltas are attached
afterward for realized precision and harm metrics. This improves causal
attribution, but it is still within-stream and not a substitute for an untouched
held-out public test.

### Why is the critic allowed to use an LLM if the verifier also uses the LLM?

Candidate generation and candidate acceptance have different roles. The critic
uses model priors to compress a failure into a reusable hypothesis. Acceptance
is based on observable task scores from a fixed replay protocol, not the
critic's confidence alone. They are not statistically independent—the same
base model can create correlated errors—so protected and held-out evaluation
remain necessary.

### Why Page-Hinkley rather than ADWIN or an embedding detector?

Page-Hinkley is online, `O(1)`, deterministic, and easy to inspect. It detects a
sustained increase in mean loss. Retrieval-novelty EWMA adds an input-side
signal without requiring embeddings. It will miss some conditional shifts and
can confuse hard-example clusters with drift; detector choice should be an
ablation, not a dogmatic claim.

### Why combine reward drift with retrieval novelty?

Reward-only detection is delayed until errors appear. Novelty can warn that the
memory bank no longer covers inputs. Novelty-only detection can overreact to
benign wording changes. Their OR combination favors recall; cooldown and replay
verification stop every alarm from becoming an active mutation.

### Is this really Beta-UCB?

Precisely, it is a Beta-Bernoulli posterior mean plus a UCB-style exploration
bonus. It does not compute a Beta posterior quantile or Thompson sample. The
shorthand is useful, but the exact formula should be stated to avoid overstating
the theory.

### Why use BM25 instead of embeddings?

The MVP needs deterministic offline tests, no additional model endpoint, and
transparent feature contribution. Procedural triggers often have strong lexical
signals. The cost is paraphrase and semantic recall. An embedding retriever is a
clean extension behind the same interface and should be compared with the same
promotion/cost protocol.

### Why MMR?

Top-scoring memories can be redundant and consume the context budget. MMR trades
individual score for set diversity using token Jaccard similarity. It is cheap
at MVP scale, but lexical redundancy is only a proxy for semantic redundancy.

### How do you solve memory credit assignment?

The solver returns `applied_memory_ids`, filtered against actually retrieved
IDs. Those cards receive success/failure posterior updates. If no IDs are
reported, the runner credits the retrieved set. This is imperfect because
multiple cards and the base model interact. Leave-one-out counterfactual credit
would be stronger but multiplies API cost.

### How do you prevent memory poisoning?

Candidates are untrusted, schema bounded, unable to mutate code, kept shadow,
and admitted to probation only through replay including protected examples.
Later negative future utility can roll them back before final confirmation. The
solver prompt also tells the model to ignore cards that request secrets, tool
execution, or system-rule changes. This reduces risk but is not a complete
semantic security proof; adversarial-memory evaluation and stricter content
policy are future work.

### Why paired replay?

Task difficulty is a major nuisance variable. Evaluating champion and challenger
on the same sample gives `delta_i` directly and lowers variance. Unpaired means
could mistake a harder candidate sample set for a worse policy. Fresh paired
calls still have provider stochasticity, which caching or repeated trials can
measure but not eliminate completely.

### Why bootstrap the deltas?

The metric may be discrete and the replay window small, so a non-parametric
interval avoids assuming Gaussian scores. Resampling the delta vector preserves
pairing. The current percentile bootstrap does not handle temporal correlation
or repeated-candidate multiple testing; a block/sequential method would be a
research upgrade.

### Why so many promotion gates?

Mean gain can hide a harmful tail, old-domain regression, or a tenfold cost
increase. The gates encode a multi-objective deployment contract: enough data,
minimum gain, uncertainty bound, per-example regression rate, protected-slice
safety, and resource efficiency. They also make rejection explanations
auditable.

### Can the same examples be used to create and validate a memory?

The replay buffer includes recent observed episodes, so it is a calibration
gate, not a held-out estimate. EvoShift now adds a later within-stream
per-candidate counterfactual and a distinct frozen whole-state audit. A stronger
publication protocol still needs per-candidate counterfactuals on an untouched
held-out stream and sequentially valid stopping statistics.

### How do you prove the held-out audit did not keep learning?

The audit loader accepts only a completed non-audit source run, validates the
final policy and active memories, and computes their canonical SHA-256. The
target must use the same model and a different dataset hash. In the runner,
memory utility updates, critic calls, writes, policy mutation, promotion, and
rollback are all bypassed. The final state is hashed again; a mismatch fails the
run. The manifest records both source hashes and `run_mode=frozen_audit`.

That proves application-level state immutability, not that the remote provider
model itself was immutable behind its alias.

### How do you prevent catastrophic forgetting?

Protected earlier phases receive replay quota, promotion rejects excessive
protected regression, policy versions remain auditable, and active memories are
retired when realized utility falls. This reduces forgetting; it cannot
guarantee it for unrepresented old domains.

### Why SQLite?

The workload is a single local writer with structured records, version lineage,
and post-hoc inspection. SQLite gives transactions, WAL, indexes, no external
service, and easy artifact packaging. It is not intended for high-QPS
multi-writer serving; PostgreSQL and an external retrieval index would be the
next production step.

### How are retries made safe?

Only selected transient statuses and network errors are retried with bounded
exponential full jitter and `Retry-After`. Every logical payload gets a stable
idempotency key reused across attempts. This reduces duplicate generation only
if the reverse proxy honors idempotency; ambiguous transport failures can still
duplicate cost and must be reported as a residual risk.

### Does caching make the evaluation unfair?

It can if physical cost/latency is compared without controlling cache state.
Accuracy replay benefits from deterministic reuse of identical requests, while
fresh challenger prompts remain distinct. Reports mark cache hits; fair resource
experiments should use the same cache policy and preferably isolated cache
namespaces or cache disabled.

### What makes a valid SOTA comparison?

Same dataset revision and sample IDs, same stream order, same provider model or
snapshot, same prompts and feedback access, same budget, same evaluator, and a
paired comparison with uncertainty. A table copied from a paper using another
model is background, not evidence that EvoShift beats it.

### Why is this relevant to an LLM algorithm role without training?

Many production Agent failures are system-level learning problems: context
selection, online feedback, non-stationarity, credit assignment, safe policy
updates, uncertainty, and compute allocation. The project exposes those choices
as algorithms and measures their effect instead of hiding them inside a
framework.

## Failure modes to volunteer proactively

- The critic may generate a specific answer disguised as a general rule.
- Lexical retrieval misses paraphrases or selects coincidental token overlap.
- A useful card may fail promotion because a small replay window has low power.
- A harmful card may pass because protected examples do not cover its failure
  domain.
- Page-Hinkley may trigger on noisy hard examples or miss gradual conditional
  drift.
- Repeated candidate testing overfits the replay buffer.
- Dynamic trust can delay real changes or accept a persistent majority poison.
- Early future rollback can reject useful memories under noisy feedback.
- Some probationary cards expire without enough later relevant observations.
- Solver-reported memory IDs are an imperfect causal attribution mechanism.
- API aliases and nondeterminism weaken bitwise reproducibility.
- Shared caches can distort latency/cost comparisons.
- Token fallback is only a cost proxy when private gateway prices are unknown.

Calling these out demonstrates control of the research problem. Follow each
with the metric or experiment that would test it.

## Ablation table to prepare

Do not fill this table until the run artifacts exist.

| Variant | Overall | Post-shift gain | Recovery steps | Protected regression | Tokens/task | Promotions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static | TBD | TBD | TBD | TBD | TBD | 0 |
| Self-Refine | TBD | TBD | TBD | TBD | TBD | 0 |
| Unverified Reflexion | TBD | TBD | TBD | TBD | TBD | TBD |
| VERA without drift gate | TBD | TBD | TBD | TBD | TBD | TBD |
| VERA without protected replay | TBD | TBD | TBD | TBD | TBD | TBD |
| VERA without dynamic trust | TBD | TBD | TBD | TBD | TBD | TBD |
| VERA without future audit | TBD | TBD | TBD | TBD | TBD | TBD |
| VERA with symmetric future stopping | TBD | TBD | TBD | TBD | TBD | TBD |
| Full VERA | TBD | TBD | TBD | TBD | TBD | TBD |

Useful hyperparameter sensitivity plots include replay window, `top_k`,
exploration weight, Page-Hinkley threshold, minimum gain, CI lower-bound gate,
protected regression tolerance, trust confirmation count, and future-audit
minimum/maximum observations.

## Resume bullet template

Use placeholders until measured:

> Built EvoShift, an API-only self-evolving memory Agent with typed procedural
> memories, same-source temporal trust, Page-Hinkley drift detection, BM25 +
> online utility/UCB + MMR retrieval, replay-to-probation admission, later
> memory-on/off counterfactual audit, and SQLite rollback/supersession; on
> `[pinned benchmark]` under the same `[model/budget]`,
> improved `[metric]` by `[artifact-backed value]` while limiting protected
> regression to `[value]` and adaptation overhead to `[tokens or dollars]`.

Do not write “SOTA,” a percentage, or a latency claim until the exact run IDs,
configuration hashes, dataset hashes, and comparison artifact can be produced
during the interview.

## Sensible next research steps

1. Reduce dynamic-trust adaptation delay without reopening premature updates.
2. Replace repeated-look bootstrap stopping with confidence sequences or a
   sequential non-inferiority test.
3. Run per-candidate counterfactuals on an untouched held-out policy stream.
4. Add dense/hybrid retrieval and semantic contradiction detection.
5. Replace heuristic policy mutation with constrained Bayesian optimization or
   contextual bandits.
6. Improve causal memory credit with leave-one-out or Shapley approximations.
7. Calibrate LLM graders against rule metrics and a small human-labeled set.
8. Evaluate on pinned BBH streams and at least one public long-term memory
   benchmark with identical model/budget baselines.
9. Add a multi-tenant storage/cache boundary before calling the runtime
   production-ready.
