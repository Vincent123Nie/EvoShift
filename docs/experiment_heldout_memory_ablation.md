# Pre-registration: held-out per-memory ablation audit

## Question

Do individual active EvoShift memory cards make a reproducible contribution on an
independent held-out stream, or is the observed gain attributable only to the
whole memory state / model sampling noise?

## Hypothesis

For each active card `m`, removing only `m` from the evolved state should reduce
held-out task performance when `m` is useful, while leaving unrelated/protected
behavior unchanged. A card is considered useful only when the paired full-state
minus-card delta is positive on the pre-registered primary slice and does not
introduce a protected regression.

## Protocol (frozen before inspecting results)

1. Train an ordinary prequential run and export its final active state.
2. Load a distinct held-out benchmark with a different dataset fingerprint.
3. Run a frozen audit for the full evolved state and one leave-one-memory-out
   audit per `(memory_id, version)` card.
4. Use the same model, resolved prompt/config, held-out sample IDs and seed for
   every condition. Disable the LLM cache for all conditions. No adaptation,
   memory updates, policy updates, feedback trust updates, or audits are allowed
   during evaluation.
5. Pair predictions by `sample_id`; the full-state run is the control for every
   ablation. The audit runner must not reuse the source training stream.
6. Preserve the exact source state fingerprint and record the removed card in
   each ablation manifest. Do not use oracle metadata for online decisions;
   oracle labels are used only for post-hoc slice metrics.

## Primary metrics

- `heldout_mean_score`: mean primary score over all held-out samples.
- `useful_card_delta`: `full_score - leave_one_out_score`; positive means the
  removed card helped.
- `protected_regression`: change in protected-slice success/score; a useful card
  must not trade away protected behavior.
- `changed_success`: success rate on post-policy-change samples.
- `future_change_success`: success rate on the future-change slice, when present.

## Secondary diagnostics

- success deltas and paired bootstrap 95% CIs per card;
- harmful contribution (`delta < 0`) and neutral contribution (`|delta| < 0.01`);
- retrieval coverage: fraction of held-out samples on which the card was
  retrieved/applied in the full-state control;
- candidate-level held-out coverage by source tag/domain;
- token/request deltas (descriptive only; only comparable when all caches are
  disabled and provenance is clean);
- state fingerprint and memory-count invariants.

## Decision gates

- Fail the audit if sample IDs are not exactly aligned, the model differs, cache
  policy differs, the source/held-out dataset hashes match, or any frozen run
  mutates the state.
- A card-level claim is valid only with a paired CI and a non-zero full-state
  retrieval/application count. Zero-coverage cards are reported as “not tested”.
- Do not call a card useful from a single positive mean: require positive paired
  delta and no protected regression on the pre-registered slice.

## Minimal validation matrix

The first implementation is validated on the deterministic demo and
`policy_shift` benchmarks. Real-model runs are optional and must use a secret
provided through the configured environment variable; credentials are never
stored in artifacts or Git.

