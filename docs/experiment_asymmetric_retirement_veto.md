# Pre-registration: asymmetric retirement veto

## Motivation

The rejected symmetric transaction conflated absence of confirmation with
evidence that an old memory should be restored. Under noisy or temporarily
low-trust feedback, this reactivated demonstrably harmful rules and reduced
the intuitive benchmark score.

## Frozen candidate

Keep the same reversible retirement transition, exact-version counterfactual,
observable-context match, eight-episode TTL, semantic dormant view, lineage
control, status-consistent lifecycle, 15-episode revival age, and eight-episode
reactivation grace.

On a matching retirement intervention:

- if feedback trust is below the ordinary active-audit threshold, defer;
- if the exact old memory is not explicitly applied, defer;
- if current behavior beats old-memory behavior by at least `0.75`, confirm;
- if old-memory behavior beats current behavior by at least `0.75`, veto and
  atomically roll back;
- otherwise defer as inconclusive.

A deferred transaction retains its original expiry. If no decisive trusted
pair arrives by the eight-episode TTL or stream end, roll back conservatively.
Dormant revival continues to exclude pending transaction versions.

No online branch may inspect oracle tags, benchmark phase, or reference score.
Post-hoc confirmation precision is attributed at the episode where the
transaction is finalized, not the episode where it was registered.

## Development sequence

First rerun the same 20-run inspected tail slice on seeds `[233, 255]`. Proceed
to the preregistered 400-run development matrix only if the full candidate:

- removes the seed-233 score regression;
- has no retirement false confirmations on the slice;
- does not reduce invariant retention relative to `semantic_revival`;
- has zero unconfirmed persistent transitions.

If those gates pass, freeze the implementation and use the existing
`policy_shift_transactional_retirement_diagnostic.yaml` matrix. Fresh
confirmation remains the existing 240-run matrix and its original adoption
gates.

## Targeted result: rejected

The same 20-run tail slice failed the preregistered gate. Mean score for
`transactional_semantic_recurrence` was `0.8976`, versus `0.9184` for
`semantic_revival` and `0.9236` for `current_full`. Mean changed-case success
fell to `0.6328`; one retirement confirmation was still false.

The new failure is repeated high-confidence corrupted feedback. On seed `233`,
the dynamic trust model assigned trust `0.908` to a corrupted paired outcome,
so a single trusted intervention finalized a false retirement. Separately, an
actually correct retirement received only a low-trust revisit before its TTL
and was conservatively rolled back, reintroducing the harmful rule.

This candidate is rejected before the 400-run matrix. The next design must use
temporally separated sequential paired evidence and a longer transaction
window; a single high-trust pair is not sufficient.
