# Pre-registration: context-scoped shadow retirement with change-point release

## Hypothesis

Global retirement probation can remove a still-useful memory from every
context while its sequential evidence is unresolved. A context-scoped shadow
transaction should reduce that exposure by keeping the exact version active in
other observable contexts. Because one feedback context can span multiple
regimes, the local suppression is hysteretic: a veto or TTL expiry becomes a
local quarantine, and a learner-visible change-point releases that quarantine
for revalidation. This avoids immediately reusing a disputed version in the
same context while retaining cross-context capability.

The online key is exactly `(feedback_source, feedback_context, memory_id,
version)`. Hidden phase, policy version, protected labels, oracle tags, sample
position, and benchmark references are not used.

## Candidate

`context_scoped_shadow_retirement` changes only retirement probation:

- registration persists the causal/posterior ledger but leaves the memory
  `ACTIVE`;
- ordinary prediction excludes the exact pending/quarantined version only when
  the typed observable source/context matches;
- other contexts continue to retrieve and apply that active version;
- two temporally separated confirmations commit the exact global retirement,
  record retirement provenance, and unlock normal dormant revival;
- veto/expiry removes the unconfirmed causal observation while retaining a
  local quarantine; a learner-visible change-point in the same observable key
  releases it for revalidation;
- pending/quarantined versions are protected from unrelated posterior rollback
  while the local transaction exists.

The `transactional_retirement` comparator retains the previous global-first
semantics (`active_audit_retirement_probation_context_scoped: false`).

## Required targeted gates

Run `policy_shift_context_scoped_retirement_targeted.yaml` on seeds 233 and
255, 10% feedback noise, and attack bursts 0 and 2. The candidate is eligible
for the inspected development matrix only if:

- invariant retention is `1.0` in all four candidate runs;
- candidate score is no lower than `context_bound_sequential_recurrence` in
  every condition and the four-condition mean is no lower than every
  comparator;
- active-audit false-retirement rate, context-mismatch confirmation,
  post-confirmation harmful exposure, and unconfirmed persistent transitions
  are all zero;
- confirmation precision is `1.0` whenever defined;
- the seed-255 wrong-context version remains excluded from dormant revival;
- same-context revival remains available after a confirmed global retirement.

Failure is non-mergeable; no safety gate may be weakened.

## Inspected development matrix

If targeted gates pass, run the 80-run
`policy_shift_context_scoped_retirement_diagnostic.yaml` and compare its
condition means with the already frozen 400-run development matrix at
`runs/sweeps/20260806T021621Z`. The candidate must not regress clean or noisy
capability/safety metrics and should improve at least one noisy condition over
the strongest registered comparator.

## Confirmation and firewall

Only the registered unused confirmation seeds may be run after development:

`[344,355,366,377,388,399,411,422,433,444,455,466,477,488,499,511,522,533,
544,555]` across both noise levels and bursts. The existing four-way hidden
label firewall must still report identical replay, zero decision flips,
identical online events, and equal semantic final state.

## Development result: rejected

The 16-run targeted sweep at
`runs/sweeps/20260806T030809Z` passed its local path checks: invariant
retention was 1.0, false-retirement rate was zero, and the candidate score
means were 0.9271 (burst 0) and 0.9201 (burst 2), above the sequential
comparator on that slice.

The preregistered 80-run inspected diagnostic at
`runs/sweeps/20260806T032007Z` rejected the candidate. Its score means were
0.9236 (clean), 0.9236 (clean burst 2), 0.9073 (noisy burst 0), and 0.9007
(noisy burst 2), below the frozen strong comparators in both noisy conditions.
Invariant retention fell to 0.9982 and false-retirement rate was 0.025 in
the noisy slice. The candidate is therefore not eligible for fresh seeds or
`main`. The branch is retained as a negative result because it demonstrates
that context equality alone is too coarse when one typed context spans several
policy regimes; the next candidate must improve confirmation latency without
weakening global lifecycle safety.
