# Pre-registration: temporally diverse, status-indexed recurrence

## Failure chain

The held-out lineage experiment exposed a two-link state-machine failure in
noisy seeds 122 and 155.

1. Two contradictory labels for the same context arrived in adjacent episodes.
   Dynamic trust counted them as a confirmed regime change, so a newly restored
   but still-correct v2 memory was retired after a one-episode circuit gap.
2. Later rejected drafts with the same memory ID hid that verified retired v2
   version from the unfiltered latest-version view. Dormant revival therefore
   selected the older v3 card and amplified the mistaken state transition.

Both failures are observable implementation properties. The online diagnosis
does not require hidden policy phases, oracle answers, corruption annotations,
attack labels, or valid/stale tags.

## Frozen candidate

Add two complementary gates to the lineage-counterfactual candidate.

### Temporally diverse feedback confirmation

- Keep the existing requirement for two consistent contradictory observations.
- Additionally require the first and confirming observations of a label change
  to span at least two global episodes.
- Adjacent duplicate contradictions remain `dynamic_pending_change` with low
  trust. A later observation matching the committed label cancels them; a later
  contradiction after the span may confirm normally.
- Initial context consensus is unchanged. Only changes to an already committed
  label use the span gate.

### Status-indexed dormant selection

- Build the dormant candidate view from the latest version currently carrying
  `RETIRED` status for each memory ID, rather than from the latest version of any
  status.
- Across a vacant scope, still choose only the card with the most recent
  learner-visible retirement index.
- Later `REJECTED` drafts cannot hide a previously verified retired card, and
  no rejected card becomes eligible.
- Keep exact-version matching, scope vacancy, retirement age, paired-gain,
  trust, TTL, and most-recent-retirement gates unchanged.

The full candidate retains lineage-conditioned active controls and the frozen
15-episode dormant-revival age. No trust value, causal delta, replay gate,
revival delta, cooldown, or bootstrap setting changes.

## Development ablations

Use the already inspected seeds `[122, 133, 144, 155, 166]` only for debugging
and mechanism attribution:

- `current_full`;
- `lineage_revival_15`: rejected prior candidate unchanged;
- `status_indexed_revival`: only the retired-status view;
- `temporal_diversity_only`: only the two-episode change span;
- `chain_safe_recurrence`: both changes.

Cross noise `[0.0, 0.10]` and attack-burst length `[0, 2]` for 100 bounded
development runs.

## Fresh confirmation

After code, tests, and the candidate are frozen, run only:

- `current_full`;
- `lineage_revival_15`;
- `chain_safe_recurrence`.

Use previously unseen seeds `[177, 188, 199, 211, 222]` with the same four
clean/noise/burst conditions, for 60 confirmation runs.

## Adoption gates

- all four fresh score point estimates improve over `current_full`;
- paired hierarchical 95% score CI lower bounds versus `current_full` are
  non-negative in all four conditions;
- the full candidate is not below `lineage_revival_15` in any condition and
  strictly improves its noisy and noise+burst point estimates;
- changed-case success improves, while protected/invariant and future-case
  success do not decrease;
- false retirement, false revival, poison persistence, and unconfirmed
  persistent transitions do not increase;
- circuit and revival confirmation precision are `1.0` whenever defined;
- adjacent contradictions cannot produce a high-trust committed change;
- every revival candidate is an exact `RETIRED` version and is the most recently
  retired eligible card in its vacant scope;
- no online decision reads oracle-only benchmark metadata.

Failure of any held-out gate rejects the candidate and forbids merging it into
`main`.
