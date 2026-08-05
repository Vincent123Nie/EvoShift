# Pre-registration: lineage-conditioned counterfactual memory audit

## Diagnostic finding

The held-out cooldown experiment is now development evidence and cannot confirm
another candidate. Its episode traces expose a structural attribution error in
the active-memory circuit rather than a cooldown problem.

The current audit compares an applied successor card with an empty-memory
control. A successor can preserve most of its predecessor's rule while changing
only one policy clause. Removing the entire successor therefore also removes
still-valid shared capability. Under noisy feedback this caused foreground
losses on v2-compatible cases even though the useful causal question was
whether v3 or its explicit v2 predecessor was better for that context. Seeds 66
and 88 did not need dormant revival because v2 never became dormant; seed 111's
negative active-circuit delta came from this coarse control choice.

## Frozen candidate

Add a lineage-conditioned control to the reversible active-memory circuit.

1. When auditing an active memory with explicit `supersedes_memory_ids`, find
   exact stored predecessor versions that are currently `SUPERSEDED`, share the
   successor's scope, and are direct members of that declared lineage.
2. Rank eligible predecessors with the existing query/domain retrieval score
   and select at most one. Do not infer lineage from hidden benchmark metadata,
   text labels, oracle policy versions, or feedback-corruption annotations.
3. During low-trust canary registration, compare the served successor against
   the selected predecessor. Fall back to the existing memory-off control when
   no eligible predecessor exists.
4. During the matching foreground intervention, exclude the exact successor
   version and temporarily force the exact predecessor version. Confirmation
   compares that served predecessor output with a forced-successor control.
5. Keep the existing two-observation trust rule, exact-version checks, TTL,
   provisional-ledger rollback, retirement rule, and predecessor restoration.
6. The lineage substitution is ephemeral until ordinary retirement confirms;
   it cannot itself change persistent memory status.

The combined candidate retains the previously fixed 15-episode dormant-revival
configuration. No cooldown, trust, causal-delta, TTL, or bootstrap parameter is
part of this experiment.

## Variants

- `current_full`: adopted recurrence-aware rollback without speculative active
  or dormant circuits;
- `bidirectional_recurrence_circuits_15`: the rejected-but-promising prior
  candidate, unchanged;
- `lineage_counterfactual_circuits`: the prior candidate plus lineage-
  conditioned active-audit controls.

## Development and held-out protocol

The already inspected seeds `[66, 77, 88, 99, 111]` may be used only for
mechanism debugging and unit/integration tests. They cannot establish adoption.

After the implementation and named-variant tests are frozen, run one
confirmatory matrix on previously unused seeds `[122, 133, 144, 155, 166]`.
Cross the three variants with feedback noise `[0.0, 0.10]` and attack-burst
length `[0, 2]`, for 60 deterministic runs. Do not inspect alternative seeds or
change the candidate after confirmation begins.

## Adoption gates

- on fresh seeds, all four score point estimates improve over `current_full`;
- paired hierarchical 95% score CI lower bounds versus `current_full` are
  non-negative in clean, noise, burst, and noise+burst conditions;
- the lineage candidate is not below the prior bidirectional candidate in any
  condition and strictly improves its noisy and noise+burst point estimates;
- changed-case success improves over `current_full`, while protected/invariant
  and future-case success do not decrease;
- false retirement, false revival, poison persistence, and unconfirmed
  persistent transitions do not increase;
- circuit and revival confirmation precision remain `1.0` whenever defined;
- every lineage intervention uses an exact direct predecessor version, and a
  cancelled or expired canary leaves persistent state unchanged;
- no online choice reads oracle truth, policy phase/version, valid/stale tags,
  attack labels, or corruption annotations.

If any held-out score gate fails, retain the result as a negative experiment and
do not merge the mechanism into `main`.

## Development result and freeze

The diagnostic matrix completed at `runs/sweeps/20260805T112429Z` on the five
already inspected development seeds. The implementation, thresholds, variants,
metrics, and held-out seed list are frozen after this result.

| Condition | Current | Bidirectional-15 | Lineage control | Lineage delta vs current |
|---|---:|---:|---:|---:|
| clean | 0.9306 | 0.9514 | 0.9514 | `+0.0208 [+0.0111, +0.0319]` |
| attack burst | 0.9306 | 0.9514 | 0.9514 | `+0.0208 [+0.0111, +0.0319]` |
| 10% noise | 0.9167 | 0.9250 | 0.9278 | `+0.0111 [+0.0014, +0.0222]` |
| 10% noise + attack burst | 0.9056 | 0.9153 | 0.9181 | `+0.0125 [+0.0042, +0.0236]` |

The mechanism preserved clean capability and removed the two diagnosed
foreground losses in noisy seed 111. Across the diagnostic matrix, invariant
retention and selective-forgetting precision remained unchanged, false
retirement stayed zero, and the lineage variant strictly improved both noisy
point estimates over the prior candidate.

Pre-confirmation quality evidence: 162 tests passed with 85.09% branch-aware
coverage; Ruff, formatting, strict mypy over 54 source files, and package build
passed. These development results are encouraging but are not adoption
evidence.
