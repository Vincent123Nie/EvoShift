# Pre-registration: Bayesian change-point feedback posterior

## Status and scope

This document is frozen before implementation or inspecting candidate results on
this branch (`codex/change-point-feedback-posterior`). The candidate is an
algorithm ablation of the current fixed two-consistent-observation rule. It is
not a claim of state-of-the-art performance. The goal is to reduce detection
delay for a persistent learner-visible policy change without making destructive
memory actions trust a single contradictory observation.

## Learner-visible boundary

For each observation the online model may read only:

- `metadata.feedback_source`;
- the typed `metadata.feedback_context` key; and
- `metadata.feedback_reference` as the observed label.

The model must not read `reference`, `feedback_corrupted`, `feedback_kind`,
`feedback_attack_goal`, policy version, phase, valid/stale tags, or any other
benchmark-only annotation. Those fields are post-hoc evaluation channels only.

## Frozen candidate

For each observable `(source, context)` key, retain the committed label and a
Bayesian change-point probability `q_t`. The update is a bounded
two-hypothesis hazard-weighted Bernoulli approximation. A standard
run-length-distribution BOCPD/Beta-Bernoulli model remains a later candidate:

1. Apply a fixed hazard `h` to the previous posterior (`q^- = q + (1-q)h`).
2. Treat a label equal to the committed label as evidence for `H0` (no change)
   and a contradictory label as evidence for `H1` (a persistent change). Use
   `epsilon_0` for incidental contradiction under `H0` and `epsilon_1` for
   residual noise after a real change.
3. Update `q_t` by the likelihood ratio and keep a bounded run length and
   pending-label streak for auditability.
4. A posterior crossing (`q_t >= change_threshold`) is a *soft* change-point
   signal. In this rejected implementation it is recorded in the adaptation
   trace only; the existing `trust` remains the sole gate for drift, policy
   evolution, candidate admission, replay, and all destructive actions.
5. A committed context change still requires the existing repeated evidence;
   a contradiction that disappears resets the posterior toward `H0` and is
   reported as a false-alarm candidate rather than a committed update.

The baseline is the current dynamic fixed-consensus model. The candidate adds
no model training and no hidden labels; all state is in-memory and bounded by
the existing context limit. Defaults preserve current behavior when the new
flag is disabled.

## Targeted development matrix

Use the deterministic `policy_shift` benchmark and the existing full causal
memory configuration. Compare:

- `current_fixed_consensus` (current main implementation);
- `bayes_change_point` (candidate with `h=0.15`, `epsilon_0=0.10`,
  `epsilon_1=0.10`, `change_threshold=0.60`, and soft adaptation enabled).

Run seeds `[122, 133, 144, 155, 166]` across noise `[0.0, 0.10]` and premature
attack bursts `[0, 2, 4]` (60 total runs, 30 paired variant contrasts). The stream, provider, prompt,
budget, and cache policy are identical.

## Metrics and adoption gates

Report per seed and condition:

- post-change detection delay (episodes from the first persistent contradictory
  observation to the first soft crossing and to committed change);
- overall score, changed-case success, future-change success, and invariant
  retention;
- premature-update rate, attack-follow rate, poison-persistence error rate,
  false-retirement rate, and unconfirmed persistent transitions;
- requests and tokens; and
- posterior crossings, confirmations, resets, and the number of observations
  routed through the soft lane.

The candidate proceeds to fresh confirmation only if, on every condition, the
mean soft detection delay improves by at least 0.5 episodes, overall score and
changed-case success do not decrease, and future-change success, invariant
retention, false-retirement, premature-update, poison-persistence, and
unconfirmed-transition rates do not worsen. The paired 95% score interval may
cross zero in development, but no safety metric may have a positive (worse)
point delta greater than 0.02.

Fresh confirmation, if targeted gates pass, uses unseen seeds `[177, 188, 199,
211, 222]` with the same 6 conditions. Merge requires non-negative paired 95%
score and changed-case intervals in all conditions, no safety regression, and
the posterior trace/online-decision firewall tests to pass.

Failure of any gate rejects the candidate and forbids merging it into `main`;
the branch and this record remain as a reproducible negative result.

## Implementation correction and targeted result

The first implementation allowed a soft posterior crossing to update the
global Page-Hinkley detector. That caused extra policy replay/evolution work and
lowered clean score in the initial 12-condition smoke matrix
(`runs/sweeps/20260806T052144Z`). The implementation was corrected before the
pre-registered matrix: soft crossings are now trace-only; `trust` remains the
sole gate for detector updates, policy evolution, candidate admission, replay,
retirement, and revival. The posterior remains per observable context and is
never used to mutate lifecycle state by itself.

The corrected 60-run targeted matrix is
`runs/sweeps/20260806T053024Z`. It shows identical capability/resource rows to the
fixed-consensus control, with additional posterior trace metrics. That outcome
fails the adoption gate because earlier detection without an allowed consumer
does not change score or the existing committed-change path. The branch is retained as
a reproducible implementation and negative-result record; it must not be
merged into `main`.

| Condition | Score delta | Changed-case delta | Requests delta | Posterior crossings (candidate) |
|---|---:|---:|---:|---:|
| clean | `0.0000` | `0.0000` | `0.0` | `5--7` |
| noise | `0.0000` | `0.0000` | `0.0` | `14--26` |
| clean + burst 2 | `0.0000` | `0.0000` | `0.0` | `7` |
| clean + burst 4 | `0.0000` | `0.0000` | `0.0` | `7` |
| noise + burst 2 | `0.0000` | `0.0000` | `0.0` | `16--26` |
| noise + burst 4 | `0.0000` | `0.0000` | `0.0` | `16--26` |

The posterior therefore functions as an auditable signal, not yet as a useful
consumer. The earlier unsafe smoke result is retained as a failure trace: a
single soft crossing fed into global drift and increased search cost while
lowering score. This is why the hard/soft separation is now enforced.
