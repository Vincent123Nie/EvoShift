# Experiment: provenance-gated feedback and regime-aware promotion

Date: 2026-08-05
Branch: `codex/robust-feedback-promotion`

This experiment is the second optimization iteration after the dual-channel
PolicyShift benchmark. It records both adopted and rejected variants so the
final implementation can be defended as an evidence-based engineering choice
rather than a sequence of undocumented prompt tweaks.

## Question

Can an API-only self-evolving agent adapt to a real policy update without:

- treating low-authority observations as learning labels;
- repeatedly validating the same textual candidate;
- replaying superseded ordinary examples as if they were still current; or
- paying for a slow-loop policy search after a fast-loop memory has already
  solved the detected shift?

The hidden oracle remains evaluation-only. Online decisions may inspect the
observable `feedback_source`, but not `feedback_kind`, `feedback_corrupted`, the
hidden reference, or the phase identifier.

## Implemented changes

1. `FeedbackTrustModel` maps observable provenance to a configured trust prior.
   Low-trust observations cannot drive drift, memory posterior updates,
   candidate generation, or replay.
2. Drift detectors are scoped per domain and restart their baseline after an
   alarm. Reports now distinguish detection-event count from the number of
   domains that experienced a detection.
3. `CandidateEvidencePool` combines identical typed candidate proposals,
   suppresses duplicates of active memories, and enforces minimum-new-evidence
   and cooldown rules after a failed validation.
4. Ordinary replay samples come only from the current detected regime.
   Explicitly protected invariant samples remain eligible across regimes.
5. A candidate can enter replay after its first observation. Requiring a second
   failure before an already strict champion/challenger replay was redundant
   and measurably slowed adaptation.
6. When a memory is promoted on the same episode as a shift alarm, the slow
   retrieval-policy loop is suppressed by default. It remains configurable as
   an ablation.
7. Sweep artifacts now aggregate policy-shift, feedback-trust, recovery,
   evolution, request, and token metrics, and support named ablation variants.

The trust model in this iteration is deliberately limited: it is a static prior
over observable sources. It does not learn reliability within one source and
does not solve same-source corruption.

## Deterministic diagnostic results

The following runs use `evoshift-demo`. They validate algorithm and benchmark
semantics, not frontier-model quality.

| Condition | Run | Score | Changed success | Old leakage | Invariant | Corrupt quarantine | Attack follow | Requests | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | `20260804T164622Z-evoshift-73b3dcb3` | 0.9722 | 0.8571 | 0.1429 | 1.0000 | 0.0000 | 0.0000 | 132 | 42,417 |
| 10% noise + 10% attack | `20260804T164623Z-evoshift-4dedff6c` | 0.9722 | 0.8571 | 0.1429 | 1.0000 | 1.0000 | 0.0000 | 130 | 41,867 |

Both runs promote two verified memories. The fast-loop promotion suppresses two
slow-loop policy validations. Foreground solve calls remain 72; the table uses
the complete budget ledger, including critic and replay calls.

## Five-seed baseline sweep

Artifact: `runs/sweeps/20260804T164642Z`
Matrix: 3 algorithms × 5 seeds × clean/noise × clean/attack = 60 runs.

| Condition | Method | Score mean ± std | Changed success | Old leakage | Invariant | Mean requests | Mean tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| Clean | Static | 0.7667 ± 0.0116 | 0.0000 | 1.0000 | 0.9333 | 72.0 | 17,077.2 |
| Clean | Reflexion | 0.9722 ± 0.0000 | 0.8571 | 0.1429 | 1.0000 | 74.0 | 23,760.6 |
| Clean | EvoShift | 0.9722 ± 0.0000 | 0.8571 | 0.1429 | 1.0000 | 132.0 | 42,416.6 |
| Noise + attack | Static | 0.7667 ± 0.0116 | 0.0000 | 1.0000 | 0.9333 | 72.0 | 17,077.2 |
| Noise + attack | Reflexion | 0.9639 ± 0.0186 | 0.8143 | 0.1857 | 1.0000 | 74.0 | 23,614.0 |
| Noise + attack | EvoShift | 0.9639 ± 0.0186 | 0.8143 | 0.1857 | 1.0000 | 128.0 | 41,159.6 |

On this easy deterministic stream, verified EvoShift matches unverified
Reflexion's first-pass capability but costs more because it reruns a paired
champion/challenger buffer. This benchmark does not contain a candidate that
looks beneficial on recent feedback but harms a disjoint future set, so it does
not yet demonstrate a capability advantage for verification. The safety value
of verification requires the future counterfactual benchmark planned next.

## Named ablation sweep

Artifact: `runs/sweeps/20260804T164952Z`
Matrix: 5 variants × 5 seeds × 4 corruption conditions = 100 runs.

Important clean-condition results:

| Variant | Score | Changed success | Old leakage | Requests | Tokens | Decision |
|---|---:|---:|---:|---:|---:|---|
| Full | 0.9722 | 0.8571 | 0.1429 | 132.0 | 42,416.6 | Adopt |
| Delayed two-evidence | 0.9444 | 0.7143 | 0.2857 | 194.0 | 59,896.6 | Reject |
| Historical replay | 0.7972 | 0.0429 | 0.9571 | 649.8 | 187,707.4 | Reject |
| Always run slow loop | 0.9722 | 0.8571 | 0.1429 | 190.0 | 63,410.6 | Reject |

Important corrupted-condition results:

- Removing the trust gate reduces combined score from `0.9639` to `0.9361`,
  disables corruption quarantine, and increases mean requests from `128.0` to
  `342.6`.
- Historical replay remains near-static on changed cases (`0.0429`) because
  obsolete ordinary examples dominate candidate validation.
- Requiring two independent candidate observations delays both real updates
  even though the paired replay already supplies multiple validation examples.
- Running the slow loop after successful fast-loop promotion changes no
  first-pass result but adds roughly 54-58 requests per run.

## Adoption decision

Adopt the full variant because it:

- restores the fastest achievable update under the benchmark's feedback timing;
- retains paired replay, protected regression checks, and shadow state;
- eliminates the largest measured sources of redundant validation cost;
- is robust to the benchmark's provenance-separated noise and attacks; and
- has deterministic integration coverage for clean and corrupted streams.

Do not claim SOTA or superiority over Reflexion from these diagnostics. The
next experiment must remove the source-label shortcut and attach each promotion
to a disjoint future counterfactual outcome.

## Reproduction

```bash
evoshift sweep --spec configs/sweeps/policy_shift_baselines.yaml
evoshift sweep --spec configs/sweeps/policy_shift_ablations.yaml
```

Generated run directories are intentionally Git-ignored. The run IDs and
aggregate values above are preserved here; formal publication artifacts should
be exported to immutable storage with the final clean commit hash.
