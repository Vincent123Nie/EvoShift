# Rejected diagnostic: symmetric transactional retirement

## Frozen diagnostic slice

The preregistered implementation was first exercised on the previously
inspected tail seeds `233` and `255`, with feedback noise `0.10`, attack-burst
lengths `[0, 2]`, and all five development variants. This bounded 20-run slice
was used only for mechanism debugging; the fresh confirmation seeds were not
opened.

Across the four noisy conditions, mean score was:

| Variant | Mean score | Changed-case success | False-retirement rate |
| --- | ---: | ---: | ---: |
| `current_full` | 0.9236 | 0.6875 | 0.3750 |
| `lineage_revival_15` | 0.9184 | 0.6953 | 0.3333 |
| `semantic_revival` | 0.9184 | 0.7031 | 0.3750 |
| `transactional_retirement` | 0.9063 | 0.6406 | 0.2500 |
| `transactional_semantic_recurrence` | 0.9063 | 0.6406 | 0.2500 |

The symmetric transaction reduced measured false retirement, but failed the
primary score objective and is rejected before the full 400-run diagnostic.

## Failure mechanism

On seed `233`, a correct retirement was revisited under feedback trust `0.10`.
The post-retirement behavior was oracle-correct, while the forced old memory
was oracle-wrong, but corrupted learner-visible feedback reversed the pair.
The frozen rule treated insufficient trust as failed confirmation and rolled
the harmful memory back into `ACTIVE`, causing a long changed-case regression.

On seed `255`, a transaction registered just before a real policy shift and
was confirmed after the shift. The online action was correct at confirmation,
but the diagnostic labelled it false using the oracle phase at registration.
This is a metric timestamp bug, not an online-information leak.

## Decision

Reject the symmetric rule "anything other than decisive current-over-old gain
rolls back." Preserve these results as negative evidence. The next candidate
must separate three outcomes: confirm, veto/rollback, and defer.
