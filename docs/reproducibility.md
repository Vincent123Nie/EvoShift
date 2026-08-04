# Reproducibility guide

This guide distinguishes three levels of evidence:

1. **offline correctness**: unit/contract tests and deterministic demo clients;
2. **pipeline demonstration**: controlled SyntheticShift or demo JSONL runs;
3. **research evidence**: same-model public benchmark comparisons with pinned
   data, controlled budgets, uncertainty, ablations, and held-out audits.

The deterministic demo checks orchestration, candidate promotion, persistence,
and report generation. Rollback behavior is covered by deterministic component
tests. Neither one estimates real LLM self-evolution quality, and neither may be
presented as a benchmark result.

## Environment

The repository targets Python 3.11 locally and declares support for Python
3.9–3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional generic Hugging Face datasets support:

```bash
python -m pip install -e ".[dev,hf]"
```

Record the exact interpreter and dependency resolution used for a formal run:

```bash
python --version
python -m pip freeze > environment.freeze.txt
git rev-parse HEAD
git status --short
```

`environment.freeze.txt` is an experiment attachment, not necessarily a file to
commit. The current repository has bounded dependency ranges but no complete
lockfile, so `pip install` on different dates may resolve different patch
versions.

## Offline verification

`doctor` performs local checks and never calls a paid API:

```bash
evoshift doctor
```

Run the quality gates:

```bash
ruff check .
ruff format --check .
mypy src/evoshift
pytest --cov=evoshift --cov-report=term-missing
python -m build
```

Provider contract tests use mocked HTTP and sentinel credentials. Fake and demo
clients make end-to-end tests possible without network access.

## Deterministic pipeline demonstration

Run the checked-in compact stream:

```bash
evoshift run --config configs/experiments/offline_demo.yaml
```

Or run the generated deterministic SyntheticShift configurations:

```bash
evoshift run --config configs/experiments/static_demo.yaml
evoshift run --config configs/experiments/self_refine_demo.yaml
evoshift run --config configs/experiments/reflexion_demo.yaml
```

`HeuristicDemoClient` is deliberately programmed to make common shifted-rule
errors, accept a matching verified experience card, and correct through the
Self-Refine path. It is useful for state-machine and regression tests only.

For an apples-to-apples four-algorithm pipeline comparison, keep one config and
change only algorithm behavior. For example, all four commands below use the
same checked-in JSONL stream and demo model:

```bash
evoshift run --config configs/experiments/offline_demo.yaml \
  --set algorithm=static --set evolution.enabled=false

evoshift run --config configs/experiments/offline_demo.yaml \
  --set algorithm=self_refine --set evolution.enabled=false

evoshift run --config configs/experiments/offline_demo.yaml \
  --set algorithm=reflexion --set evolution.enabled=true

evoshift run --config configs/experiments/offline_demo.yaml \
  --set algorithm=evoshift --set evolution.enabled=true
```

Do not compare `offline_demo.yaml` directly with another config unless their
resolved dataset, sample IDs, model behavior, and budgets match.

## Explicit paid API smoke test

The smoke command is the only diagnostic command that intentionally sends a
paid request:

```bash
export EVOSHIFT_OPENAI_API_KEY="your-key"
evoshift provider smoke --config configs/models/gateway_gpt56.yaml
```

It sends a minimal `/responses` request asking for `OK`. The key is read from
the environment and should never be placed in YAML, a command-line flag, or Git.

Passing smoke proves connectivity and response-shape compatibility. It does not
prove benchmark correctness, stable model behavior, cost accuracy, or proxy
idempotency.

## Pinned BBH workflow

The BBH adapter pins commit
`9ee07bd481feebf959a6b59d61ea57bdcf30964d` and verifies checked-in SHA-256
metadata for each selected task file.

Download and verify the smoke subsets:

```bash
evoshift data pull bbh \
  --config configs/experiments/evoshift_bbh_smoke.yaml
```

Then run:

```bash
export EVOSHIFT_OPENAI_API_KEY="your-key"
evoshift run --config configs/experiments/evoshift_bbh_smoke.yaml
```

Before the full run, estimate worst-case requests. One foreground episode costs
one call; each eligible critic adds one; each paired memory or policy replay can
cost up to `2 * validation_window` calls. The budget config is a hard local
guard, not a substitute for the provider's billing cap.

```bash
evoshift run --config configs/experiments/evoshift_bbh_full.yaml
```

Do not launch the full configuration merely because smoke passed. Inspect smoke
failure rate, promotion frequency, replay overhead, and provider invoice first.

## Frozen held-out audit

Run the no-memory control and evolved-state audit with the same held-out config,
model, seed, and cache policy:

```bash
evoshift run --config configs/experiments/static_bbh_heldout.yaml

evoshift audit \
  --source-run runs/<completed-source-run> \
  --config configs/experiments/audit_bbh_heldout.yaml

evoshift compare runs/<static-heldout-run> runs/<audit-run>
```

`evoshift audit` fails closed when the source run is incomplete, is itself an
audit, contains invalid/non-active memory state, uses a different configured
model, or has the same dataset hash as the target. It copies the state into a
new isolated SQLite database, disables outcome updates and all evolution calls,
and checks that the final policy/memory SHA-256 equals the source state hash.

Keep the source and audit run directories together. Whole-state gain is not a
substitute for per-card future utility; candidate-level held-out promotion
precision remains `N/A` unless separately measured.

## Same-model baseline protocol

A defensible experiment changes one algorithmic factor at a time.

Freeze:

- provider base URL and model/snapshot;
- system prompts and prompt-version identifiers;
- benchmark revision, subset order, limit, and sample IDs;
- feedback mode;
- evaluator;
- maximum output tokens and reasoning effort;
- cache policy;
- request/token/dollar budget;
- seed for sample ordering and bootstrap.

Run at least:

- static;
- Self-Refine;
- unverified Reflexion-style memory;
- full EvoShift/VERA;
- VERA ablations relevant to the claimed contribution.

Use separate experiment YAML files or CLI overrides that resolve to a documented
matrix. Archive the resolved configuration produced in every run directory.

## Cache protocol

The default response-cache namespace is provider kind plus configured model.
It can be shared across runs. This improves cost and deterministic replay but
changes physical latency/cost.

Choose and declare one protocol:

1. disable cache for every compared run;
2. use a fresh, distinct cache path for every run;
3. deliberately prewarm an identical cache and report logical versus physical
   usage separately.

Example isolated path:

```bash
evoshift run --config configs/experiments/evoshift_bbh_smoke.yaml \
  --set storage.cache_path=data/llm_cache_evo_smoke.sqlite3
```

Never give one algorithm a warm cache and another a cold cache when comparing
latency or paid cost.

## Run artifact contract

Every completed run directory should contain:

- `manifest.json`: Git/config/dataset/model/environment identity;
- `resolved_config.yaml`: exact resolved experiment configuration;
- `predictions.jsonl`: normalized episodes and scores;
- `traces.jsonl`: retrieval IDs, policy version, shift, score, usage;
- `failures.jsonl`: typed attributions, when generated;
- `promotion_decisions.jsonl`: replay evidence, when generated;
- `metrics.json` and `report.md`;
- `costs.json`: complete connected-provider budget ledger;
- `summary.json`: final policy, active memories, counts, notes;
- `state.sqlite3`: versioned audit database when run isolation is enabled.

The manifest contains a SHA-256 hash of the normalized sample stream and a hash
of the resolved config. A dirty Git flag is recorded. A formal claim should use
a clean commit and retain the exact environment freeze because dependencies are
not fully captured in the manifest today.

Inspect memory state:

```bash
evoshift memory inspect \
  --database runs/<run-id>/state.sqlite3 \
  --status all
```

Regenerate a report from its metrics:

```bash
evoshift report runs/<run-id>
```

## Paired comparison

Compare only runs with the same unique sample IDs:

```bash
evoshift compare runs/<baseline-run-id> runs/<candidate-run-id>
```

The command aligns by `sample_id`, calculates per-sample deltas, paired
bootstrap confidence bounds, post-shift gain, and resource summaries. It writes
comparison JSON/Markdown into the candidate directory.

Before quoting the output, manually verify:

- configured and returned model identity are equivalent;
- sample order and dataset hash match;
- prompts and feedback modes match except for intended algorithm behavior;
- neither run exceeded its budget;
- cache protocol is the same;
- the comparison is not selected after trying many unreported variants.

## Randomness and repeated trials

The seed controls synthetic generation, deterministic shuffling, stable sample
selection, and bootstrap resampling. It does not make a remote reasoning model
deterministic. Model aliases can also change server-side without repository
changes.

For API experiments:

- prefer a dated/snapshot model when the gateway exposes one;
- omit unsupported sampling parameters consistently;
- repeat each major configuration enough times to measure provider variance;
- report mean and uncertainty across runs, not only the best seed/run;
- retain raw per-sample scores for paired hierarchical analysis;
- record the time window because provider implementations can change.

Caching identical calls can reduce repeated-call variance, but then the result
measures a cached policy execution rather than fresh stochastic generations.
State which one is intended.

## Hyperparameter protocol

Tunable values include:

- retrieval `top_k`, BM25 `k1/b`, score weights, MMR lambda, and the
  `allow_cross_domain_transfer` ablation;
- memory token budget, write threshold, dedup threshold;
- Page-Hinkley `delta`, threshold, minimum instances, novelty EWMA and cooldown;
- replay window and protected quota through phase configuration;
- bootstrap samples/confidence;
- minimum gain, CI lower bound, regression and cost gates;
- rollback utility threshold and minimum uses.

Use an explicit calibration stream or validation subset. Freeze the chosen
configuration before running the final test. If the final test is consulted to
choose a threshold, it is no longer a test, and the bootstrap interval does not
repair that leakage.

## Result-reporting checklist

Before putting a number on a resume or slide, be able to provide:

- repository commit and dirty flag;
- run IDs and comparison artifact;
- resolved configs and environment freeze;
- configured model and provider time window;
- dataset revision/hash and selected sample IDs;
- baseline definitions;
- feedback access for each method;
- cache and budget protocol;
- metric definition, window, and confidence method;
- all relevant ablations, not only the winning variant;
- failed runs and selection procedure;
- limitations and any manual intervention.

Use “improved over same-model baselines on this pinned evaluation” unless a
leaderboard-compatible protocol actually demonstrates SOTA. The existence of a
deterministic demo, green tests, or successful memory promotions is not a SOTA
result.

## Known reproducibility gaps

- No complete dependency lockfile is currently committed.
- The configured model alias may not identify an immutable provider snapshot.
- Run manifests do not yet store the full dependency freeze or provider API
  version.
- Replay calibration is not a separate held-out promotion split. The frozen
  audit is a distinct future stream but operates on the final state as a whole.
- Remote calls can remain nondeterministic.
- The LLM cache is shared by kind/model unless isolated explicitly.
- `promotion_precision` defaults to replay-estimated rather than
  candidate-level future-realized gain. The implemented audit does not yet
  assign its whole-state transfer result back to individual promotions.

These gaps should be disclosed and, for publication-quality work, closed before
making a strong empirical claim.
