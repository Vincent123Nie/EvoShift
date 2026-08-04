# EvoShift

> Verified test-time self-evolution for API-only LLM agents under distribution
> shift.

EvoShift is a research-oriented Python framework for agents that improve from
environment feedback without fine-tuning model weights or requiring a GPU. Its
core algorithm, **VERA** (Verified Experience Replay and Adaptation), evolves
versioned procedural memory and a bounded retrieval policy while keeping the
base LLM frozen.

The project is designed for reproducible Agent/LLM-algorithm experiments, not
as a claim of state-of-the-art performance. Its emphasis is on non-stationary
evaluation, falsifiable promotion decisions, rollback, cost accounting, and an
immutable held-out audit path.

## Core pipeline

```mermaid
flowchart LR
    X["Task stream x_t"] --> R["Retrieve active memories"]
    R --> S["Frozen LLM solver"]
    S --> Y["Score before feedback"]
    Y --> D["Page-Hinkley + novelty drift"]
    Y --> C["Failure attribution"]
    C --> M["Shadow experience card"]
    M --> V["Paired champion/challenger replay"]
    D --> P["Allowlisted policy patch"]
    P --> V
    V --> G{"Quality, CI, regression, cost gates"}
    G -->|pass| A["Promote version"]
    G -->|fail| J["Reject"]
    A --> U["Online utility + rollback"]
```

VERA has two time scales:

- The fast loop turns eligible failures into typed experience cards, stages
  them in shadow state, and promotes them only after paired replay.
- The slow loop detects sustained reward/retrieval shift and proposes a typed
  patch over allowlisted retrieval and write hyperparameters.

No model-generated Python is executed. Every episode, memory version, policy,
validation decision, event, and rollback is persisted with provenance.

## What is implemented

- OpenAI-compatible `/responses` client with bounded retries, `Retry-After`,
  stable idempotency keys, caching, budgets, usage/cost normalization, and
  secret redaction.
- Deterministic fake/demo providers for offline tests and full pipeline demos.
- Pinned BIG-Bench Hard adapter with fixed Git revision, SHA-256, byte-size,
  manifest, and canary validation.
- Generic Hugging Face and JSONL adapters plus a deterministic distribution
  shift benchmark.
- BM25 retrieval with Beta posterior utility, UCB-style exploration, MMR
  diversity, token budgeting, versioning, deduplication, and rollback.
- Paired bootstrap promotion gates with protected-slice and resource checks.
- Immutable run artifacts, SQLite audit state, sweeps, paired run comparison,
  and Markdown/JSON reports.
- Frozen held-out audit that enforces the same model, rejects source-stream
  reuse, disables all adaptation, and proves state immutability with SHA-256.

The Static, Self-Refine, and Reflexion modes are controlled conceptual
baselines implemented in this repository; they are not claimed to be official
reproductions of the original projects.

## Quick start

Python 3.11 is recommended; the package supports Python 3.9–3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

evoshift doctor
evoshift benchmark list
evoshift run --config configs/experiments/offline_demo.yaml
```

The offline demo is a deterministic state-machine test. It is not evidence of
real-model quality.

## OpenAI-compatible provider

Credentials and endpoint overrides are environment-only:

```bash
export EVOSHIFT_OPENAI_API_KEY="your-key"
export EVOSHIFT_OPENAI_BASE_URL="https://api.openai.com/v1"
export EVOSHIFT_OPENAI_MODEL="gpt-5.6"

evoshift provider smoke --config configs/models/gateway_gpt56.yaml
```

`EVOSHIFT_OPENAI_BASE_URL` may point to a compatible reverse proxy. Request
metadata is local-only by default because some gateways reject unknown fields.

## Public BBH experiment

```bash
evoshift data pull bbh \
  --config configs/experiments/evoshift_bbh_smoke.yaml

evoshift run --config configs/experiments/evoshift_bbh_smoke.yaml \
  --set algorithm=static --set evolution.enabled=false
evoshift run --config configs/experiments/evoshift_bbh_smoke.yaml
```

For formal comparisons, use the same BBH config and override only the algorithm
and preregistered ablation fields. The full stream and bounded sweep specs are
under `configs/experiments/` and `configs/sweeps/`. The committed BBH experiment
configs disable the shared LLM cache so run order cannot create zero-cost,
zero-latency results for later methods.

## Frozen held-out audit

After a completed EvoShift run, evaluate its final state on unseen BBH task
families without consuming audit feedback:

```bash
evoshift run --config configs/experiments/static_bbh_heldout.yaml

evoshift audit \
  --source-run runs/<source-run-id> \
  --config configs/experiments/audit_bbh_heldout.yaml

evoshift compare runs/<static-heldout-run> runs/<audit-run>
```

The audit command:

- accepts only a completed prequential source run;
- validates and fingerprints the exact final policy and active memories;
- requires the target model to match the source model;
- rejects a target dataset hash equal to the source stream;
- skips memory utility updates, critic calls, writes, policy mutations,
  promotions, and rollbacks;
- recomputes the final state hash and fails if any state changed.

This measures whole-state forward transfer. Per-candidate held-out attribution
and held-out promotion precision remain future work.

## Reproducibility and quality gates

```bash
ruff check .
ruff format --check .
mypy src/evoshift
pytest --cov=evoshift --cov-report=term-missing
python -m build
python scripts/verify_bbh_manifest.py
```

Verified locally on 2026-08-04:

- 72 tests passed;
- branch-aware coverage: 82.89% (`fail_under = 80`);
- Ruff and strict mypy passed;
- source and wheel distributions built successfully;
- the OpenAI-compatible provider contract suite passed, and an opt-in live
  smoke against a private compatible gateway returned exactly `OK`;
- four BBH task files and 16 public smoke samples passed integrity checks.

The live smoke proves protocol compatibility only. It is not a benchmark result.

## Run artifact contract

Each run writes an isolated directory containing:

- `manifest.json` and `resolved_config.yaml`;
- `predictions.jsonl`, `traces.jsonl`, and failure/promotion logs;
- `metrics.json`, `costs.json`, `summary.json`, and `report.md`;
- `state.sqlite3` with versioned memories, policies, validations, and events.

Use `evoshift compare` for same-sample paired confidence intervals and resource
deltas. Do not report a number that cannot be traced to a run ID, config hash,
dataset hash, and comparison artifact.

## Documentation

- [Architecture](docs/architecture.md)
- [Algorithm](docs/algorithm.md)
- [Evaluation protocol](docs/evaluation.md)
- [Research landscape](docs/research_landscape.md)
- [Live API engineering validation](docs/live_validation.md)
- [Reproducibility](docs/reproducibility.md)
- [Security](docs/security.md)
- [Limitations](docs/limitations.md)
- [Interview defense notes](docs/interview_notes.md)
- [Results template](docs/results_template.md)

## Security

API keys are never accepted as CLI arguments or persisted in requests, cache
keys, run artifacts, or logs. Benchmark data, caches, `.env`, and generated run
directories are ignored by Git. Model-generated memories are untrusted bounded
data and cannot mutate executable code.

## License

Private portfolio and research repository. All rights reserved. Third-party
papers, codebases, and datasets retain their original licenses.
