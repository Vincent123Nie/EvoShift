# Security and privacy

EvoShift processes benchmark prompts, model outputs, feedback, generated
memories, and API credentials. This document describes the current controls and
their limits. It is not a claim of compliance certification or hardened
multi-tenant isolation.

## Threat model

The MVP considers:

- accidental API-key disclosure through configuration, logs, exceptions, or
  cache keys;
- authorization forwarding to an unexpected host;
- prompt injection or poisoned experience entering long-lived memory;
- model-generated mutation of executable code or unrestricted configuration;
- duplicate paid requests during retry;
- unbounded token/request/cost consumption;
- stale or cross-run cache entries affecting experiments;
- benchmark or artifact data leaking from the local machine;
- tampered downloaded benchmark files;
- corrupted or partially written experiment state.

Out of scope for the current local MVP:

- hostile users sharing one running service;
- encrypted database/search over sensitive production data;
- operating-system compromise;
- a malicious or compromised LLM/reverse-proxy provider;
- formal noninterference or semantic prompt-injection proofs.

## Credential handling

Production CLI paths resolve the API key from the environment variable named by
`provider.api_key_env`, normally `EVOSHIFT_OPENAI_API_KEY`. The key is not a YAML
field and is not accepted as a CLI flag. The Python `ResponsesClient`
constructor permits an explicit key for tests and controlled embedding, so
callers must not serialize that argument.

The provider adapter:

- sends the key only in `Authorization: Bearer ...`;
- does not place it in the canonical cache payload or idempotency key;
- never logs headers, request bodies, response bodies, or exception text from
  transport failures;
- redacts the exact configured key, Bearer-looking values, and `sk-...` tokens
  from remote error messages;
- disables HTTP redirects, reducing the risk of forwarding authorization to a
  redirected host;
- exposes a `repr` containing model and credential-free endpoint identity only.

Do not place credentials in the base URL, prompt, metadata, benchmark JSONL, or
shell history. A reverse proxy is part of the trusted computing base: it can see
the key and all submitted content.

If a credential is pasted into chat, a ticket, a terminal recording, or Git,
rotate it. Redaction in this repository cannot revoke copies outside the
process.

## Provider request safety

The Responses client retries network errors and HTTP 408, 409, 425, 429, 500,
502, 503, and 504 up to the configured bound. Delay uses exponential full
jitter and honors `Retry-After` as a minimum.

A stable SHA-256-derived `Idempotency-Key` is computed from the canonical JSON
payload and reused across attempts. This is a mitigation, not a guarantee:
OpenAI-compatible reverse proxies may ignore idempotency. A request that reaches
the provider but times out before the response can still be executed twice and
billed twice. Strict deployments should test proxy idempotency explicitly or
disable ambiguous transport retries.

Sampling parameters are omitted by default because reasoning-model gateways may
reject them. The adapter only sends `temperature` when
`allow_sampling_params=true`.

## Budget denial-of-wallet controls

`BudgetLedger` reserves the worst-case logical request envelope before a paid
call. Concurrent reservations are included when checking:

- maximum logical requests;
- maximum total tokens;
- maximum configured dollar cost.

On success, the reservation is reconciled with provider usage. If usage is
missing, the Responses client conservatively accounts using estimated input
tokens plus `max_output_tokens`. Failed attempted logical requests count toward
the request limit even when token usage is unknown.

Retries sharing an idempotency key count as one logical ledger request. If the
proxy executes duplicate attempts, actual provider billing can therefore exceed
the local ledger. Provider-side invoices remain the final authority.

The concurrency semaphore limits simultaneous logical calls. It is a local
process control, not a distributed rate limiter.

## Cache isolation and privacy

`SQLiteLLMCache` stores normalized `GenerationResponse` JSON and timestamps. It
does not store request bodies or authorization values; lookup uses a canonical
hash of provider identity and payload. The response `raw` object may still
contain sensitive model output or provider metadata.

Current cache namespaces are normally `provider-kind:model`. Consequences:

- two runs using the same provider kind/model can share cached responses;
- the cache is useful for cost control and deterministic replay;
- it is not a tenant or user isolation boundary;
- physical latency/cost comparisons can be biased by different cache warmth.

For sensitive or fair-comparison runs, disable caching or use a distinct
`storage.cache_path`. Keep the cache on an access-controlled local volume. The
cache is not encrypted at rest.

## Prompt and memory poisoning

Experience cards are untrusted model-generated data. Current defenses are:

- `MemoryItem` is a strict bounded schema;
- memories cannot introduce tools, functions, imports, or executable code;
- new candidates enter `SHADOW`, never normal retrieval;
- low-confidence candidates are rejected before replay;
- champion/challenger replay must pass task, protected-regression, uncertainty,
  and resource gates;
- the solver system prompt says to ignore cards that request secrets, tool/code
  execution, or changes to system rules;
- hallucinated `applied_memory_ids` are filtered against retrieved IDs;
- poorly performing active memories can be retired through posterior rollback.

Residual risk remains. Natural-language directives can contain indirect prompt
injection, exfiltration requests, biased policy, or behavior that passes a small
replay set. Protected replay only tests represented cases. Before production,
add content-policy validation, adversarial memory tests, provenance trust
levels, tenant-scoped stores, and human approval for high-impact domains.

## Policy mutation safety

The slow loop does not execute LLM-generated Python. `PolicyPatch.changes` is
restricted to an explicit allowlist, and `PolicyGenome` enforces numeric ranges.
Patches carry `base_version`; stale patches are rejected. Candidate policies are
stored as rejected or activated versions after replay, preserving lineage.

This prevents arbitrary code execution and broad configuration mutation. It
does not prove that every value inside the allowed range is operationally safe;
the replay and cost gates provide empirical control.

## Storage integrity and exposure

The experiment state database uses:

- parameterized SQL values;
- foreign keys;
- WAL mode;
- explicit transactions with rollback on failure;
- versioned memories and policies;
- a schema migration record.

Run JSONL files are flushed and `fsync`ed after append, and final text/JSON
files use atomic replacement where implemented. These controls reduce partial
writes but do not make artifacts tamper-evident. A future release should add a
signed manifest or per-file hash chain for stronger provenance.

Run artifacts intentionally contain benchmark prompts, references, answers,
failure summaries, retrieved memory IDs, and metrics. They are ignored by Git
but remain plaintext on disk. Do not run private customer data without an
approved retention policy, redaction layer, and encrypted storage.

The project stores short rationale summaries, not requested hidden
chain-of-thought. Provider raw responses may be present in the separate cache,
so inspect retention requirements there as well.

## Benchmark supply-chain controls

The BBH adapter pins a Git commit, limits individual download size, validates
the revision syntax, and checks every file against an embedded SHA-256/byte-size
manifest before use. It writes a local manifest atomically and rejects caches
that are incomplete, unmanifested, or tampered.

The generic Hugging Face adapter relies on the configured dataset revision and
the external `datasets` package. Pin the dataset revision and package versions
before using it for a formal claim.

Python dependencies are range-bounded in `pyproject.toml` but the repository
does not currently include a complete cross-platform lockfile. Generate and
review a lock or constraints file for a production or archival release.

## Logging and incident checklist

If a secret or sensitive prompt is suspected to have leaked:

1. stop the run and rotate/revoke the provider key;
2. preserve only the minimum logs needed for investigation;
3. inspect shell history, CI output, run artifacts, the LLM cache, crash traces,
   and Git history;
4. remove local sensitive artifacts using the organization's approved secure
   deletion process;
5. invalidate shared caches;
6. document the affected provider, time range, and data classes;
7. add a regression test before restoring service.

Never print a real key to verify redaction. Tests use synthetic sentinel keys.

## Deployment checklist

- Use a dedicated, least-privilege provider key and spending cap.
- Set explicit request, token, and dollar budgets.
- Confirm the reverse proxy supports TLS and expected idempotency behavior.
- Keep `follow_redirects` disabled.
- Use separate state/cache paths per tenant or environment.
- Disable or isolate cache for benchmark cost comparisons.
- Restrict filesystem permissions on `data/` and `runs/`.
- Review dataset licenses and data-export rules before sending prompts to an
  external API.
- Run contract tests for parsing, retry, and redaction.
- Treat model-generated memories as untrusted until verified.
- Require held-out and adversarial audits before enabling automatic promotion
  in a high-impact production workflow.
