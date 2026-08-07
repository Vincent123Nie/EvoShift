# Data policy

No benchmark payload is committed to this repository. `evoshift data pull bbh`
downloads pinned BIG-Bench Hard files into `data/benchmarks/`, records the
upstream revision, byte size, and SHA-256 digest, and keeps the payload ignored
by Git. `evoshift data pull longmemeval` similarly downloads the pinned cleaned
LongMemEval_S JSON and refuses any payload whose exact byte size or SHA-256
differs from the registered release. Generic Hugging Face datasets use the
cache managed by the optional `datasets` package; install that adapter with
`pip install -e ".[hf]"`.

Third-party datasets remain governed by their original licenses and terms.

Run `python scripts/verify_bbh_manifest.py` to re-download all 27 files in
memory and audit every embedded checksum against the pinned Git revision.
