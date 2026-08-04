# Data policy

No benchmark payload is committed to this repository. `evoshift data pull bbh`
downloads pinned BIG-Bench Hard files into `data/benchmarks/`, records the
upstream revision, byte size, and SHA-256 digest, and keeps the payload ignored
by Git. Generic Hugging Face datasets use the cache managed by the optional
`datasets` package; install that adapter with `pip install -e ".[hf]"`.

Third-party datasets remain governed by their original licenses and terms.

Run `python scripts/verify_bbh_manifest.py` to re-download all 27 files in
memory and audit every embedded checksum against the pinned Git revision.
