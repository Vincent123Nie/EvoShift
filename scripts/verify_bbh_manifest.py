"""Audit every embedded BBH checksum against the pinned upstream revision."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from evoshift.benchmarks import BBH_FILE_MANIFEST, DEFAULT_BBH_REVISION

RAW_ROOT = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard"


@dataclass(frozen=True)
class AuditResult:
    task: str
    expected_bytes: int
    actual_bytes: int
    expected_sha256: str
    actual_sha256: str

    @property
    def matches(self) -> bool:
        return (
            self.expected_bytes == self.actual_bytes and self.expected_sha256 == self.actual_sha256
        )


def audit_task(task: str, *, timeout: float) -> AuditResult:
    expected = BBH_FILE_MANIFEST[task]
    url = f"{RAW_ROOT}/{DEFAULT_BBH_REVISION}/bbh/{task}.json"
    request = urllib.request.Request(url, headers={"User-Agent": "EvoShift-manifest-audit/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return AuditResult(
        task=task,
        expected_bytes=int(expected["bytes"]),
        actual_bytes=len(payload),
        expected_sha256=str(expected["sha256"]),
        actual_sha256=hashlib.sha256(payload).hexdigest(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    results: list[AuditResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(audit_task, task, timeout=args.timeout): task for task in BBH_FILE_MANIFEST
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"ERROR {task}: {exc}")
                return 1

    mismatches = [result for result in results if not result.matches]
    for result in sorted(mismatches, key=lambda item: item.task):
        print(
            f"MISMATCH {result.task}: "
            f"bytes {result.expected_bytes}->{result.actual_bytes}, "
            f"sha256 {result.expected_sha256}->{result.actual_sha256}"
        )
    if mismatches:
        print(f"BBH manifest audit failed: {len(mismatches)}/{len(results)} mismatched")
        return 1
    print(f"BBH manifest audit passed: {len(results)} files at revision {DEFAULT_BBH_REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
