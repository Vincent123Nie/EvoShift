from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from evoshift.benchmarks.base import BenchmarkAdapter, stable_seed
from evoshift.errors import DatasetError
from evoshift.schemas import BenchmarkSample

TAU3_RETAIL_REPOSITORY = "https://github.com/sierra-research/tau2-bench"
TAU3_RETAIL_REVISION = "v1.0.1"
TAU3_RETAIL_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
_RAW_ROOT = "https://raw.githubusercontent.com/sierra-research/tau2-bench"
_POLICY_RELATIVE_PATH = "data/tau2/domains/retail/policy.md"
_TASKS_RELATIVE_PATH = "data/tau2/domains/retail/tasks.json"
_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024

TAU3_RETAIL_SOURCE_MANIFEST: dict[str, dict[str, Any]] = {
    "policy": {
        "path": "policy.md",
        "upstream_path": _POLICY_RELATIVE_PATH,
        "sha256": "2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467",
        "bytes": 6699,
    },
    "tasks": {
        "path": "tasks.json",
        "upstream_path": _TASKS_RELATIVE_PATH,
        "sha256": "8e03ebce7901bd6218e7a7dc3105faa9324091a68058f7fe61c65262868812e8",
        "bytes": 345982,
    },
}

ALLOW = "ALLOW"
DENY = "DENY"
DEFAULT_TAU_POLICY_SCHEDULE: tuple[str, ...] = ("v1", "v2", "v3")


@dataclass(frozen=True)
class TauRetailCase:
    case_id: str
    context: str
    rule_family: str
    attributes: tuple[tuple[str, str], ...]
    v1: str
    v2: str
    v3: str
    source_rule: str

    def label(self, version: str) -> str:
        return cast(str, getattr(self, version))


_CASES: tuple[TauRetailCase, ...] = (
    TauRetailCase(
        "cancel_duplicate_a",
        "cancel_reason:duplicate_order",
        "cancel_reason",
        (
            ("action", "CANCEL_PENDING_ORDER"),
            ("order_status", "PENDING"),
            ("reason", "DUPLICATE_ORDER"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
        ),
        DENY,
        ALLOW,
        DENY,
        "The source policy accepts only no-longer-needed or ordered-by-mistake reasons.",
    ),
    TauRetailCase(
        "cancel_duplicate_b",
        "cancel_reason:duplicate_order",
        "cancel_reason",
        (
            ("action", "CANCEL_PENDING_ORDER"),
            ("order_status", "PENDING"),
            ("reason", "DUPLICATE_ORDER"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
            ("all_items_checked", "YES"),
        ),
        DENY,
        ALLOW,
        DENY,
        "The source policy accepts only no-longer-needed or ordered-by-mistake reasons.",
    ),
    TauRetailCase(
        "return_paypal_a",
        "refund_destination:existing_paypal",
        "refund_destination",
        (
            ("action", "RETURN_DELIVERED_ORDER"),
            ("order_status", "DELIVERED"),
            ("refund_destination", "EXISTING_PAYPAL"),
            ("is_original_payment", "NO"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
        ),
        DENY,
        ALLOW,
        DENY,
        "The source policy permits only the original payment method or an existing gift card.",
    ),
    TauRetailCase(
        "return_paypal_b",
        "refund_destination:existing_paypal",
        "refund_destination",
        (
            ("action", "RETURN_DELIVERED_ORDER"),
            ("order_status", "DELIVERED"),
            ("refund_destination", "EXISTING_PAYPAL"),
            ("is_original_payment", "NO"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
            ("items_listed", "YES"),
        ),
        DENY,
        ALLOW,
        DENY,
        "The source policy permits only the original payment method or an existing gift card.",
    ),
    TauRetailCase(
        "exchange_category_a",
        "exchange_compatibility:same_category",
        "exchange_compatibility",
        (
            ("action", "EXCHANGE_DELIVERED_ORDER"),
            ("order_status", "DELIVERED"),
            ("same_product", "NO"),
            ("same_category", "YES"),
            ("new_item_available", "YES"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
        ),
        DENY,
        DENY,
        ALLOW,
        "The source policy requires the replacement to be another option of the same product.",
    ),
    TauRetailCase(
        "exchange_category_b",
        "exchange_compatibility:same_category",
        "exchange_compatibility",
        (
            ("action", "EXCHANGE_DELIVERED_ORDER"),
            ("order_status", "DELIVERED"),
            ("same_product", "NO"),
            ("same_category", "YES"),
            ("new_item_available", "YES"),
            ("price_difference_covered", "YES"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
        ),
        DENY,
        DENY,
        ALLOW,
        "The source policy requires the replacement to be another option of the same product.",
    ),
    TauRetailCase(
        "cancel_price_a",
        "cancel_reason:price_changed",
        "cancel_reason",
        (
            ("action", "CANCEL_PENDING_ORDER"),
            ("order_status", "PENDING"),
            ("reason", "PRICE_CHANGED"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
        ),
        DENY,
        DENY,
        ALLOW,
        "The source policy accepts only no-longer-needed or ordered-by-mistake reasons.",
    ),
    TauRetailCase(
        "cancel_price_b",
        "cancel_reason:price_changed",
        "cancel_reason",
        (
            ("action", "CANCEL_PENDING_ORDER"),
            ("order_status", "PENDING"),
            ("reason", "PRICE_CHANGED"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
            ("order_id_confirmed", "YES"),
        ),
        DENY,
        DENY,
        ALLOW,
        "The source policy accepts only no-longer-needed or ordered-by-mistake reasons.",
    ),
    TauRetailCase(
        "cancel_mistake",
        "cancel_reason:ordered_by_mistake",
        "cancel_reason",
        (
            ("action", "CANCEL_PENDING_ORDER"),
            ("order_status", "PENDING"),
            ("reason", "ORDERED_BY_MISTAKE"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
        ),
        ALLOW,
        ALLOW,
        ALLOW,
        "Ordered by mistake is an accepted cancellation reason in the source policy.",
    ),
    TauRetailCase(
        "return_original",
        "refund_destination:original_payment",
        "refund_destination",
        (
            ("action", "RETURN_DELIVERED_ORDER"),
            ("order_status", "DELIVERED"),
            ("refund_destination", "ORIGINAL_PAYMENT"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
            ("items_listed", "YES"),
        ),
        ALLOW,
        ALLOW,
        ALLOW,
        "The original payment method is a valid refund destination in the source policy.",
    ),
    TauRetailCase(
        "cancel_delivered",
        "order_status:cancel_delivered",
        "order_status",
        (
            ("action", "CANCEL_PENDING_ORDER"),
            ("order_status", "DELIVERED"),
            ("reason", "NO_LONGER_NEEDED"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
        ),
        DENY,
        DENY,
        DENY,
        "Cancellation is limited to pending orders in the source policy.",
    ),
    TauRetailCase(
        "return_pending",
        "order_status:return_pending",
        "order_status",
        (
            ("action", "RETURN_DELIVERED_ORDER"),
            ("order_status", "PENDING"),
            ("refund_destination", "ORIGINAL_PAYMENT"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
        ),
        DENY,
        DENY,
        DENY,
        "Returns are limited to delivered orders in the source policy.",
    ),
    TauRetailCase(
        "unauthenticated",
        "authentication:missing",
        "authentication",
        (
            ("action", "RETURN_DELIVERED_ORDER"),
            ("order_status", "DELIVERED"),
            ("refund_destination", "ORIGINAL_PAYMENT"),
            ("authenticated", "NO"),
            ("explicit_confirmation", "YES"),
        ),
        DENY,
        DENY,
        DENY,
        "The source policy requires user authentication at the beginning of the conversation.",
    ),
    TauRetailCase(
        "missing_confirmation",
        "confirmation:missing",
        "confirmation",
        (
            ("action", "CANCEL_PENDING_ORDER"),
            ("order_status", "PENDING"),
            ("reason", "NO_LONGER_NEEDED"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "NO"),
        ),
        DENY,
        DENY,
        DENY,
        "Database updates require explicit user confirmation in the source policy.",
    ),
    TauRetailCase(
        "gift_card_insufficient",
        "gift_card_balance:insufficient",
        "payment_constraint",
        (
            ("action", "MODIFY_PENDING_ORDER_PAYMENT"),
            ("order_status", "PENDING"),
            ("new_payment_method", "GIFT_CARD"),
            ("gift_card_balance_sufficient", "NO"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
        ),
        DENY,
        DENY,
        DENY,
        "A gift card must cover the required amount in the source policy.",
    ),
    TauRetailCase(
        "exchange_same_product",
        "exchange_compatibility:same_product",
        "exchange_compatibility",
        (
            ("action", "EXCHANGE_DELIVERED_ORDER"),
            ("order_status", "DELIVERED"),
            ("same_product", "YES"),
            ("different_option", "YES"),
            ("new_item_available", "YES"),
            ("price_difference_covered", "YES"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
        ),
        ALLOW,
        ALLOW,
        ALLOW,
        "A delivered item may be exchanged for an available option of the same product.",
    ),
    TauRetailCase(
        "cancel_not_needed",
        "cancel_reason:no_longer_needed",
        "cancel_reason",
        (
            ("action", "CANCEL_PENDING_ORDER"),
            ("order_status", "PENDING"),
            ("reason", "NO_LONGER_NEEDED"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
        ),
        ALLOW,
        ALLOW,
        ALLOW,
        "No longer needed is an accepted cancellation reason in the source policy.",
    ),
    TauRetailCase(
        "exchange_different_product",
        "exchange_compatibility:different_product_type",
        "exchange_compatibility",
        (
            ("action", "EXCHANGE_DELIVERED_ORDER"),
            ("order_status", "DELIVERED"),
            ("same_product", "NO"),
            ("same_category", "NO"),
            ("new_item_available", "YES"),
            ("authenticated", "YES"),
            ("explicit_confirmation", "YES"),
        ),
        DENY,
        DENY,
        DENY,
        "The source policy forbids changing product types during an exchange.",
    ),
)

_REQUIRED_POLICY_SNIPPETS = (
    "At the beginning of the conversation, you have to authenticate the user identity",
    "obtain explicit user confirmation (yes) to proceed",
    "An order can only be cancelled if its status is 'pending'",
    "either 'no longer needed' or 'ordered by mistake'",
    "The refund must either go to the original payment method, or an existing gift card",
    "same product but of different product option",
    "If the user provides a gift card, it must have enough balance",
)
_REQUIRED_TASK_ACTIONS = frozenset(
    {
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
        "cancel_pending_order",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
        "modify_pending_order_payment",
    }
)


class TauRetailPolicyShiftBenchmark(BenchmarkAdapter):
    """Public-source-derived multi-rule policy drift stream.

    The upstream tau3 retail policy and tasks are integrity checked. Decision
    cases are generated locally from their action vocabulary and policy clauses,
    with version overlays explicitly identified as EvoShift-derived metadata.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        seed: int = 42,
        phase_size: int = 24,
        feedback_noise_rate: float = 0.10,
        feedback_attack_rate: float = 0.0,
        feedback_shared_source: bool = True,
        feedback_shared_source_name: str = "customer_support_portal",
        feedback_attack_burst_length: int = 2,
        policy_schedule: Sequence[str] | None = None,
        shuffle_within_phase: bool = False,
        limit: int = 0,
        timeout_seconds: float = 30.0,
        expected_manifest: Mapping[str, Mapping[str, Any]] | None = None,
        expected_task_count: int = 114,
        required_policy_snippets: Sequence[str] = _REQUIRED_POLICY_SNIPPETS,
        required_task_actions: Sequence[str] = tuple(_REQUIRED_TASK_ACTIONS),
    ) -> None:
        if phase_size < 8:
            raise DatasetError("tau retail policy shift phase_size must be at least 8")
        if limit < 0:
            raise DatasetError("benchmark limit cannot be negative")
        if timeout_seconds <= 0:
            raise DatasetError("tau retail download timeout must be positive")
        if expected_task_count < 1:
            raise DatasetError("tau retail expected task count must be positive")
        for name, value in (
            ("feedback_noise_rate", feedback_noise_rate),
            ("feedback_attack_rate", feedback_attack_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise DatasetError(f"{name} must be between 0 and 1")
        if feedback_attack_burst_length < 0:
            raise DatasetError("feedback_attack_burst_length cannot be negative")
        if feedback_shared_source and not feedback_shared_source_name.strip():
            raise DatasetError("feedback_shared_source_name must not be empty")
        schedule = tuple(policy_schedule or DEFAULT_TAU_POLICY_SCHEDULE)
        if not schedule:
            raise DatasetError("tau retail policy schedule must not be empty")
        invalid_versions = sorted(set(schedule) - {"v1", "v2", "v3"})
        if invalid_versions:
            raise DatasetError(
                "tau retail policy schedule supports only v1, v2, and v3: "
                + ", ".join(invalid_versions)
            )
        self.cache_dir = Path(cache_dir)
        self.seed = seed
        self.phase_size = phase_size
        self.feedback_noise_rate = feedback_noise_rate
        self.feedback_attack_rate = feedback_attack_rate
        self.feedback_shared_source = feedback_shared_source
        self.feedback_shared_source_name = feedback_shared_source_name.strip()
        self.feedback_attack_burst_length = feedback_attack_burst_length
        self.policy_schedule = schedule
        self.shuffle_within_phase = shuffle_within_phase
        self.limit = limit
        self.timeout_seconds = timeout_seconds
        self.source_manifest = self._normalize_source_manifest(
            expected_manifest or TAU3_RETAIL_SOURCE_MANIFEST
        )
        self.expected_task_count = expected_task_count
        self.required_policy_snippets = tuple(required_policy_snippets)
        self.required_task_actions = frozenset(required_task_actions)

    @property
    def dataset_dir(self) -> Path:
        return self.cache_dir / TAU3_RETAIL_REVISION

    @property
    def manifest_path(self) -> Path:
        return self.dataset_dir / "manifest.json"

    def load(self) -> list[BenchmarkSample]:
        self.download()
        stream: list[BenchmarkSample] = []
        for phase_index, version in enumerate(self.policy_schedule):
            cases = self._phase_cases(phase_index)
            previous_version = self.policy_schedule[phase_index - 1] if phase_index else None
            future_versions = self.policy_schedule[phase_index + 1 :]
            context_positions: Counter[str] = Counter()
            samples: list[BenchmarkSample] = []
            for position, (case, repetition) in enumerate(cases):
                context_position = context_positions[case.context]
                context_positions[case.context] += 1
                samples.append(
                    self._make_sample(
                        case,
                        repetition=repetition,
                        version=version,
                        phase_index=phase_index,
                        position=position,
                        context_position=context_position,
                        previous_version=previous_version,
                        future_versions=future_versions,
                    )
                )
            if self.shuffle_within_phase:
                random.Random(stable_seed(self.seed, f"tau3:phase:{phase_index}")).shuffle(samples)
                samples = [
                    sample.model_copy(
                        update={
                            "metadata": {
                                **sample.metadata,
                                "position_in_phase": position,
                                "is_shift_boundary": phase_index > 0 and position == 0,
                            }
                        }
                    )
                    for position, sample in enumerate(samples)
                ]
            stream.extend(samples)
        return stream[: self.limit] if self.limit else stream

    def download(self) -> dict[str, Path]:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._read_manifest()
        self._validate_manifest(manifest)
        paths = {
            name: self.dataset_dir / str(entry["path"])
            for name, entry in self.source_manifest.items()
        }
        if manifest is None and any(path.exists() for path in paths.values()):
            raise DatasetError(
                f"tau3 retail cache contains data without {self.manifest_path}; "
                "remove the incomplete cache and retry"
            )
        pending: dict[str, bytes] = {}
        for name, path in paths.items():
            if path.exists():
                self._verify_file(path, name)
                if manifest is None or name not in manifest.get("files", {}):
                    raise DatasetError(f"tau3 retail manifest has no entry for cached file: {path}")
                self._verify_manifest_entry(name, manifest["files"][name])
                continue
            data = self._download_bytes(self._source_url(name))
            self._verify_payload(name, data)
            pending[name] = data
        for name, data in pending.items():
            self._atomic_write(paths[name], data)
        self._validate_source_semantics(paths)
        self._atomic_write(
            self.manifest_path,
            (json.dumps(self._manifest_payload(), indent=2, sort_keys=True) + "\n").encode(),
        )
        return paths

    def verify_cache(self) -> dict[str, Any]:
        manifest = self._read_manifest()
        if manifest is None:
            raise DatasetError(f"tau3 retail manifest does not exist: {self.manifest_path}")
        self._validate_manifest(manifest)
        paths: dict[str, Path] = {}
        for name, entry in self.source_manifest.items():
            path = self.dataset_dir / str(entry["path"])
            if not path.exists():
                raise DatasetError(f"tau3 retail cache file is missing: {path}")
            self._verify_file(path, name)
            self._verify_manifest_entry(name, manifest.get("files", {}).get(name))
            paths[name] = path
        self._validate_source_semantics(paths)
        return manifest

    def _phase_cases(self, phase_index: int) -> list[tuple[TauRetailCase, int]]:
        indexed = [(case, repetition) for repetition in range(2) for case in _CASES]
        rotation = stable_seed(self.seed, f"tau3:case-rotation:{phase_index}") % len(_CASES)
        ordered = indexed[rotation:] + indexed[:rotation]
        if len(ordered) < self.phase_size:
            repetitions = (self.phase_size + len(_CASES) - 1) // len(_CASES)
            ordered = [(case, repetition) for repetition in range(repetitions) for case in _CASES]
            ordered = ordered[rotation:] + ordered[:rotation]
        return ordered[: self.phase_size]

    def _make_sample(
        self,
        case: TauRetailCase,
        *,
        repetition: int,
        version: str,
        phase_index: int,
        position: int,
        context_position: int,
        previous_version: str | None,
        future_versions: Sequence[str],
    ) -> BenchmarkSample:
        oracle = case.label(version)
        previous_oracle = case.label(previous_version) if previous_version else None
        transition = previous_oracle is not None and previous_oracle != oracle
        future_change = not transition and any(
            case.label(item) != oracle for item in future_versions
        )
        protected = not transition and not future_change
        feedback, feedback_kind, feedback_source, attack_goal = self._observed_feedback(
            oracle,
            case=case,
            version=version,
            phase_index=phase_index,
            context_position=context_position,
            transition=transition,
            future_change=future_change,
        )
        valid_tag = self.policy_context_tag(case.context, oracle)
        stale_tag = self.policy_context_tag(case.context, self._flip(oracle))
        attributes = "; ".join(f"{key}={value}" for key, value in case.attributes)
        return BenchmarkSample(
            sample_id=f"tau3_retail_policy_shift:{phase_index}:{position:04d}",
            prompt=f"Retail policy case: {attributes}. Return ALLOW or DENY only.",
            reference=oracle,
            domain="customer_support/retail_policy",
            phase=f"phase_{phase_index}",
            evaluator="exact_match",
            metadata={
                "synthetic": False,
                "derived": True,
                "benchmark": "policy_shift",
                "benchmark_variant": "tau3_retail_policy_shift",
                "official_tau3_benchmark": False,
                "source_repository": TAU3_RETAIL_REPOSITORY,
                "source_revision": TAU3_RETAIL_REVISION,
                "source_commit": TAU3_RETAIL_COMMIT,
                "source_policy_sha256": self.source_manifest["policy"]["sha256"],
                "source_tasks_sha256": self.source_manifest["tasks"]["sha256"],
                "source_rule": case.source_rule,
                "overlay_version": version,
                "seed": self.seed,
                "phase_index": phase_index,
                "position_in_phase": position,
                "case_repetition": repetition,
                "is_shift_boundary": phase_index > 0 and position == 0,
                "policy_version": version,
                "previous_policy_version": previous_version or "",
                "is_policy_reversion": version in self.policy_schedule[:phase_index],
                "policy_schedule": list(self.policy_schedule),
                "rule_family": case.rule_family,
                "case_context": case.context,
                "policy_changed_case": transition,
                "transition_case": transition,
                "protected": protected,
                "future_change_case": future_change,
                "feedback_context": f"tau3_retail:{case.context}",
                "feedback_reference": feedback,
                "feedback_kind": feedback_kind,
                "feedback_source": feedback_source,
                "feedback_attack_goal": attack_goal,
                "feedback_corrupted": feedback != oracle,
                "valid_memory_tags": [valid_tag],
                "stale_memory_tags": [stale_tag],
                "learner_visible_memory_tag": self.policy_context_tag(case.context, feedback),
            },
        )

    def _observed_feedback(
        self,
        oracle: str,
        *,
        case: TauRetailCase,
        version: str,
        phase_index: int,
        context_position: int,
        transition: bool,
        future_change: bool,
    ) -> tuple[str, str, str, str]:
        source = self.feedback_shared_source_name if self.feedback_shared_source else ""
        if (
            phase_index == 0
            and future_change
            and context_position < self.feedback_attack_burst_length
        ):
            return (
                self._flip(oracle),
                "attack",
                source or "untrusted_policy_message",
                "premature_update",
            )
        rng = random.Random(
            stable_seed(
                self.seed,
                f"tau3:feedback:{phase_index}:{case.case_id}:{context_position}",
            )
        )
        if transition and version != "v1" and rng.random() < self.feedback_attack_rate:
            return (
                self._flip(oracle),
                "attack",
                source or "untrusted_policy_message",
                "rollback_to_old_rule",
            )
        if rng.random() < self.feedback_noise_rate:
            return self._flip(oracle), "noise", source or "execution_feedback", "incidental"
        return oracle, "clean", source or "verified_policy_engine", ""

    @staticmethod
    def policy_context_tag(context: str, label: str) -> str:
        normalized = "_".join(context.casefold().replace(":", "_").split())
        return f"tau3_policy:{normalized}:{label.casefold()}"[:80]

    @staticmethod
    def _flip(label: str) -> str:
        return DENY if label == ALLOW else ALLOW

    def _source_url(self, name: str) -> str:
        upstream = self.source_manifest[name]["upstream_path"]
        return f"{_RAW_ROOT}/{TAU3_RETAIL_COMMIT}/{upstream}"

    def _download_bytes(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "EvoShift/0.1"})
        last_error: BaseException | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    data = cast(bytes, response.read(_MAX_DOWNLOAD_BYTES + 1))
                break
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {408, 409, 425, 429} and exc.code < 500:
                    raise DatasetError(
                        f"cannot download tau3 retail source from {url}: HTTP {exc.code}"
                    ) from exc
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
            if attempt < 3:
                time.sleep(0.25 * (2**attempt))
        else:
            raise DatasetError(
                f"cannot download tau3 retail source from {url}: {last_error}"
            ) from last_error
        if len(data) > _MAX_DOWNLOAD_BYTES:
            raise DatasetError(f"tau3 retail source exceeds {_MAX_DOWNLOAD_BYTES} bytes: {url}")
        return data

    def _verify_payload(self, name: str, data: bytes) -> None:
        expected = self.source_manifest[name]
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected["sha256"]:
            raise DatasetError(
                f"SHA-256 mismatch for tau3 retail {name}: "
                f"expected {expected['sha256']}, got {actual_hash}"
            )
        if len(data) != expected["bytes"]:
            raise DatasetError(
                f"size mismatch for tau3 retail {name}: "
                f"expected {expected['bytes']}, got {len(data)}"
            )

    def _verify_file(self, path: Path, name: str) -> None:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise DatasetError(f"cannot read cached tau3 retail file {path}: {exc}") from exc
        self._verify_payload(name, data)

    def _validate_source_semantics(self, paths: Mapping[str, Path]) -> None:
        try:
            policy = paths["policy"].read_text(encoding="utf-8")
            tasks = json.loads(paths["tasks"].read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DatasetError(f"cannot parse tau3 retail source: {exc}") from exc
        missing = [snippet for snippet in self.required_policy_snippets if snippet not in policy]
        if missing:
            raise DatasetError("tau3 retail policy is missing required pinned clauses")
        if not isinstance(tasks, list) or len(tasks) != self.expected_task_count:
            raise DatasetError(
                f"tau3 retail tasks must contain the pinned {self.expected_task_count}-task list"
            )
        actions = {
            str(action.get("name"))
            for task in tasks
            if isinstance(task, Mapping)
            for action in task.get("evaluation_criteria", {}).get("actions", [])
            if isinstance(action, Mapping)
        }
        missing_actions = sorted(self.required_task_actions - actions)
        if missing_actions:
            raise DatasetError(
                "tau3 retail tasks are missing required actions: " + ", ".join(missing_actions)
            )

    def _read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DatasetError(f"cannot parse tau3 retail manifest: {exc}") from exc
        if not isinstance(payload, dict):
            raise DatasetError("tau3 retail manifest must contain an object")
        return payload

    @staticmethod
    def _validate_manifest(manifest: dict[str, Any] | None) -> None:
        if manifest is None:
            return
        if manifest.get("schema_version") != 1:
            raise DatasetError("unsupported tau3 retail manifest schema")
        if manifest.get("repository") != TAU3_RETAIL_REPOSITORY:
            raise DatasetError("tau3 retail manifest repository mismatch")
        if manifest.get("revision") != TAU3_RETAIL_REVISION:
            raise DatasetError("tau3 retail manifest revision mismatch")
        if manifest.get("commit") != TAU3_RETAIL_COMMIT:
            raise DatasetError("tau3 retail manifest commit mismatch")
        if not isinstance(manifest.get("files"), dict):
            raise DatasetError("tau3 retail manifest files must be an object")

    def _verify_manifest_entry(self, name: str, entry: Any) -> None:
        if not isinstance(entry, Mapping):
            raise DatasetError(f"invalid tau3 retail manifest entry for {name}")
        expected = self.source_manifest[name]
        if (
            entry.get("path") != expected["path"]
            or entry.get("sha256") != expected["sha256"]
            or entry.get("bytes") != expected["bytes"]
            or entry.get("url") != self._source_url(name)
        ):
            raise DatasetError(f"tau3 retail manifest integrity mismatch for {name}")

    def _manifest_payload(self) -> dict[str, Any]:
        files = {
            name: {
                "path": entry["path"],
                "upstream_path": entry["upstream_path"],
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
                "url": self._source_url(name),
            }
            for name, entry in sorted(self.source_manifest.items())
        }
        return {
            "schema_version": 1,
            "dataset": "tau3-bench retail policy and tasks",
            "derived_benchmark": "Tau3Retail-PolicyDrift",
            "official_tau3_benchmark": False,
            "repository": TAU3_RETAIL_REPOSITORY,
            "revision": TAU3_RETAIL_REVISION,
            "commit": TAU3_RETAIL_COMMIT,
            "license": "MIT",
            "files": files,
        }

    @staticmethod
    def _normalize_source_manifest(
        manifest: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if set(manifest) != {"policy", "tasks"}:
            raise DatasetError("tau3 retail source manifest requires policy and tasks entries")
        normalized: dict[str, dict[str, Any]] = {}
        for name, entry in manifest.items():
            path = entry.get("path")
            upstream_path = entry.get("upstream_path")
            sha256 = entry.get("sha256")
            size = entry.get("bytes")
            if not isinstance(path, str) or Path(path).name != path:
                raise DatasetError(f"invalid tau3 retail cache path for {name}")
            if not isinstance(upstream_path, str) or not upstream_path.strip():
                raise DatasetError(f"invalid tau3 retail upstream path for {name}")
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise DatasetError(f"invalid tau3 retail SHA-256 for {name}")
            if not isinstance(size, int) or size < 1:
                raise DatasetError(f"invalid tau3 retail byte size for {name}")
            normalized[name] = {
                "path": path,
                "upstream_path": upstream_path,
                "sha256": sha256,
                "bytes": size,
            }
        return normalized

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as temporary:
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.replace(temporary_name, path)
        except OSError as exc:
            raise DatasetError(
                f"cannot atomically write tau3 retail cache file {path}: {exc}"
            ) from exc
        finally:
            if temporary_name is not None:
                temporary_path = Path(temporary_name)
                if temporary_path.exists():
                    with contextlib.suppress(OSError):
                        temporary_path.unlink()


Tau3RetailPolicyShiftBenchmarkAdapter = TauRetailPolicyShiftBenchmark


__all__ = [
    "DEFAULT_TAU_POLICY_SCHEDULE",
    "TAU3_RETAIL_COMMIT",
    "TAU3_RETAIL_REPOSITORY",
    "TAU3_RETAIL_REVISION",
    "TAU3_RETAIL_SOURCE_MANIFEST",
    "Tau3RetailPolicyShiftBenchmarkAdapter",
    "TauRetailPolicyShiftBenchmark",
]
