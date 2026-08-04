import pytest
from pydantic import ValidationError

from evoshift.errors import EvolutionError
from evoshift.memory import apply_policy_patch
from evoshift.schemas import PolicyGenome, PolicyPatch


def test_policy_patch_is_bounded_and_versioned() -> None:
    base = PolicyGenome(version=2, top_k=4)
    patch = PolicyPatch(
        patch_id="p1",
        base_version=2,
        changes={"top_k": 6},
        hypothesis="A larger candidate set may recover retrieval misses.",
    )
    evolved = apply_policy_patch(base, patch)
    assert evolved.version == 3
    assert evolved.top_k == 6


def test_policy_patch_rejects_code_and_stale_versions() -> None:
    with pytest.raises(ValidationError):
        PolicyPatch(
            patch_id="unsafe",
            base_version=1,
            changes={"python_code": "import os"},
            hypothesis="unsafe",
        )
    with pytest.raises(EvolutionError):
        apply_policy_patch(
            PolicyGenome(version=3),
            PolicyPatch(
                patch_id="stale",
                base_version=2,
                changes={"top_k": 5},
                hypothesis="stale",
            ),
        )
