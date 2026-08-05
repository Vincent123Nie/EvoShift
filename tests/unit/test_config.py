import pytest

from evoshift.config import EvolutionConfig


def test_retirement_context_bound_revival_requires_dormant_revival() -> None:
    with pytest.raises(
        ValueError,
        match="retirement-context-bound revival requires dormant memory revival",
    ):
        EvolutionConfig(dormant_revival_retirement_context_enabled=True)


def test_retirement_context_bound_revival_accepts_complete_dependency_chain() -> None:
    config = EvolutionConfig(
        dynamic_feedback_trust_enabled=True,
        dormant_revival_enabled=True,
        dormant_revival_retirement_context_enabled=True,
    )

    assert config.dormant_revival_retirement_context_enabled
