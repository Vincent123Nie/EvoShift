from evoshift.agents.baselines import behavior_for
from evoshift.schemas import Algorithm


def test_replay_only_is_verified_without_post_promotion_governance() -> None:
    behavior = behavior_for(Algorithm.REPLAY_ONLY)

    assert behavior.use_memory is True
    assert behavior.generate_experience is True
    assert behavior.verify_before_promotion is True
    assert behavior.evolve_policy is True
    assert behavior.dynamic_feedback_trust is False
    assert behavior.future_audit is False
    assert behavior.active_audit is False


def test_full_evoshift_retains_all_governance_layers() -> None:
    behavior = behavior_for(Algorithm.EVOSHIFT)

    assert behavior.verify_before_promotion is True
    assert behavior.dynamic_feedback_trust is True
    assert behavior.future_audit is True
    assert behavior.active_audit is True
