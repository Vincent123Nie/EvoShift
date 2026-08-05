from dataclasses import dataclass

from evoshift.schemas import Algorithm


@dataclass(frozen=True)
class AlgorithmBehavior:
    use_memory: bool
    self_refine: bool
    generate_experience: bool
    verify_before_promotion: bool
    evolve_policy: bool
    dynamic_feedback_trust: bool
    future_audit: bool
    active_audit: bool


BEHAVIORS = {
    Algorithm.STATIC: AlgorithmBehavior(
        True, False, False, False, False, False, False, False
    ),
    Algorithm.SELF_REFINE: AlgorithmBehavior(
        False, True, False, False, False, False, False, False
    ),
    Algorithm.REFLEXION: AlgorithmBehavior(
        True, False, True, False, False, False, False, False
    ),
    Algorithm.REPLAY_ONLY: AlgorithmBehavior(
        True, False, True, True, True, False, False, False
    ),
    Algorithm.EVOSHIFT: AlgorithmBehavior(
        True, False, True, True, True, True, True, True
    ),
}


def behavior_for(algorithm: Algorithm) -> AlgorithmBehavior:
    return BEHAVIORS[algorithm]
