from __future__ import annotations

from evoshift.config import ShiftConfig
from evoshift.schemas import ShiftReport


class PageHinkleyShiftDetector:
    """Online loss drift detector augmented with an input-novelty EWMA."""

    def __init__(self, config: ShiftConfig):
        self.config = config
        self.count = 0
        self.mean_loss = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0
        self.novelty_ewma = 0.0
        self.cooldown_remaining = 0

    def update(
        self, reward: float, novelty: float, episode_index: int, domain: str = "global"
    ) -> ShiftReport:
        if not self.config.enabled:
            return ShiftReport(episode_index=episode_index, domain=domain, reason="disabled")
        bounded_reward = max(0.0, min(1.0, reward))
        bounded_novelty = max(0.0, min(1.0, novelty))
        loss = 1.0 - bounded_reward
        self.count += 1
        self.mean_loss += (loss - self.mean_loss) / self.count
        self.cumulative += loss - self.mean_loss - self.config.delta
        self.minimum = min(self.minimum, self.cumulative)
        statistic = self.cumulative - self.minimum
        alpha = self.config.novelty_ewma_alpha
        self.novelty_ewma = (
            bounded_novelty
            if self.count == 1
            else alpha * bounded_novelty + (1.0 - alpha) * self.novelty_ewma
        )

        eligible = self.count >= self.config.min_instances and self.cooldown_remaining == 0
        performance_shift = eligible and statistic > self.config.threshold
        novelty_shift = eligible and self.novelty_ewma > self.config.novelty_threshold
        detected = performance_shift or novelty_shift
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        reason = ""
        if performance_shift:
            reason = "page-hinkley loss statistic crossed threshold"
        elif novelty_shift:
            reason = "retrieval novelty EWMA crossed threshold"
        elif self.cooldown_remaining > 0:
            reason = "detector cooldown"
        if detected:
            self.cumulative = 0.0
            self.minimum = 0.0
            self.cooldown_remaining = self.config.cooldown_episodes
        return ShiftReport(
            detected=detected,
            detector="page_hinkley+novelty_ewma",
            statistic=statistic,
            threshold=self.config.threshold,
            novelty=self.novelty_ewma,
            domain=domain,
            episode_index=episode_index,
            reason=reason,
        )

    def state(self) -> dict[str, float]:
        return {
            "count": float(self.count),
            "mean_loss": self.mean_loss,
            "cumulative": self.cumulative,
            "minimum": self.minimum,
            "novelty_ewma": self.novelty_ewma,
            "cooldown_remaining": float(self.cooldown_remaining),
        }
