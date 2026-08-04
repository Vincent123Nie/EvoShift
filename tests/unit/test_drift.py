from evoshift.config import ShiftConfig
from evoshift.evolution import PageHinkleyShiftDetector


def test_page_hinkley_detects_sustained_loss_shift() -> None:
    detector = PageHinkleyShiftDetector(
        ShiftConfig(min_instances=4, threshold=1.0, delta=0.0, novelty_threshold=1.0)
    )
    reports = []
    for index, reward in enumerate([1, 1, 1, 1, 0, 0, 0, 0]):
        reports.append(detector.update(reward, novelty=0.0, episode_index=index))

    assert any(report.detected for report in reports[4:])
    assert any("page-hinkley" in report.reason for report in reports)


def test_detector_can_use_novelty_signal() -> None:
    detector = PageHinkleyShiftDetector(
        ShiftConfig(min_instances=3, threshold=99.0, novelty_threshold=0.6)
    )
    detector.update(1.0, 0.8, 0)
    detector.update(1.0, 0.8, 1)
    report = detector.update(1.0, 0.8, 2)
    assert report.detected
    assert "novelty" in report.reason


def test_detector_restarts_its_baseline_after_an_alarm() -> None:
    detector = PageHinkleyShiftDetector(
        ShiftConfig(
            min_instances=3,
            threshold=0.5,
            delta=0.0,
            novelty_threshold=1.0,
            cooldown_episodes=0,
        )
    )
    detector.update(1.0, 0.0, 0)
    detector.update(1.0, 0.0, 1)
    report = detector.update(0.0, 0.0, 2)

    assert report.detected
    assert detector.state()["count"] == 0.0
    assert detector.state()["mean_loss"] == 0.0
