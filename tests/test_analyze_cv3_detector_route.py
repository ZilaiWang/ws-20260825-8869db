from types import SimpleNamespace

from scripts.analyze_cv3_detector_route import select_threshold


def _metric(recall: float, fdr: float) -> SimpleNamespace:
    return SimpleNamespace(
        recall=recall,
        fdr=fdr,
        details={"tp": 1, "fp": 2, "fn": 3},
    )


def test_select_threshold_maximizes_recall_under_fdr_constraint() -> None:
    selected = select_threshold(
        [
            (0.1, _metric(0.9, 0.2)),
            (0.2, _metric(0.8, 0.1)),
            (0.3, _metric(0.7, 0.05)),
        ],
        0.15,
    )
    assert selected["threshold"] == 0.2
    assert selected["train_recall"] == 0.8


def test_select_threshold_fails_closed_when_no_point_is_feasible() -> None:
    selected = select_threshold(
        [(0.1, _metric(0.9, 0.3)), (0.2, _metric(0.7, 0.2))],
        0.15,
    )
    assert selected["threshold"] == 0.2
