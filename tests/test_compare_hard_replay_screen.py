from scripts.compare_hard_replay_screen import compare_frontiers


def _frontier(recall: float, fdr: float, floor: float, vehicle: float = 0.6) -> dict:
    return {
        "candidate_floor": {"recall": floor},
        "frontiers": {
            "0.150": {
                "crossfit": {
                    "recall": recall,
                    "fdr": fdr,
                    "per_coarse": {
                        "ship": {"recall": 0.8},
                        "aircraft": {"recall": 0.95},
                        "vehicle": {"recall": vehicle},
                    },
                }
            }
        },
    }


def test_recall_improvement_passes() -> None:
    result = compare_frontiers(
        _frontier(0.80, 0.15, 0.97),
        _frontier(0.803, 0.151, 0.969),
    )
    assert result["screen_passed"] is True


def test_precision_improvement_passes_without_material_recall_loss() -> None:
    result = compare_frontiers(
        _frontier(0.80, 0.15, 0.97),
        _frontier(0.799, 0.14, 0.969),
    )
    assert result["screen_passed"] is True


def test_candidate_floor_drop_stops() -> None:
    result = compare_frontiers(
        _frontier(0.80, 0.15, 0.97),
        _frontier(0.81, 0.14, 0.96),
    )
    assert result["screen_passed"] is False
