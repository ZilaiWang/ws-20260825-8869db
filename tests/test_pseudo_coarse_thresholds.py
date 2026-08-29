from scripts.analyze_cv3_pseudo_coarse_thresholds import select_joint_thresholds


def test_joint_thresholds_allocate_risk_across_classes() -> None:
    thresholds = [0.1, 0.5, 0.9]
    counts = {
        "ship": [(8, 8), (7, 1), (2, 0)],
        "aircraft": [(10, 0), (6, 0), (1, 0)],
        "vehicle": [(5, 5), (4, 0), (1, 0)],
    }
    result = select_joint_thresholds(counts, thresholds, target_fdr=0.10)
    assert result["thresholds"] == {
        "ship": 0.5,
        "aircraft": 0.1,
        "vehicle": 0.5,
    }
    assert result["train_tp"] == 21
    assert result["train_fp"] == 1


def test_joint_thresholds_tie_break_prefers_conservative_tuple() -> None:
    thresholds = [0.1, 0.5]
    counts = {
        "ship": [(1, 0), (1, 0)],
        "aircraft": [(1, 0), (1, 0)],
        "vehicle": [(1, 0), (1, 0)],
    }
    result = select_joint_thresholds(counts, thresholds, target_fdr=0.0)
    assert result["thresholds"] == {
        "ship": 0.5,
        "aircraft": 0.5,
        "vehicle": 0.5,
    }


def test_joint_thresholds_rejects_invalid_contract() -> None:
    thresholds = [0.1]
    counts = {"ship": [(1, 0)], "aircraft": [(1, 0)]}
    try:
        select_joint_thresholds(counts, thresholds, target_fdr=0.1)
    except ValueError as error:
        assert "exactly" in str(error)
    else:
        raise AssertionError("invalid class contract must fail")
