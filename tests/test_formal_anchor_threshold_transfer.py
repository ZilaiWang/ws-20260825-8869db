from scripts.analyze_formal_anchor_threshold_transfer import _project


def test_project_adds_paired_delta_and_clips_rates() -> None:
    anchor = {name: {"recall": 0.9, "fdr": 0.3} for name in ("ship", "aircraft", "vehicle")}
    baseline = {name: {"recall": 0.8, "fdr": 0.4} for name in anchor}
    candidate = {name: {"recall": 0.7, "fdr": 0.1} for name in anchor}
    projected = _project(anchor, baseline, candidate)
    assert projected["ship"] == {"recall": 0.8, "fdr": 0.0}
