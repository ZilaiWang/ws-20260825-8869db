from rsdet.evaluation.macro_risk_v2 import logit_movement_cap, shrink_with_cap


def test_logit_cap_is_tighter_for_tail_classes() -> None:
    assert logit_movement_cap(10, 2) == 0.20
    assert logit_movement_cap(50, 5) == 0.40
    assert logit_movement_cap(100, 6) == 0.80


def test_shrink_cannot_escape_evidence_cap() -> None:
    selected, audit = shrink_with_cap(
        0.99, 0.15, evidence=10, group_count=2, prior_strength=0.0, minimum_evidence=0
    )
    assert audit["selected_logit_delta"] <= 0.20 + 1e-12
    assert 0.15 < selected < 0.99
