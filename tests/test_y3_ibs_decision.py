"""Decision test for the gated single-factor Y3 branch."""

from __future__ import annotations

from scripts.y3_decide_ibs import build_ibs_decision


def _result(vehicle_fdr: float, fold_fdrs: tuple[float, float, float]) -> dict:
    held = {
        "recall": 0.91,
        "fdr": 0.17,
        "official_gate_passed": True,
        "official_ranking": {
            "overall_macro_recall": 0.86,
            "overall_macro_fdr": 0.20,
            "per_coarse": {"vehicle": {"pooled_recall": 0.64, "pooled_fdr": vehicle_fdr}},
        },
    }
    return {
        "merged_held_out": {"C0_global": held},
        "per_fold": [
            {
                "held_out_fold": fold,
                "methods": {
                    "C0_global": {
                        "held_out": {
                            "official_ranking": {
                                "per_coarse": {"vehicle": {"pooled_fdr": fold_fdrs[fold]}}
                            }
                        }
                    }
                },
            }
            for fold in range(3)
        ],
    }


def test_y3_requires_material_quality_gain_and_two_fold_direction() -> None:
    p2 = _result(0.55, (0.50, 0.55, 0.60))
    y3 = _result(0.52, (0.47, 0.51, 0.61))
    metadata = {
        "contract_version": "cv3_oof_v1",
        "status": "complete_downstream_ready",
        "model_key": "Y3",
        "image_count": 4481,
        "fold_count": 3,
        "low_score_threshold": 0.001,
    }
    decision = build_ibs_decision(
        p2_result=p2,
        y3_result=y3,
        y3_metadata=metadata,
    )
    assert decision["ibs_p2_pair_admission"] is True
    assert decision["checks"]["vehicle_fdr_fold_direction"] is True
