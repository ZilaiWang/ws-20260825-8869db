"""Unit tests for Y2 P2 formal runtime and scientific decision."""

from __future__ import annotations

from scripts.y2_decide_p2 import build_p2_decision
from scripts.y2_p2_runtime import (
    EXPECTED_ARCHITECTURE,
    EXPECTED_STRIDES,
    audit_p2_architecture,
)


class _Array:
    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(EXPECTED_STRIDES)


class _Parameter:
    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class _Head:
    stride = _Array()
    f = (19, 22, 25, 28)


class _TorchModel:
    model = [object(), _Head()]

    @staticmethod
    def parameters():
        return [_Parameter(5), _Parameter(7)]


class _Model:
    model = _TorchModel()


def test_architecture_audit_requires_four_stride_levels() -> None:
    assert EXPECTED_ARCHITECTURE == "yolo26s-p2.yaml"
    audit = audit_p2_architecture(_Model())
    assert audit["detect_strides"] == [4.0, 8.0, 16.0, 32.0]
    assert audit["detect_sources"] == [19, 22, 25, 28]
    assert audit["parameter_count"] == 12


def _result(
    *,
    overall_recall: float,
    overall_fdr: float,
    macro_recall: float,
    vehicle_recall: float,
    vehicle_fdr: float,
    folds: tuple[float, float, float],
) -> dict:
    merged = {
        "recall": overall_recall,
        "fdr": overall_fdr,
        "official_gate_passed": overall_recall >= 0.85 and overall_fdr <= 0.20,
        "official_ranking": {
            "overall_macro_recall": macro_recall,
            "overall_macro_fdr": 0.22,
            "per_coarse": {
                "vehicle": {
                    "pooled_recall": vehicle_recall,
                    "pooled_fdr": vehicle_fdr,
                }
            },
        },
    }
    return {
        "merged_held_out": {"C0_global": merged},
        "per_fold": [
            {
                "held_out_fold": fold,
                "methods": {
                    "C0_global": {
                        "held_out": {
                            "official_ranking": {
                                "per_coarse": {"vehicle": {"pooled_recall": folds[fold]}}
                            }
                        }
                    }
                },
            }
            for fold in range(3)
        ],
    }


def test_p2_decision_admits_only_formal_stable_vehicle_gain() -> None:
    m1 = _result(
        overall_recall=0.918,
        overall_fdr=0.199,
        macro_recall=0.865,
        vehicle_recall=0.612,
        vehicle_fdr=0.624,
        folds=(0.55, 0.62, 0.66),
    )
    p2 = _result(
        overall_recall=0.916,
        overall_fdr=0.198,
        macro_recall=0.864,
        vehicle_recall=0.642,
        vehicle_fdr=0.61,
        folds=(0.58, 0.65, 0.68),
    )
    metadata = {
        "contract_version": "cv3_oof_v1",
        "status": "complete_downstream_ready",
        "model_key": "P2",
        "image_count": 4481,
        "fold_count": 3,
        "low_score_threshold": 0.001,
    }
    decision = build_p2_decision(
        m1_result=m1,
        p2_result=p2,
        p2_metadata=metadata,
    )
    assert decision["p2_structure_admission"] is True
    assert decision["quality_stage_admission"] is True
