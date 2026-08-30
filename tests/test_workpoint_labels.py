from rsdet.analysis.workpoint_labels import build_workpoint_labels
from rsdet.evaluation.official_frontier import official_fixed_risk_frontier


def test_inactive_tail_fp_is_not_an_active_fp_label() -> None:
    result = official_fixed_risk_frontier(
        gt_boxes={1: [{"bbox_xyxy": [0, 0, 10, 10], "category_id": 0}]},
        predictions=[
            {
                "image_id": 1,
                "category_id": 0,
                "score": 0.9,
                "bbox_xyxy": [0, 0, 10, 10],
                "source_prediction_index": 1,
            },
            {
                "image_id": 1,
                "category_id": 0,
                "score": 0.1,
                "bbox_xyxy": [20, 20, 30, 30],
                "source_prediction_index": 2,
            },
        ],
        category_mapping={0: "aircraft"},
        iou_thresholds={"aircraft": 0.5},
        fdr_levels=(0.12,),
    )
    labels = build_workpoint_labels(result, target_fdr=0.12)
    assert labels[1].role == "protected_tp"
    assert labels[2].role == "inactive_tail"
    assert not labels[2].is_selected
