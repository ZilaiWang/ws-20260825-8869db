import copy

import pytest

from rsdet.experiments.paired_trend import VERSION, read, sha, write
from scripts import analyze_paired_development_capacity as capacity
from scripts import run_paired_trend as pipeline


def fixture(tmp_path):
    bundle, evaluation = tmp_path / "bundle", tmp_path / "eval"
    images = [{"id": i, "file_name": f"{i}.png"} for i in range(25)]
    annotations = [{"image_id": i, "category_id": i, "bbox": [0, 0, 10, 10]} for i in range(25)]
    write(bundle / "development_gt.json", {"images": images, "annotations": annotations})
    write(bundle / "manifest.json", {"samples": [
        {"image_id": i, "split": "development", "group_id": f"g{i}"} for i in range(25)
    ]})
    config = {"score_floor": .001}
    write(bundle / "contract.json", {
        "version": VERSION, "project_config_sha256": sha(pipeline.PROJECT),
        "files": {p.name: sha(p) for p in bundle.iterdir()}, "inference": config,
    })
    pred = [{**a, "score": .8} for a in annotations if a["category_id"] not in (0, 1, 2)]
    pred += [{**annotations[0], "score": .2},
             {**annotations[2], "category_id": 0, "score": .8},
             {"image_id": 24, "category_id": 24, "bbox": [100, 100, 10, 10], "score": .1}]
    write(evaluation / "development/predictions_low.json", pred)
    write(evaluation / "development/inference.json", {
        "request": {"config": config, "checkpoint_sha256": "frozen"},
        "image_ids": list(range(25)),
    })
    write(evaluation / "threshold.json", {
        "threshold": .5, "checkpoint_sha256": "frozen",
        "development_gt_sha256": sha(bundle / "development_gt.json"),
        "development_predictions_sha256": sha(evaluation / "development/predictions_low.json"),
    })
    write(evaluation / "development_metrics.json", {})
    write(evaluation / "review.json", {
        "bundle_sha256": sha(bundle / "contract.json"),
        "artifacts": {str(p.relative_to(evaluation)): sha(p) for p in evaluation.rglob("*.json")},
    })
    return bundle, evaluation


def test_development_only_distinguishes_recoverable_and_absent(tmp_path):
    bundle, evaluation = fixture(tmp_path)
    # No confirmation file exists in this fixture; it must not be needed.
    before = sha(evaluation / "threshold.json")
    report = capacity.analyze(bundle, evaluation)
    assert report["selected_fn_recovered_at_floor"] == 1
    assert report["per_fine"]["0"]["fn_recovered_at_decoded_floor"] == 1
    assert report["per_fine"]["1"]["floor_error_roles"]["FN_MISS"] == 1
    assert report["per_fine"]["2"]["floor_error_roles"]["FN_CLS"] == 1
    assert report["floor"]["fp_counts"]["FP_BG"] == 1
    assert not report["confirmation_used"] and not report["thresholds_fitted"]
    assert sha(evaluation / "threshold.json") == before


def test_development_artifact_tamper_rejected(tmp_path):
    bundle, evaluation = fixture(tmp_path)
    path = evaluation / "threshold.json"
    value = copy.deepcopy(read(path))
    value["threshold"] = .01
    path.unlink()
    write(path, value)
    with pytest.raises(ValueError, match="artifact changed"):
        capacity.analyze(bundle, evaluation)
