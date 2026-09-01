from rsdet.evaluation.background_stress import evaluate_background_stress
from rsdet.evaluation.error_route import decide_ship_training_direction
from rsdet.evaluation.module_admission import ModuleAdmission, compose_final_recipe


def test_ship_route_selects_objectness_for_background_miss_dominance() -> None:
    result = decide_ship_training_direction(
        {"FP_BG": 576, "FP_CLS": 83}, {"FN_MISS": 295, "FN_CLS": 83}
    )
    assert result.admitted
    assert result.selected_direction == "objectness_quality"


def test_background_metric_scales_to_100mp() -> None:
    manifest = [{"image_id": 1, "width": 1000, "height": 1000}]
    result = evaluate_background_stress(
        manifest,
        {1: [{"category_id": 0}, {"category_id": 24}]},
        category_mapping={0: "ship", 24: "vehicle"},
    )
    assert result.false_positives_per_100mp == 200.0


def test_only_independently_admitted_modules_are_composed() -> None:
    good = ModuleAdmission(
        "vehicle_rescue", "platform_observed_20260831", True, True,
        0.006, -0.01, 1.0, 0.0, 0.3, 0.0, True
    )
    bad = ModuleAdmission(
        "ship_fine_tail", "platform_observed_20260831", True, True,
        0.0, -0.01, 0.1, 0.0, 0.0, 0.0, True
    )
    recipe = compose_final_recipe({"name": "baseline"}, [good, bad])
    assert recipe["accepted_modules"] == ["vehicle_rescue"]
    assert "ship_fine_tail" in recipe["rejected_modules"]
