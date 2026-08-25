"""对象几何可见性公式、边界与风险分箱测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rsdet.analysis.object_visibility import (
    ObjectBox,
    Representation,
    Scenario,
    classify_risk,
    compute_scenario_values,
    load_object_boxes,
    parse_analysis_config,
)


def make_box(**overrides) -> ObjectBox:
    values = {
        "annotation_uid": "ann_1",
        "image_uid": "img_1",
        "category_id": 4,
        "category_name": "A1_SU-35",
        "macro_category": "aircraft",
        "image_width": 200.0,
        "image_height": 200.0,
        "x0": 50.0,
        "y0": 75.0,
        "x1": 150.0,
        "y1": 125.0,
        "width": 100.0,
        "height": 50.0,
        "issue_flags": "",
    }
    values.update(overrides)
    return ObjectBox(**values)


def test_tile_1280_to_1024_exact_scale() -> None:
    scenario = Scenario("tile", "tile", tile_size=1280, input_size=1024)
    representations = [Representation("vit_p14", "vit", 14)]
    values = compute_scenario_values([make_box()], scenario, representations)
    assert values.resized_width[0] == pytest.approx(80.0)
    assert values.resized_height[0] == pytest.approx(40.0)
    assert values.grid_short["vit_p14"][0] == pytest.approx(40 / 14)
    assert values.allocation_gain[0] == pytest.approx(0.8)


def test_equal_scale_tile_scenarios_are_numerically_identical() -> None:
    representations = [Representation("s8", "test", 8)]
    first = compute_scenario_values(
        [make_box()], Scenario("640", "tile", tile_size=640, input_size=640), representations
    )
    second = compute_scenario_values(
        [make_box()], Scenario("1280", "tile", tile_size=1280, input_size=1280), representations
    )
    np.testing.assert_array_equal(first.resized_short, second.resized_short)
    np.testing.assert_array_equal(first.grid_short["s8"], second.grid_short["s8"])


def test_crop_formula_and_allocation_gain_do_not_claim_native_detail() -> None:
    scenario = Scenario("crop", "crop", crop_resolution=224, context_scale=1.25)
    representations = [Representation("diffusion_s8", "diffusion", 8)]
    values = compute_scenario_values([make_box()], scenario, representations)
    assert values.crop_nominal_side[0] == pytest.approx(125.0)
    assert values.resized_width[0] == pytest.approx(179.2)
    assert values.resized_height[0] == pytest.approx(89.6)
    assert values.grid_short["diffusion_s8"][0] == pytest.approx(11.2)
    assert values.allocation_gain[0] == pytest.approx(89.6 / 50.0)


def test_crop_scenarios_with_same_resolution_context_ratio_are_identical() -> None:
    representations = [Representation("p14", "test", 14)]
    objects = [make_box(), make_box(width=73.0, height=91.0, x1=123.0, y1=166.0)]
    first = compute_scenario_values(
        objects,
        Scenario("224/1", "crop", crop_resolution=224, context_scale=1.0),
        representations,
    )
    second = compute_scenario_values(
        objects,
        Scenario("336/1.5", "crop", crop_resolution=336, context_scale=1.5),
        representations,
    )
    np.testing.assert_array_equal(first.resized_width, second.resized_width)
    np.testing.assert_array_equal(first.resized_height, second.resized_height)
    np.testing.assert_array_equal(first.grid_short["p14"], second.grid_short["p14"])


def test_crop_padding_and_bbox_visible_fraction_for_out_of_bounds_box() -> None:
    box = make_box(
        x0=-10.0,
        x1=90.0,
        y0=0.0,
        y1=50.0,
        issue_flags="out_of_bounds;touches_edge_within_1px",
    )
    scenario = Scenario("crop", "crop", crop_resolution=224, context_scale=1.0)
    values = compute_scenario_values([box], scenario, [Representation("s8", "test", 8)])
    assert values.bbox_visible_fraction[0] == pytest.approx(0.9)
    # 名义方窗 x 方向 90% 有效、y 方向 75% 有效，padding=1-0.9*0.75。
    assert values.padding_fraction[0] == pytest.approx(0.325)
    assert 0 <= values.padding_fraction[0] <= 1


def test_risk_boundaries_are_mutually_exclusive_and_complete() -> None:
    values = np.asarray([0.0, 1.9999, 2.0, 3.9999, 4.0, 7.9999, 8.0, 100.0])
    bins = classify_risk(values, (2.0, 4.0, 8.0))
    assert bins.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_invalid_or_duplicate_config_is_rejected() -> None:
    config = {
        "tile_scenarios": [
            {"id": "same", "tile_size": 100, "input_size": 100},
            {"id": "same", "tile_size": 200, "input_size": 200},
        ],
        "crop_scenarios": {"resolutions": [], "context_scales": []},
        "representations": [{"id": "s8", "family": "test", "step": 8}],
        "primary_representations": {"diffusion": "s8", "discriminative": "s8"},
        "risk_thresholds": {
            "severe_upper": 2,
            "constrained_upper": 4,
            "moderate_upper": 8,
        },
    }
    with pytest.raises(ValueError, match="ID 重复"):
        parse_analysis_config(config)


def test_bom_input_flagged_row_is_retained_and_mapping_conflict_fails(tmp_path: Path) -> None:
    image_path = tmp_path / "images.csv"
    image_path.write_text("\ufeffimage_uid,width,height\nimg_1,200,200\n", encoding="utf-8")
    bbox_path = tmp_path / "boxes.csv"
    header = (
        "\ufeffannotation_uid,image_uid,category_id,category_name,macro_category,"
        "width_norm,height_norm,x0,y0,x1,y1,width_px,height_px,issue_flags\n"
    )
    row = (
        "ann_1,img_1,4,A1_SU-35,aircraft,0.5,0.25,-10,0,90,50,100,50,"
        "out_of_bounds;touches_edge_within_1px\n"
    )
    bbox_path.write_text(header + row, encoding="utf-8")
    objects = load_object_boxes(bbox_path, image_path)
    assert len(objects) == 1
    assert objects[0].source_edge_risk

    bbox_path.write_text((header + row).replace("aircraft", "ship"), encoding="utf-8")
    with pytest.raises(ValueError, match="类别映射冲突"):
        load_object_boxes(bbox_path, image_path)
