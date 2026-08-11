"""P0-2 crop manifest 几何和防泄漏规则单元测试。"""

from __future__ import annotations

import math
from pathlib import Path

from rsdet.analysis.crop_manifest import (
    CropPolicy,
    ImageSource,
    ObjectAnnotation,
    build_leakage_safe_assignments,
    compute_crop_geometry,
    load_crop_manifest_config,
    parse_crop_policies,
)


def _source(
    uid: str,
    *,
    group: str = "group_1",
    split: str = "train",
    fold: int = 0,
    width: float = 200,
    height: float = 160,
) -> ImageSource:
    return ImageSource(
        image_uid=uid,
        relative_path=f"images/{split}/{uid}.png",
        width=width,
        height=height,
        sha256="0" * 64,
        estimated_group_uid=group,
        group_basis="test",
        group_confidence="high",
        original_main_split=split,
        original_fold=fold,
        difficult_quality_subset=False,
        edge_target_subset=False,
        pure_background_subset=False,
        speed_proxy_subset=False,
        domain_cluster=0,
    )


def _annotation() -> ObjectAnnotation:
    return ObjectAnnotation(
        annotation_uid="ann_1",
        image_uid="img_1",
        category_id=0,
        category_name="HM",
        macro_category="ship",
        x0=50,
        y0=60,
        x1=150,
        y1=110,
        width=100,
        height=50,
        issue_flags="",
    )


def _policies() -> dict[str, CropPolicy]:
    config_path = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "analysis"
        / "exploratory_crop_manifest.yaml"
    )
    return {
        policy.id: policy for policy in parse_crop_policies(load_crop_manifest_config(config_path))
    }


def test_tight_and_context_windows_keep_exact_square_contract() -> None:
    policies = _policies()
    source = _source("img_1")
    annotation = _annotation()
    tight = compute_crop_geometry(annotation, source, policies["tight"], "v1")
    context = compute_crop_geometry(annotation, source, policies["context_1p25"], "v1")

    assert tight["crop_side_px"] == 100
    assert (tight["crop_x0"], tight["crop_y0"], tight["crop_x1"], tight["crop_y1"]) == (
        50,
        35,
        150,
        135,
    )
    assert tight["gt_coverage_fraction"] == 1
    assert context["crop_side_px"] == 125
    assert (context["crop_x0"], context["crop_y0"]) == (37.5, 22.5)
    assert context["gt_coverage_fraction"] == 1


def test_jitter_is_deterministic_bounded_and_versioned() -> None:
    policy = _policies()["jitter_light"]
    source = _source("img_1")
    annotation = _annotation()
    first = compute_crop_geometry(annotation, source, policy, "manifest_v1")
    second = compute_crop_geometry(annotation, source, policy, "manifest_v1")
    changed_version = compute_crop_geometry(annotation, source, policy, "manifest_v2")

    assert first == second
    assert first["jitter_seed_u64"] != changed_version["jitter_seed_u64"]
    assert -0.08 <= first["jitter_dx_fraction"] <= 0.08
    assert -0.08 <= first["jitter_dy_fraction"] <= 0.08
    assert 0.9 <= first["jitter_width_scale"] <= 1.1
    assert 0.9 <= first["jitter_height_scale"] <= 1.1
    assert math.isclose(first["crop_x1"] - first["crop_x0"], first["crop_side_px"], abs_tol=1e-12)
    assert 0 <= first["gt_coverage_fraction"] <= 1


def test_near_duplicate_union_repairs_split_and_fold_with_minimum_moves() -> None:
    config = load_crop_manifest_config(
        Path(__file__).resolve().parent.parent
        / "configs"
        / "analysis"
        / "exploratory_crop_manifest.yaml"
    )
    sources = {
        "img_a": _source("img_a", group="group_a", split="train", fold=0),
        "img_b": _source("img_b", group="group_b", split="val", fold=1),
        "img_c": _source("img_c", group="group_c", split="train", fold=2),
    }
    assignments, components = build_leakage_safe_assignments(
        sources,
        [("img_a", "img_b", "near duplicate")],
        config,
    )

    assert assignments["img_a"].leakage_group_id == assignments["img_b"].leakage_group_id
    assert assignments["img_a"].main_split == assignments["img_b"].main_split
    assert assignments["img_a"].fold == assignments["img_b"].fold
    assert sum(item.main_split_changed for item in assignments.values()) == 1
    assert sum(item.fold_changed for item in assignments.values()) == 1
    merged = next(row for row in components if row["estimated_group_count"] == 2)
    assert merged["near_duplicate_edge_count"] == 1


def test_padding_and_visible_fraction_are_explicit_for_edge_object() -> None:
    policies = _policies()
    source = _source("img_1", width=100, height=100)
    annotation = ObjectAnnotation(
        annotation_uid="ann_edge",
        image_uid="img_1",
        category_id=0,
        category_name="HM",
        macro_category="ship",
        x0=-10,
        y0=20,
        x1=30,
        y1=60,
        width=40,
        height=40,
        issue_flags="out_of_bounds;touches_edge_within_1px",
    )
    result = compute_crop_geometry(annotation, source, policies["tight"], "v1")

    assert result["padding_fraction"] == 0.25
    assert result["source_gt_visible_fraction"] == 0.75
    assert result["gt_coverage_fraction"] == 1
