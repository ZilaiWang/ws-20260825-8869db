from __future__ import annotations

import numpy as np
from PIL import Image

from rsdet.augmentation.jitter_hard_negative import (
    Box,
    JitterPolicy,
    iou,
    sample_hard_negative_boxes,
)
from rsdet.augmentation.prototype_memory import PrototypeMemoryBank
from rsdet.augmentation.scene_scale_materialization import reflect_crop, yolo_rows_for_crop


def test_jitter_hard_negatives_are_deterministic_and_in_band() -> None:
    policy = JitterPolicy(iou_low=0.1, iou_high=0.42, count=3, seed=7)
    target = Box(50, 50, 100, 100)
    first = sample_hard_negative_boxes(
        target, image_width=200, image_height=200, policy=policy, stable_key="x"
    )
    second = sample_hard_negative_boxes(
        target, image_width=200, image_height=200, policy=policy, stable_key="x"
    )
    assert first == second
    assert len(first) == 3
    assert all(policy.iou_low <= iou(target, row) <= policy.iou_high for row in first)


def test_prototype_memory_excludes_source_group() -> None:
    memory = PrototypeMemoryBank(dimension=2)
    memory.update([1, 0], class_id=0, role="positive", source_group="a", scale_bin="small")
    memory.update([0.8, 0.2], class_id=0, role="positive", source_group="b", scale_bin="small")
    memory.update([0, 1], class_id=0, role="hard_negative", source_group="b", scale_bin="small")
    score = memory.boundary_score(
        np.asarray([1, 0]), class_id=0, scale_bin="small", exclude_source_group="a"
    )
    assert score is not None
    assert score > 0.5


def test_scene_scale_materialization_reflects_and_preserves_labels() -> None:
    image = Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))
    crop = Box(-2, -2, 6, 6)
    rendered = reflect_crop(image, crop, output_size=16)
    assert rendered.size == (16, 16)
    rows = yolo_rows_for_crop(
        [Box(0, 0, 4, 4)], [24], [0], crop=crop, output_size=16
    )
    fields = rows[0].split()
    assert fields[0] == "24"
    assert all(0.0 <= float(value) <= 1.0 for value in fields[1:])
