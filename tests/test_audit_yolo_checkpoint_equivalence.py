import numpy as np

from scripts.audit_yolo_checkpoint_equivalence import compare_detections


def test_compare_detections_passes_within_tolerance() -> None:
    reference = np.asarray([[1.0, 0.8, 1.0, 2.0, 3.0, 4.0]])
    candidate = reference.copy()
    candidate[0, 1] += 1e-7
    result = compare_detections(reference, candidate, atol=1e-6)
    assert result["passed"] is True


def test_compare_detections_rejects_class_or_shape_change() -> None:
    reference = np.asarray([[1.0, 0.8, 1.0, 2.0, 3.0, 4.0]])
    changed = reference.copy()
    changed[0, 0] = 2.0
    assert compare_detections(reference, changed, atol=1e-6)["passed"] is False
    assert compare_detections(reference, reference[:0], atol=1e-6)["passed"] is False
