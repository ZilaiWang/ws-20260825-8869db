import numpy as np
from PIL import Image

from rsdet.hera_guard.dual_view import render_seven_channel_view
from rsdet.hera_guard.voi import recenter_windows, select_voi_budget


def test_seven_channel_view_masks_object_and_reflects() -> None:
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(32, dtype=np.uint8)[None, :]
    pixels[12:20, 12:20] = (255, 255, 255)
    view = render_seven_channel_view(
        Image.fromarray(pixels), (-2.0, 10.0, 10.0, 22.0), resolution=32
    )
    assert view.shape == (7, 32, 32)
    assert view.dtype == np.uint8
    assert view[6].max() == 255
    mask = view[6] > 0
    assert not np.array_equal(view[:3, mask], view[3:6, mask])


def test_voi_budget_and_recenter_are_deterministic() -> None:
    rows = [
        {"image_id": 1, "score": 0.50, "bbox": [0, 0, 8, 8]},
        {"image_id": 1, "score": 0.95, "bbox": [40, 40, 30, 30]},
    ]
    selected = select_voi_budget(
        rows, {1: (100, 100)}, budget=1, decision_threshold=0.5
    )
    assert selected[0].candidate_index == 0
    windows = recenter_windows([10, 20, 10, 20])
    assert len(windows) == 5
    assert windows == recenter_windows([10, 20, 10, 20])
