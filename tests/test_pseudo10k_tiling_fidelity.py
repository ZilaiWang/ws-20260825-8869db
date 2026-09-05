from scripts.audit_pseudo10k_tiling_fidelity import (
    _axis_starts,
    _content_rect,
    _crossed_cells,
    _intersection_area,
)


def test_production_tiles_always_cross_multiple_artificial_cells() -> None:
    starts = _axis_starts(10_000, 1024, 256)
    assert len(starts) == 13
    assert all(
        _crossed_cells((x, y, x + 1024, y + 1024)) >= 4
        for y in starts
        for x in starts
    )


def test_content_rect_centers_letterbox_inside_cell() -> None:
    rect, scale, width, height = _content_rect(0, 800, 400)
    assert scale == 1.25
    assert (width, height) == (1000, 500)
    assert rect == (0.0, 250.0, 1000.0, 750.0)


def test_intersection_area_is_half_open_geometry() -> None:
    assert _intersection_area((0, 0, 10, 10), (5, 5, 20, 20)) == 25
    assert _intersection_area((0, 0, 10, 10), (10, 0, 20, 10)) == 0
