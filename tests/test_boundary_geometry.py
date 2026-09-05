from rsdet.contracts import TileRecord
from rsdet.tiling.boundary_geometry import (
    best_geometry,
    build_virtual_tiles,
    edge_aligned_axis_starts,
    guaranteed_full_containment,
    locate_owner_tile,
    tile_owner_lookup,
)


def test_current_10k_grid_has_13_by_13_tiles() -> None:
    starts = edge_aligned_axis_starts(10000, 1024, 768)
    assert len(starts) == 13
    assert starts[0] == 0
    assert starts[-1] == 8976
    assert len(build_virtual_tiles(10000, 10000, 1024, 256)) == 169


def test_overlap_guarantees_small_box_containment() -> None:
    assert guaranteed_full_containment(256, 256, tile_size=1024, overlap=256)
    assert not guaranteed_full_containment(257, 100, tile_size=1024, overlap=256)
    result = best_geometry((730, 700, 830, 800), build_virtual_tiles(10000, 10000, 1024, 256))
    assert result["has_fully_contained_view"] is True


def test_actual_irregular_edge_grid_has_unique_owner() -> None:
    starts = [0, 768, 1536, 1976]
    tiles = [
        TileRecord(index, 3, x, 0, 1024, 1024)
        for index, x in enumerate(starts)
    ]
    x_cores, y_cores, lookup = tile_owner_lookup(
        tiles,
        image_width=3000,
        image_height=1024,
    )
    assert locate_owner_tile(0, 500, x_cores=x_cores, y_cores=y_cores, lookup=lookup) == 0
    assert locate_owner_tile(2999, 500, x_cores=x_cores, y_cores=y_cores, lookup=lookup) == 3


def test_shifted_phase_covers_image_with_padded_tiles() -> None:
    tiles = build_virtual_tiles(
        10000,
        10000,
        1024,
        256,
        phase_x=384,
        phase_y=384,
        padded_phase=True,
    )
    assert min(tile.x_start for tile in tiles) == -384
    assert min(tile.y_start for tile in tiles) == -384
    assert max(tile.x_start + tile.tile_size for tile in tiles) >= 10000
