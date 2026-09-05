import pytest

from scripts.run_p40_weak_rfs import validate_inference_geometry


def test_current_inference_geometry_receipt_passes():
    validate_inference_geometry({
        "imgsz": 1280,
        "pipeline": {"tile_size": 1024, "overlap": 256},
    })


def test_legacy_geometry_requires_explicit_opt_in():
    legacy = {"pipeline": {"tile_size": 1024, "overlap": 256}}
    with pytest.raises(ValueError, match="no imgsz"):
        validate_inference_geometry(legacy)
    validate_inference_geometry(legacy, allow_legacy_imgsz=True)


@pytest.mark.parametrize(
    "summary",
    [
        {"imgsz": 1024, "pipeline": {"tile_size": 1024, "overlap": 256}},
        {"imgsz": 1280, "pipeline": {"tile_size": 512, "overlap": 256}},
        {"imgsz": 1280, "pipeline": {"tile_size": 1024, "overlap": 128}},
    ],
)
def test_different_geometry_is_rejected(summary):
    with pytest.raises(ValueError):
        validate_inference_geometry(summary)
