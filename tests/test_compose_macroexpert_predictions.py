from scripts.compose_macroexpert_predictions import compose


def _row(image: int, label: int, score: float = 0.5) -> dict:
    return {"image_id": image, "category_id": label, "score": score,
            "bbox": [0, 0, 1, 1]}


def test_compose_routes_only_selected_images_and_preserves_aircraft() -> None:
    primary = [_row(1, 0), _row(1, 7), _row(2, 0), _row(2, 8)]
    specialist = [_row(1, 24), _row(1, 2)]

    rows = compose(primary, specialist, {1})

    assert {(r["image_id"], r["category_id"]) for r in rows} == {
        (1, 7), (1, 24), (1, 2), (2, 0), (2, 8)
    }


def test_compose_rejects_specialist_aircraft_label() -> None:
    try:
        compose([], [_row(1, 9)], {1})
    except ValueError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("expected forbidden specialist label to fail")
