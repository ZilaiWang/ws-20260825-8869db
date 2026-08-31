import os
from pathlib import Path

import yaml

from scripts.materialize_external_role_view import materialize_role_view


def test_materialize_external_role_view_isolates_label_cache(tmp_path: Path) -> None:
    source = tmp_path / "source"
    images = source / "images" / "train"
    labels = source / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for name in ("a", "b"):
        (images / f"{name}.png").write_bytes(b"image")
        (labels / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    role_list = source / "train-ext-g.txt"
    role_list.write_text(
        f"{images / 'a.png'}\n{images / 'a.png'}\n{images / 'b.png'}\n",
        encoding="utf-8",
    )
    role_yaml = source / "dataset-ext-g.yaml"
    role_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(source),
                "train": str(role_list),
                "val": str(role_list),
                "names": {0: "aircraft", 1: "ship", 2: "vehicle", 3: "other"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    output = tmp_path / "role-view"
    first = materialize_role_view(role_yaml, output)
    second = materialize_role_view(role_yaml, output)
    assert first["sampled_row_count"] == 3
    assert first["unique_image_count"] == 2
    assert second["label_materialization"]["already_verified"] == 2
    assert first["label_materialization"]["hard_linked"] == 0
    assert os.stat(labels / "a.txt").st_ino != os.stat(output / "labels/train/a.txt").st_ino
    assert (output / "images" / "train").is_symlink()
    assert not (output / "labels" / "train").is_symlink()
    lines = (output / "train-role.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith(str(output / "images" / "train"))
    assert yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))["path"] == str(
        output.resolve()
    )
