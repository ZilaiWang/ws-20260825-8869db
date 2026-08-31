from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.extract_external_archives import extract_archives


def test_extract_external_archives_freezes_sha(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("images/a.txt", "content")
    result = extract_archives([archive], tmp_path / "out")
    assert result["extracted_file_count"] == 1
    assert len(result["archives"][0]["sha256"]) == 64


def test_extract_external_archives_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        extract_archives([archive], tmp_path / "out")


def test_extract_external_archives_rejects_cross_archive_overwrite(tmp_path: Path) -> None:
    archives = []
    for name in ("one.zip", "two.zip"):
        archive = tmp_path / name
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("images/a.txt", name)
        archives.append(archive)
    with pytest.raises(ValueError, match="duplicate archive member"):
        extract_archives(archives, tmp_path / "out")
