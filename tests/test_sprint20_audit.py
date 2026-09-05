from __future__ import annotations

import subprocess

import pytest

from sprint20.cli import require_base_ancestor


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def test_audited_base_may_have_additive_descendant_commits(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Sprint20 Test")
    (tmp_path / "first").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "first")
    _git(tmp_path, "commit", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "second").write_text("additive\n", encoding="utf-8")
    _git(tmp_path, "add", "second")
    _git(tmp_path, "commit", "-m", "additive")
    assert require_base_ancestor(tmp_path, base) == _git(tmp_path, "rev-parse", "HEAD")


def test_unrelated_commit_fails_closed(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Sprint20 Test")
    (tmp_path / "first").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "first")
    _git(tmp_path, "commit", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "--orphan", "unrelated")
    _git(tmp_path, "rm", "-rf", ".")
    (tmp_path / "other").write_text("other\n", encoding="utf-8")
    _git(tmp_path, "add", "other")
    _git(tmp_path, "commit", "-m", "unrelated")
    with pytest.raises(RuntimeError, match="not an ancestor"):
        require_base_ancestor(tmp_path, base)
