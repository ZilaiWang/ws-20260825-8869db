"""配置加载测试。"""

import tempfile
from pathlib import Path

import pytest

from rsdet.utils.config import load_config, merge_configs


class TestLoadConfig:
    def test_load_valid_yaml(self):
        """YAML 可正常加载。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("key: value\nlist:\n  - a\n  - b\n")
            tmp_path = f.name

        try:
            config = load_config(tmp_path)
            assert config["key"] == "value"
            assert config["list"] == ["a", "b"]
        finally:
            Path(tmp_path).unlink()

    def test_load_empty_yaml(self):
        """空 YAML 返回空字典。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            tmp_path = f.name

        try:
            config = load_config(tmp_path)
            assert config == {}
        finally:
            Path(tmp_path).unlink()

    def test_file_not_found(self):
        """缺少文件时给出清楚错误。"""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_top_level_list_is_rejected(self):
        """配置顶层必须是映射。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as file:
            file.write("- invalid\n")
            tmp_path = file.name
        try:
            with pytest.raises(TypeError, match="顶层必须是映射"):
                load_config(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_project_yaml_loadable(self):
        """configs/project.yaml 可加载。"""
        project_cfg = Path(__file__).parent.parent / "configs" / "project.yaml"
        assert project_cfg.exists(), "project.yaml 不存在"
        config = load_config(project_cfg)
        assert config["project"]["name_cn"] != ""
        assert config["project"]["package_name"] == "rsdet"
        assert config["protocol_versions"] == {
            "contract_version": "contract_v1",
            "eval_version": "official_eval_v1",
        }
        assert len(config["task"]["class_names"]) == 3
        assert "ship" in config["task"]["class_names"]


class TestMergeConfigs:
    def test_override_value(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = merge_configs(base, override)
        assert result["a"] == 1
        assert result["b"] == 99

    def test_deep_merge(self):
        """嵌套字典深度合并。"""
        base = {
            "model": {"name": "dummy", "lr": 0.001},
            "data": {"root": "/data"},
        }
        override = {
            "model": {"lr": 0.01},
        }
        result = merge_configs(base, override)
        assert result["model"]["name"] == "dummy"  # 保留
        assert result["model"]["lr"] == 0.01  # 覆盖
        assert result["data"]["root"] == "/data"  # 不受影响

    def test_add_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        result = merge_configs(base, override)
        assert result["a"] == 1
        assert result["b"] == 2
