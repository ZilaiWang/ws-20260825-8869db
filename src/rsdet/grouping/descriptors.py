"""DINOv2 多层地点描述子和本地 smoke 适配器。"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from rsdet.grouping.contracts import sha256_file

DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def _verify_git_commit(repo: str | Path, expected_commit: str) -> Path:
    path = Path(repo).expanduser().resolve()
    if not (path / ".git").is_dir():
        raise FileNotFoundError(f"DINOv2 repo 不是 Git 仓库: {path}")
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise ValueError(f"DINOv2 commit expected={expected_commit}, actual={actual}")
    return path


def _l2_normalize(array: Any, torch: Any, epsilon: float = 1e-12) -> Any:
    return array / array.norm(dim=-1, keepdim=True).clamp_min(epsilon)


def _signed_gem(patches: Any, p: float, torch: Any) -> Any:
    """适用于 LayerNorm 后有符号 token 的显式 signed GeM。"""

    if p <= 0:
        raise ValueError("GeM p 必须大于 0")
    positive = patches.clamp_min(0).pow(p).mean(dim=1).clamp_min(1e-12).pow(1.0 / p)
    negative = (-patches).clamp_min(0).pow(p).mean(dim=1).clamp_min(1e-12).pow(1.0 / p)
    return positive - negative


def _masked_mean(patches: Any, valid_mask: Any, torch: Any) -> Any:
    if patches.ndim != 3 or valid_mask.ndim != 2 or patches.shape[:2] != valid_mask.shape:
        raise ValueError("patches/valid_mask 形状不匹配")
    weights = valid_mask.to(dtype=patches.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return (patches * weights).sum(dim=1) / counts


def _masked_signed_gem(patches: Any, valid_mask: Any, p: float, torch: Any) -> Any:
    if p <= 0:
        raise ValueError("GeM p 必须大于 0")
    if patches.ndim != 3 or valid_mask.ndim != 2 or patches.shape[:2] != valid_mask.shape:
        raise ValueError("patches/valid_mask 形状不匹配")
    weights = valid_mask.to(dtype=patches.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    positive = (
        ((patches.clamp_min(0).pow(p) * weights).sum(dim=1) / counts).clamp_min(1e-12).pow(1.0 / p)
    )
    negative = (
        (((-patches).clamp_min(0).pow(p) * weights).sum(dim=1) / counts)
        .clamp_min(1e-12)
        .pow(1.0 / p)
    )
    return positive - negative


class MockPlaceEncoder:
    """仅供 CPU smoke；正式服务器门禁禁止使用。"""

    feature_names = ("mock_rgb_stats",)

    def extract(self, images: Sequence[Image.Image]) -> Mapping[str, np.ndarray]:
        rows = []
        for image in images:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            rows.append(np.concatenate([array.mean((0, 1)), array.std((0, 1))]))
        return {"mock_rgb_stats": np.stack(rows).astype(np.float32)}

    def metadata(self) -> dict[str, Any]:
        return {"encoder": "mock", "formal_admission": False}


class MockMaskedPlaceEncoder:
    """Feature-level mask smoke encoder; never permitted for formal extraction."""

    feature_names = ("mock_masked_rgb",)
    patch_token_names = ("mock_patch_tokens",)

    def extract_masked(
        self, images: Sequence[Image.Image], valid_patch_masks: np.ndarray
    ) -> Mapping[str, np.ndarray]:
        masks = np.asarray(valid_patch_masks, dtype=bool)
        if masks.ndim != 2 or masks.shape[0] != len(images):
            raise ValueError("valid_patch_masks 形状不匹配")
        rows = []
        for image, mask in zip(images, masks, strict=True):
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            rows.append(np.concatenate([array.mean((0, 1)), [mask.mean()]]))
        return {"mock_masked_rgb": np.stack(rows).astype(np.float32)}

    def extract_patch_tokens(self, images: Sequence[Image.Image]) -> Mapping[str, np.ndarray]:
        rows = []
        for image in images:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            rows.append(array.reshape(-1, 3))
        return {"mock_patch_tokens": np.stack(rows).astype(np.float32)}

    def extract_masked_with_tokens(
        self, images: Sequence[Image.Image], valid_patch_masks: np.ndarray
    ) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
        return self.extract_masked(images, valid_patch_masks), self.extract_patch_tokens(images)

    def metadata(self) -> dict[str, Any]:
        return {"encoder": "mock_masked", "formal_admission": False}


class DinoV2PlaceEncoder:
    """冻结 DINOv2-B/14 的多层 CLS/mean/signed-GeM 描述子。"""

    def __init__(
        self,
        *,
        repo: str | Path,
        weights: str | Path,
        expected_weight_sha256: str,
        layers: Sequence[int] = (9, 10, 11),
        gem_powers: Sequence[float] = (2.0, 3.0, 4.0),
        device: str = "cuda",
        compute_dtype: str = "float16",
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("DINOv2 place encoder 需要 PyTorch") from error
        normalized_layers = tuple(int(value) for value in layers)
        if not normalized_layers or len(normalized_layers) != len(set(normalized_layers)):
            raise ValueError("layers 必须非空且唯一")
        if any(value < 0 or value >= 12 for value in normalized_layers):
            raise ValueError("DINOv2-B/14 zero-based layer 必须在 [0,11]")
        powers = tuple(float(value) for value in gem_powers)
        if not powers or any(value <= 0 for value in powers):
            raise ValueError("gem_powers 必须是正数")
        if compute_dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("compute_dtype 不支持")
        self.torch = torch
        self.layers = normalized_layers
        self.gem_powers = powers
        self.repo = _verify_git_commit(repo, DINOV2_COMMIT)
        self.weight_path = Path(weights).expanduser().resolve()
        if not self.weight_path.is_file():
            raise FileNotFoundError(self.weight_path)
        actual_sha = sha256_file(self.weight_path)
        if actual_sha != expected_weight_sha256.lower():
            raise ValueError(
                f"DINOv2-B 权重 SHA expected={expected_weight_sha256}, actual={actual_sha}"
            )
        self.weight_sha256 = actual_sha
        self.device = torch.device(device)
        self.dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[compute_dtype]
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        sys.path.insert(0, str(self.repo))
        try:
            backbones = importlib.import_module("dinov2.hub.backbones")
            model = backbones.dinov2_vitb14(pretrained=False)
        finally:
            if sys.path[0] == str(self.repo):
                sys.path.pop(0)
        try:
            state = torch.load(self.weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(self.weight_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[None, :, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[None, :, None, None]
        names = []
        for layer in self.layers:
            names.extend((f"block{layer}_cls", f"block{layer}_mean"))
            names.extend(f"block{layer}_signed_gem_p{power:g}" for power in powers)
        self.feature_names = tuple(names)

    def extract(self, images: Sequence[Image.Image]) -> Mapping[str, np.ndarray]:
        torch = self.torch
        if not images:
            raise ValueError("images 不能为空")
        arrays = []
        for image in images:
            rgb = image.convert("RGB")
            if rgb.size != (518, 518):
                raise ValueError(f"DINOv2 place 输入必须为 518x518，实际 {rgb.size}")
            arrays.append(np.asarray(rgb, dtype=np.float32) / 255.0)
        value = np.stack(arrays).transpose(0, 3, 1, 2)
        x = torch.from_numpy(value).to(self.device)
        x = (x - self.mean) / self.std
        amp = self.device.type == "cuda" and self.dtype != torch.float32
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=amp,
            ),
        ):
            outputs = self.model.get_intermediate_layers(
                x,
                n=self.layers,
                reshape=False,
                return_class_token=True,
                norm=True,
            )
            result: dict[str, np.ndarray] = {}
            if len(outputs) != len(self.layers):
                raise RuntimeError("DINOv2 返回的 intermediate layer 数量不一致")
            for layer, output in zip(self.layers, outputs, strict=True):
                if not isinstance(output, tuple) or len(output) != 2:
                    raise RuntimeError("DINOv2 return_class_token 合同改变")
                patches, cls = output
                vectors = {
                    f"block{layer}_cls": cls,
                    f"block{layer}_mean": patches.mean(dim=1),
                }
                for power in self.gem_powers:
                    vectors[f"block{layer}_signed_gem_p{power:g}"] = _signed_gem(
                        patches, power, torch
                    )
                for name, vector in vectors.items():
                    result[name] = _l2_normalize(vector.float(), torch).cpu().numpy()
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "encoder": "dinov2_vitb14",
            "repo_commit": DINOV2_COMMIT,
            "weight_sha256": self.weight_sha256,
            "layers_zero_based": list(self.layers),
            "gem_definition": "positive_gem_minus_negative_gem",
            "gem_powers": list(self.gem_powers),
            "normalization": "ImageNet mean/std then output L2",
            "xformers_disabled": True,
            "compute_dtype": str(self.dtype),
            "formal_admission": True,
        }


class DinoV2MaskedPlaceEncoder(DinoV2PlaceEncoder):
    """DINOv2-B/14 descriptors aggregated only over valid background tokens."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        names = []
        for layer in self.layers:
            names.append(f"block{layer}_masked_mean")
            names.extend(f"block{layer}_masked_signed_gem_p{power:g}" for power in self.gem_powers)
        self.feature_names = tuple(names)
        self.patch_token_names = tuple(f"block{layer}_patch_tokens" for layer in self.layers)

    def _forward_tokens(self, images: Sequence[Image.Image]) -> list[Any]:
        torch = self.torch
        if not images:
            raise ValueError("images 不能为空")
        arrays = []
        for image in images:
            rgb = image.convert("RGB")
            if rgb.size != (518, 518):
                raise ValueError(f"DINOv2 place 输入必须为 518x518，实际 {rgb.size}")
            arrays.append(np.asarray(rgb, dtype=np.float32) / 255.0)
        value = np.stack(arrays).transpose(0, 3, 1, 2)
        x = torch.from_numpy(value).to(self.device)
        x = (x - self.mean) / self.std
        amp = self.device.type == "cuda" and self.dtype != torch.float32
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=amp,
            ),
        ):
            outputs = self.model.get_intermediate_layers(
                x,
                n=self.layers,
                reshape=False,
                return_class_token=True,
                norm=True,
            )
        if len(outputs) != len(self.layers):
            raise RuntimeError("DINOv2 返回的 intermediate layer 数量不一致")
        return [output[0] for output in outputs]

    def extract_masked(
        self, images: Sequence[Image.Image], valid_patch_masks: np.ndarray
    ) -> Mapping[str, np.ndarray]:
        patches_by_layer = self._forward_tokens(images)
        return self._aggregate_masked(patches_by_layer, valid_patch_masks, len(images))

    def _aggregate_masked(
        self, patches_by_layer: Sequence[Any], valid_patch_masks: np.ndarray, batch_size: int
    ) -> Mapping[str, np.ndarray]:
        torch = self.torch
        valid = torch.as_tensor(valid_patch_masks, dtype=torch.bool, device=self.device)
        if valid.ndim != 2 or valid.shape[0] != batch_size:
            raise ValueError("valid_patch_masks 形状必须为 [N,P]")
        if bool((valid.sum(dim=1) == 0).any()):
            raise ValueError("存在无有效背景 patch 的样本")
        result: dict[str, np.ndarray] = {}
        for layer, patches in zip(self.layers, patches_by_layer, strict=True):
            if patches.shape[:2] != valid.shape:
                raise ValueError(
                    f"block{layer} token 形状 {tuple(patches.shape[:2])} "
                    f"与 mask {tuple(valid.shape)} 不一致"
                )
            vectors = {f"block{layer}_masked_mean": _masked_mean(patches, valid, torch)}
            for power in self.gem_powers:
                vectors[f"block{layer}_masked_signed_gem_p{power:g}"] = _masked_signed_gem(
                    patches, valid, power, torch
                )
            for name, vector in vectors.items():
                result[name] = _l2_normalize(vector.float(), torch).cpu().numpy()
        return result

    def extract_patch_tokens(self, images: Sequence[Image.Image]) -> Mapping[str, np.ndarray]:
        outputs = self._forward_tokens(images)
        return {
            name: patches.float().cpu().numpy()
            for name, patches in zip(self.patch_token_names, outputs, strict=True)
        }

    def extract_masked_with_tokens(
        self, images: Sequence[Image.Image], valid_patch_masks: np.ndarray
    ) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
        patches_by_layer = self._forward_tokens(images)
        features = self._aggregate_masked(patches_by_layer, valid_patch_masks, len(images))
        tokens = {
            name: patches.float().cpu().numpy()
            for name, patches in zip(self.patch_token_names, patches_by_layer, strict=True)
        }
        return features, tokens

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        value.update(
            {
                "encoder": "dinov2_vitb14_masked_patch",
                "aggregation": "valid_background_tokens_only",
                "formal_admission": True,
            }
        )
        return value
