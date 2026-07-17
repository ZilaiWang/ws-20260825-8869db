"""P04 教师适配器。

所有网络依赖均延迟导入；构建器只从显式本地路径读取权重和
官方源码，不在提取期间联网。
"""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from rsdet.features.p04_cache import sha256_file

DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
CLEANDIFT_COMMIT = "b070976b22b125167384eed5c96be3a694468763"
CLEANDIFT_SD15_SHA256 = "56697cc83cef762ac7ca0c8b9e749ee0abacfb426da92dc7fd5d7025ec727516"
CONVNEXT_TINY_SHA256 = "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"


def verify_file(path: str | Path, expected_sha256: str | None = None) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"权重文件不存在: {resolved}")
    if expected_sha256:
        actual = sha256_file(resolved)
        if actual != expected_sha256.lower():
            raise ValueError(
                f"文件 SHA-256 不匹配: {resolved}; expected={expected_sha256}, actual={actual}"
            )
    return resolved


def verify_git_commit(repo: str | Path, expected_commit: str) -> Path:
    resolved = Path(repo).expanduser().resolve()
    if not (resolved / ".git").exists():
        raise FileNotFoundError(f"不是 Git 仓库: {resolved}")
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=resolved,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise ValueError(f"{resolved} commit 不匹配: expected={expected_commit}, actual={actual}")
    return resolved


def _torch_dtype(torch: Any, name: str) -> Any:
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in values:
        raise ValueError(f"不支持 compute_dtype={name!r}")
    return values[name]


def _pil_batch_to_tensor(images: Sequence[Image.Image], torch: Any, device: Any) -> Any:
    arrays = [np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0 for image in images]
    value = np.stack(arrays, axis=0).transpose(0, 3, 1, 2)
    return torch.from_numpy(value).to(device=device)


class TeacherAdapter(ABC):
    """无标签教师的最小接口。"""

    teacher_id: str
    feature_names: tuple[str, ...]

    @abstractmethod
    def extract(
        self, images: Sequence[Image.Image], *, sample_keys: Sequence[str]
    ) -> Mapping[str, np.ndarray]:
        """返回若干 [N,D] float32 CPU 向量。"""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """返回参与 cache fingerprint 的模型合同。"""


class MockTeacherAdapter(TeacherAdapter):
    """只用于本地 cache/CLI 测试，服务器正式任务禁用。"""

    teacher_id = "mock"
    feature_names = ("mock_stats",)

    def extract(
        self, images: Sequence[Image.Image], *, sample_keys: Sequence[str]
    ) -> Mapping[str, np.ndarray]:
        if len(images) != len(sample_keys):
            raise ValueError("images/sample_keys 数量不一致")
        rows: list[np.ndarray] = []
        for image in images:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            rows.append(
                np.concatenate(
                    [
                        array.mean(axis=(0, 1)),
                        array.std(axis=(0, 1)),
                        np.asarray([array.min(), array.max()], dtype=np.float32),
                    ]
                )
            )
        return {"mock_stats": np.stack(rows).astype(np.float32)}

    def metadata(self) -> dict[str, Any]:
        return {"teacher_id": self.teacher_id, "implementation": "numpy_rgb_stats_v1"}


class ConvNeXtTinyAdapter(TeacherAdapter):
    teacher_id = "convnext_tiny_imagenet1k_v1"
    feature_names = ("convnext_gap",)

    def __init__(self, *, weights: str | Path, device: str, compute_dtype: str) -> None:
        try:
            import torch
            from torchvision.models import convnext_tiny
        except ImportError as error:
            raise RuntimeError("ConvNeXt teacher 需要 torch/torchvision") from error
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = _torch_dtype(torch, compute_dtype)
        self.weight_path = verify_file(weights, CONVNEXT_TINY_SHA256)
        model = convnext_tiny(weights=None)
        try:
            state = torch.load(self.weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(self.weight_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        # 保持 FP32 参数，只在前向时 autocast，与 P03 linear-probe 一致。
        self.model = model.to(device=self.device)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[None, :, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[None, :, None, None]

    def extract(
        self, images: Sequence[Image.Image], *, sample_keys: Sequence[str]
    ) -> Mapping[str, np.ndarray]:
        torch = self.torch
        x = _pil_batch_to_tensor(images, torch, self.device)
        x = (x - self.mean) / self.std
        amp = self.device.type == "cuda" and self.dtype != torch.float32
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=self.dtype, enabled=amp
        ):
            x = self.model.features(x)
            x = self.model.avgpool(x)
            x = self.model.classifier[:-1](x)
        return {"convnext_gap": x.float().cpu().numpy()}

    def metadata(self) -> dict[str, Any]:
        return {
            "teacher_id": self.teacher_id,
            "weight_filename": self.weight_path.name,
            "weight_sha256": CONVNEXT_TINY_SHA256,
            "feature_location": "features->avgpool->classifier[:-1]",
            "normalization": "ImageNet mean/std",
            "compute_dtype": str(self.dtype),
        }


class DinoV2Adapter(TeacherAdapter):
    def __init__(
        self,
        *,
        architecture: str,
        repo: str | Path,
        weights: str | Path,
        expected_weight_sha256: str,
        device: str,
        compute_dtype: str,
        include_patch_mean: bool = False,
    ) -> None:
        if architecture not in {"dinov2_vits14", "dinov2_vitb14"}:
            raise ValueError("P04 只允许 DINOv2-S/14 或 B/14")
        if not expected_weight_sha256:
            raise ValueError("DINOv2 必须提供下载后冻结的 SHA-256")
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("DINOv2 teacher 需要 torch") from error
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = _torch_dtype(torch, compute_dtype)
        self.repo = verify_git_commit(repo, DINOV2_COMMIT)
        self.weight_path = verify_file(weights, expected_weight_sha256)
        self.weight_sha256 = expected_weight_sha256.lower()
        self.architecture = architecture
        self.include_patch_mean = include_patch_mean
        self.teacher_id = architecture
        names = ["dino_cls"]
        if include_patch_mean:
            names.append("dino_cls_patchmean")
        self.feature_names = tuple(names)

        # 不把另一个可选 CUDA 扩展变成隐含实验变量；官方实现会在
        # xFormers 缺失时安全回退到 PyTorch SDPA。
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        sys.path.insert(0, str(self.repo))
        try:
            backbones = importlib.import_module("dinov2.hub.backbones")
            builder = getattr(backbones, architecture)
            model = builder(pretrained=False)
        finally:
            if sys.path[0] == str(self.repo):
                sys.path.pop(0)
        try:
            state = torch.load(self.weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(self.weight_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        self.model = model.to(device=self.device)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[None, :, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[None, :, None, None]

    def extract(
        self, images: Sequence[Image.Image], *, sample_keys: Sequence[str]
    ) -> Mapping[str, np.ndarray]:
        torch = self.torch
        x = _pil_batch_to_tensor(images, torch, self.device)
        x = (x - self.mean) / self.std
        amp = self.device.type == "cuda" and self.dtype != torch.float32
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=self.dtype, enabled=amp
        ):
            output = self.model.forward_features(x)
            cls = output["x_norm_clstoken"]
            result = {"dino_cls": cls.float().cpu().numpy()}
            if self.include_patch_mean:
                patch_mean = output["x_norm_patchtokens"].mean(dim=1)
                combined = torch.cat([cls, patch_mean], dim=1)
                result["dino_cls_patchmean"] = combined.float().cpu().numpy()
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "teacher_id": self.teacher_id,
            "repo_commit": DINOV2_COMMIT,
            "weight_filename": self.weight_path.name,
            "weight_sha256": self.weight_sha256,
            "register_tokens": 0,
            "xformers_disabled": True,
            "features": list(self.feature_names),
            "normalization": "ImageNet mean/std",
            "compute_dtype": str(self.dtype),
        }


class StableDiffusionFeatureAdapter(TeacherAdapter):
    """CleanDIFT-SD1.5 或嵌套 ensemble 的 raw DIFT 全局特征。"""

    def __init__(
        self,
        *,
        mode: str,
        cleandift_repo: str | Path,
        base_model: str | Path,
        device: str,
        compute_dtype: str,
        clean_weights: str | Path | None = None,
        clean_weight_sha256: str = CLEANDIFT_SD15_SHA256,
        latent_policy: str = "mode",
        raw_ensemble_sizes: Sequence[int] = (1, 4, 8),
        global_seed: int = 42,
    ) -> None:
        if mode not in {"cleandift", "raw_dift"}:
            raise ValueError("mode 必须是 cleandift 或 raw_dift")
        if latent_policy not in {"mode", "sample"}:
            raise ValueError("latent_policy 必须是 mode 或 sample")
        try:
            import torch
            from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
            from safetensors.torch import load_file
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Stable diffusion teacher 需要 torch/diffusers/transformers/safetensors"
            ) from error

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = _torch_dtype(torch, compute_dtype)
        self.mode = mode
        self.teacher_id = "cleandift_sd15" if mode == "cleandift" else "raw_dift_sd15"
        self.repo = verify_git_commit(cleandift_repo, CLEANDIFT_COMMIT)
        self.base_model = Path(base_model).expanduser().resolve()
        if not self.base_model.is_dir():
            raise FileNotFoundError(f"SD1.5 本地 snapshot 不存在: {self.base_model}")
        self.latent_policy = latent_policy
        self.global_seed = int(global_seed)
        sizes = tuple(sorted(set(int(value) for value in raw_ensemble_sizes)))
        if not sizes or sizes[0] <= 0:
            raise ValueError("raw_ensemble_sizes 必须是正整数")
        self.raw_ensemble_sizes = sizes

        sys.path.insert(0, str(self.repo))
        try:
            module = importlib.import_module("src.sd_feature_extraction")
            extractor_cls = module.SD15UNetFeatureExtractor
        finally:
            if sys.path[0] == str(self.repo):
                sys.path.pop(0)
        unet = extractor_cls()
        if mode == "cleandift":
            if clean_weights is None:
                raise ValueError("cleandift mode 必须提供 clean_weights")
            self.clean_weight_path = verify_file(clean_weights, clean_weight_sha256)
            state = load_file(str(self.clean_weight_path), device="cpu")
            unet.load_state_dict(state, strict=True)
            self.clean_weight_sha256 = clean_weight_sha256.lower()
        else:
            base_unet = UNet2DConditionModel.from_pretrained(
                self.base_model,
                subfolder="unet",
                local_files_only=True,
            )
            unet.load_state_dict(base_unet.state_dict(), strict=True)
            del base_unet
            self.clean_weight_path = None
            self.clean_weight_sha256 = None
        self.unet = unet.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)

        self.vae = AutoencoderKL.from_pretrained(
            self.base_model, subfolder="vae", local_files_only=True
        ).eval().requires_grad_(False).to(self.device)
        tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model, subfolder="tokenizer", local_files_only=True
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self.base_model, subfolder="text_encoder", local_files_only=True
        ).eval().requires_grad_(False).to(self.device)
        tokens = tokenizer(
            [""],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            self.empty_prompt = text_encoder(tokens.input_ids.to(self.device))[0].to(self.dtype)
        del text_encoder
        self.scheduler = DDIMScheduler.from_pretrained(
            self.base_model, subfolder="scheduler", local_files_only=True
        )

        if mode == "cleandift":
            self.feature_names = ("clean_map0", "clean_map6", "clean_map9")
        else:
            names = []
            for location in ("map0_t100", "map6_t261"):
                names.extend(f"raw_{location}_e{size}" for size in self.raw_ensemble_sizes)
            self.feature_names = tuple(names)

    def _latent(self, images: Sequence[Image.Image], sample_keys: Sequence[str]) -> Any:
        torch = self.torch
        import torch.nn.functional as functional

        x = _pil_batch_to_tensor(images, torch, self.device)
        x = functional.interpolate(
            x,
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp_(0.0, 1.0)
        x = x.mul(2).sub(1)
        with torch.inference_mode():
            distribution = self.vae.encode(x).latent_dist
            if self.latent_policy == "mode":
                latent = distribution.mode()
            else:
                values = []
                for index, key in enumerate(sample_keys):
                    seed = _stable_seed(self.global_seed, key, "vae")
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                    noise = torch.randn(
                        distribution.mean[index : index + 1].shape,
                        generator=generator,
                        device=self.device,
                        dtype=distribution.mean.dtype,
                    )
                    values.append(
                        distribution.mean[index : index + 1]
                        + distribution.std[index : index + 1] * noise
                    )
                latent = torch.cat(values, dim=0)
        scale = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        return (latent * scale).to(self.dtype)

    def _forward_maps(self, latent: Any, timestep: int, prompt: Any) -> Mapping[str, Any]:
        torch = self.torch
        timesteps = torch.full(
            (latent.shape[0],), timestep, device=self.device, dtype=torch.long
        )
        with torch.inference_mode():
            return self.unet(
                latent,
                timesteps,
                encoder_hidden_states=prompt,
                added_cond_kwargs={},
            )

    def extract(
        self, images: Sequence[Image.Image], *, sample_keys: Sequence[str]
    ) -> Mapping[str, np.ndarray]:
        if len(images) != len(sample_keys):
            raise ValueError("images/sample_keys 数量不一致")
        torch = self.torch
        latent = self._latent(images, sample_keys)
        prompt = self.empty_prompt.repeat(len(images), 1, 1)
        if self.mode == "cleandift":
            maps = self._forward_maps(latent, 261, prompt)
            selected = {
                "clean_map0": maps["mid"],
                "clean_map6": maps["us6"],
                "clean_map9": maps["us9"],
            }
            return {
                name: value.float().mean(dim=(2, 3)).cpu().numpy()
                for name, value in selected.items()
            }

        if len(images) != 1:
            raise ValueError("raw DIFT 为保证每对象噪声子序列固定，batch 必须为 1")
        max_ensemble = max(self.raw_ensemble_sizes)
        repeated_latent = latent.repeat(max_ensemble, 1, 1, 1)
        repeated_prompt = prompt.repeat(max_ensemble, 1, 1)
        noises = []
        for ensemble_index in range(max_ensemble):
            seed = _stable_seed(self.global_seed, sample_keys[0], "raw", ensemble_index)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noises.append(
                torch.randn(
                    latent.shape,
                    generator=generator,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        noise = torch.cat(noises, dim=0)
        output: dict[str, np.ndarray] = {}
        for timestep, map_key, location in ((100, "mid", "map0_t100"), (261, "us6", "map6_t261")):
            timesteps = torch.full(
                (max_ensemble,), timestep, device=self.device, dtype=torch.long
            )
            noisy = self.scheduler.add_noise(repeated_latent, noise, timesteps)
            maps = self._forward_maps(noisy, timestep, repeated_prompt)[map_key]
            pooled = maps.float().mean(dim=(2, 3))
            for size in self.raw_ensemble_sizes:
                output[f"raw_{location}_e{size}"] = (
                    pooled[:size].mean(dim=0, keepdim=True).cpu().numpy()
                )
        return output

    def metadata(self) -> dict[str, Any]:
        return {
            "teacher_id": self.teacher_id,
            "cleandift_repo_commit": CLEANDIFT_COMMIT,
            "base_model_path": str(self.base_model),
            "clean_weight_filename": self.clean_weight_path.name
            if self.clean_weight_path
            else None,
            "clean_weight_sha256": self.clean_weight_sha256,
            "latent_policy": self.latent_policy,
            "empty_prompt": True,
            "prompt_utf8_sha256": hashlib.sha256(b"").hexdigest(),
            "clean_timestep": 261 if self.mode == "cleandift" else None,
            "raw_branches": {"map0": 100, "map6": 261}
            if self.mode == "raw_dift"
            else None,
            "raw_ensemble_sizes": list(self.raw_ensemble_sizes)
            if self.mode == "raw_dift"
            else None,
            "feature_locations": (
                {"clean_map0": "mid", "clean_map6": "us6", "clean_map9": "us9"}
                if self.mode == "cleandift"
                else {
                    "raw_map0": "mid@t100",
                    "raw_map6": "us6@t261",
                }
            ),
            "pooling": "global_average_over_spatial_axes",
            "compute_dtype": str(self.dtype),
        }


def _stable_seed(*parts: object) -> int:
    import hashlib

    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def build_teacher(
    teacher_id: str,
    *,
    device: str,
    compute_dtype: str,
    options: Mapping[str, Any],
) -> TeacherAdapter:
    """根据冻结配置构建教师。"""

    if teacher_id == "mock":
        return MockTeacherAdapter()
    if teacher_id == "convnext_tiny":
        return ConvNeXtTinyAdapter(
            weights=options["weights"], device=device, compute_dtype=compute_dtype
        )
    if teacher_id in {"dinov2_vits14", "dinov2_vitb14"}:
        return DinoV2Adapter(
            architecture=teacher_id,
            repo=options["repo"],
            weights=options["weights"],
            expected_weight_sha256=options["weight_sha256"],
            device=device,
            compute_dtype=compute_dtype,
            include_patch_mean=bool(options.get("include_patch_mean", False)),
        )
    if teacher_id in {"cleandift_sd15", "raw_dift_sd15"}:
        return StableDiffusionFeatureAdapter(
            mode="cleandift" if teacher_id == "cleandift_sd15" else "raw_dift",
            cleandift_repo=options["repo"],
            base_model=options["base_model"],
            clean_weights=options.get("weights"),
            clean_weight_sha256=options.get(
                "weight_sha256", CLEANDIFT_SD15_SHA256
            ),
            device=device,
            compute_dtype=compute_dtype,
            latent_policy=options.get("latent_policy", "mode"),
            raw_ensemble_sizes=options.get("raw_ensemble_sizes", (1, 4, 8)),
            global_seed=int(options.get("global_seed", 42)),
        )
    raise ValueError(f"未知 teacher_id={teacher_id!r}")
