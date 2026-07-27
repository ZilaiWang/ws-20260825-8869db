# CV3-MODEL-ASSET-ENV-TASK-00：M1/M3 共用模型资产与环境冻结

## 0. 任务结论与边界

本任务是 M1（YOLO26-s）与 M3（RT-DETR-L）正式 CV3 OOF 的共同前置门禁。只建立一次环境、下载一次官方初始化权重，并生成唯一只读锁：

```text
/workspace/cv3-model-assets/MODEL_ASSET_ENV_LOCK.json
```

本任务不训练、不生成 OOF、不比较模型。后续 M1/M3 必须在同一服务器、同一虚拟环境、同一资产目录中先运行 `verify`，禁止重新联网解析权重。

冻结版本：

```text
Python 3.10.12
torch 2.5.1+cu121
torchvision 0.20.1+cu121
CUDA runtime 12.1
ultralytics 8.4.103
numpy 1.26.4
Pillow 10.4.0
PyYAML 6.0.2
```

GPU 冻结为本服务器已批准的 `NVIDIA GeForce RTX 4080 SUPER`。若实际名称不同，停止并回报，不得自行放宽。

## 1. 固定路径

```bash
REPO=/workspace/xh-202625
MODEL_REPO=/workspace/xh-202625-model
VENV=/workspace/venvs/cv3-model-cu121
ASSET_ROOT=/workspace/cv3-model-assets
LOCK=/workspace/cv3-model-assets/MODEL_ASSET_ENV_LOCK.json
ROOT=/workspace/results/CV3-MODEL-ASSET-ENV-TASK-00
SPEC=/workspace/xh-202625/configs/experiments/cv3_model_asset_env.json
GPU_NAME="NVIDIA GeForce RTX 4080 SUPER"
```

不得把权重放进 Git。`$ASSET_ROOT/*.partial` 不是合法资产，任务结束后必须不存在。

## 2. 起始门禁

```bash
set -euo pipefail
mkdir -p "$ROOT/logs"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

python3.10 --version 2>&1 | tee "$ROOT/logs/python-bootstrap.log"
test "$(python3.10 -c 'import platform; print(platform.python_version())')" = "3.10.12"
test -f "$SPEC"
test -d "$MODEL_REPO"

if test -e "$LOCK"; then
  test -d "$VENV"
  test -d "$ASSET_ROOT"
elif test -e "$VENV" || test -e "$ASSET_ROOT"; then
  echo "BLOCKED_PARTIAL_PREEXISTING_MODEL_ENV_OR_ASSET_ROOT" >&2
  exit 2
fi
```

若 `$LOCK` 已存在，本任务不得重建或覆盖，直接跳至第 7 节验证。若 `$VENV` 或 `$ASSET_ROOT` 已有其他内容而锁不存在，停止并报告；不要猜测它们是否可复用。

## 3. 从空目录建立独立环境

仅当 `$LOCK`、`$VENV`、`$ASSET_ROOT` 均不存在时执行：

```bash
python3.10 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip

python -m pip install \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  ultralytics==8.4.103 \
  numpy==1.26.4 Pillow==10.4.0 PyYAML==6.0.2 \
  opencv-python==4.10.0.84 pytest ruff

python -m pip check 2>&1 | tee "$ROOT/logs/pip-check.log"
python -m pip freeze > "$ROOT/pip-freeze.txt"
```

说明：两个仓库的 Python distribution 都名为 `rsdet`，不可同时 editable
install。本系列任务始终通过各自仓库的显式 `PYTHONPATH=src` 导入代码；不要运行会重新解析项目依赖或覆盖另一个仓库导入路径的
`pip install -e .`。

## 4. 官方资产下载、逐文件验证与原子发布

```bash
mkdir "$ASSET_ROOT"

download_one () {
  local url="$1"
  local final="$2"
  local size="$3"
  local sha="$4"
  local partial="${final}.partial"

  test ! -e "$final"
  test ! -e "$partial"
  curl --location --fail --retry 5 --retry-all-errors \
    --output "$partial" "$url"
  test "$(stat -c %s "$partial")" = "$size"
  echo "$sha  $partial" | sha256sum -c -
  mv "$partial" "$final"
  test -f "$final"
  test ! -e "$partial"
}

download_one \
  "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt" \
  "$ASSET_ROOT/yolo26s.pt" \
  20422725 \
  646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b

download_one \
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt" \
  "$ASSET_ROOT/rtdetr-l.pt" \
  66511432 \
  6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f
```

只允许上述两个 HTTPS 官方 URL。下载写入 `.partial`，大小与 SHA-256 全部通过后才以同目录 `mv` 原子改名。任何失败都必须保留日志并停止，不得用同名镜像替代。

## 5. 代码质量门禁

```bash
cd "$REPO"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO/src"
export PYTHONNOUSERSITE=1

sha256sum -c docs/server/CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/formal-stage-code-sha256.log"
pytest -q tests/test_model_asset_env_lock.py \
  2>&1 | tee "$ROOT/logs/pytest-focused.log"
ruff check \
  src/rsdet/experiments/model_asset_env_lock.py \
  scripts/lock_cv3_model_assets.py \
  tests/test_model_asset_env_lock.py \
  2>&1 | tee "$ROOT/logs/ruff-focused.log"
```

必须全部通过。服务器不得编辑代码绕过门禁。

## 6. 创建唯一不可变锁

```bash
python scripts/lock_cv3_model_assets.py create \
  --config "$SPEC" \
  --asset-root "$ASSET_ROOT" \
  --expected-gpu "$GPU_NAME" \
  --output "$LOCK" \
  2>&1 | tee "$ROOT/logs/create-lock.log"

python - <<'PY'
from pathlib import Path

lock = Path("/workspace/cv3-model-assets/MODEL_ASSET_ENV_LOCK.json")
assert lock.is_file()
assert lock.stat().st_mode & 0o222 == 0, oct(lock.stat().st_mode)
print("IMMUTABLE_MODE_BITS_PASS")
PY
```

锁内必须同时记录：

- 规范文件绝对路径与 SHA-256；
- 六个关键 Python 包的实际版本，以及虚拟环境内全部已安装 distribution
  的名称/版本清单；
- PyTorch CUDA runtime、cuDNN；
- PyTorch GPU 名称、显存、计算能力；
- `nvidia-smi` 的 GPU UUID、驱动与显存；
- 两个权重的绝对路径、官方 URL、字节数与 SHA-256；
- 对锁正文的 `lock_fingerprint`。

脚本拒绝覆盖已存在锁，并将新锁权限设为只读。不要手工编辑或 `chmod` 后重写。

## 7. 独立重算验证

无论锁是本次新建还是先前已存在，都必须执行：

```bash
source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO/src"
export PYTHONNOUSERSITE=1

python scripts/lock_cv3_model_assets.py verify \
  --config "$SPEC" \
  --asset-root "$ASSET_ROOT" \
  --expected-gpu "$GPU_NAME" \
  --lock "$LOCK" \
  --report "$ROOT/verification.json" \
  2>&1 | tee "$ROOT/logs/verify-lock.log"

python - <<'PY'
import json
from pathlib import Path

root = Path("/workspace/results/CV3-MODEL-ASSET-ENV-TASK-00")
payload = json.loads((root / "verification.json").read_text(encoding="utf-8"))
assert payload["status"] == "pass", payload
assert payload["asset_count"] == 2, payload
assert payload["gpu_name"] == "NVIDIA GeForce RTX 4080 SUPER", payload
print("CV3_MODEL_ASSET_ENV_GATE_PASS")
PY
```

验证会重新读取配置、重新计算两个大文件的 SHA-256、重新检查所有版本/CUDA/GPU，并要求重算锁与原锁逐字段相等。

## 8. 冻结后离线 smoke

先切断模型框架的在线资产解析：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ULTRALYTICS_OFFLINE=true
```

使用正式权重做最小加载 smoke：

```bash
python - <<'PY' 2>&1 | tee "$ROOT/logs/model-load-smoke.log"
from pathlib import Path
from ultralytics import RTDETR, YOLO

root = Path("/workspace/cv3-model-assets")
yolo = YOLO(str(root / "yolo26s.pt"))
rtdetr = RTDETR(str(root / "rtdetr-l.pt"))
assert yolo.model is not None
assert rtdetr.model is not None
print("MODEL_LOAD_SMOKE_PASS")
PY
```

本 smoke 只证明两类官方权重在冻结软件栈中可离线反序列化；不形成精度结论，也不改变权重。

随后必须经过项目真实 adapter，而不只调用 Ultralytics 原生加载：

```bash
PYTHONPATH="$MODEL_REPO/src" python - <<'PY' \
  2>&1 | tee "$ROOT/logs/project-adapter-smoke.log"
import math
from pathlib import Path

import numpy as np

from rsdet.contracts import InferenceSample
from rsdet.models.ultralytics_adapter import (
    UltralyticsDetector,
    validate_pretrained_coco_label_space,
)

asset_root = Path("/workspace/cv3-model-assets")
image = np.zeros((1024, 1024, 3), dtype=np.uint8)
sample = InferenceSample(
    image_id=1,
    image=image,
    width=1024,
    height=1024,
)

for family, filename, max_det in (
    ("yolo", "yolo26s.pt", 500),
    ("rtdetr", "rtdetr-l.pt", 300),
):
    detector = UltralyticsDetector(
        family=family,
        imgsz=1024,
        confidence=0.001,
        max_detections=max_det,
        half=True,
    )
    detector.load(str(asset_root / filename))
    class_names = validate_pretrained_coco_label_space(detector._model.names)
    detector.to("cuda:0")
    detector.eval()
    outputs = detector.predict([sample])
    assert len(outputs) == 1
    prediction = outputs[0]
    assert prediction.image_id == 1
    assert (
        len(prediction.boxes_xyxy)
        == len(prediction.scores)
        == len(prediction.labels)
    )
    assert len(prediction.boxes_xyxy) <= max_det
    class_count = len(class_names)
    for box, score, label in zip(
        prediction.boxes_xyxy,
        prediction.scores,
        prediction.labels,
        strict=True,
    ):
        assert len(box) == 4 and all(math.isfinite(value) for value in box)
        x1, y1, x2, y2 = box
        assert 0 <= x1 < x2 <= 1024
        assert 0 <= y1 < y2 <= 1024
        assert math.isfinite(score) and 0 <= score <= 1
        assert isinstance(label, int) and 0 <= label < class_count
    print("PROJECT_ADAPTER_SMOKE_PASS", family, len(prediction.labels))
PY
```

官方初始化仍是其原始 COCO-80 预训练类别空间，因此 A00 先严格核验连续
0—79 索引与完整 COCO-80 类别名，再要求 adapter 输出 label 位于该空间。
正式训练后的 M1/M3 才由 OOF finalize 强制 0—24。该 smoke 用于在数百
epoch 训练前捕获错误权重、项目 adapter 的真实 `predict()` API、设备、
半精度和坐标契约问题。

## 9. 打包回传

```bash
cp "$LOCK" "$ROOT/MODEL_ASSET_ENV_LOCK.json"
sha256sum \
  "$SPEC" \
  "$ASSET_ROOT/yolo26s.pt" \
  "$ASSET_ROOT/rtdetr-l.pt" \
  "$LOCK" \
  > "$ROOT/ARTIFACTS_SHA256.txt"

tar -C /workspace/results \
  -czf /workspace/results/CV3-MODEL-ASSET-ENV-TASK-00-return.tar.gz \
  CV3-MODEL-ASSET-ENV-TASK-00
sha256sum /workspace/results/CV3-MODEL-ASSET-ENV-TASK-00-return.tar.gz \
  > /workspace/results/CV3-MODEL-ASSET-ENV-TASK-00-return.tar.gz.sha256
```

回传：

```text
CV3-MODEL-ASSET-ENV-TASK-00-return.tar.gz
CV3-MODEL-ASSET-ENV-TASK-00-return.tar.gz.sha256
```

不回传两个大权重；它们和唯一锁继续保留在服务器固定路径，供 M1/M3 只读复用。

## 10. 最终回报格式

1. 状态：`complete` 或明确失败门禁；
2. Python/torch/torchvision/CUDA/ultralytics/numpy/Pillow/PyYAML 实际版本；
3. GPU 名称、UUID、驱动、计算能力、显存；
4. 两个资产的 URL、字节数、SHA-256 是否匹配；
5. focused pytest 与 Ruff；
6. `verification.json.status`；
7. 离线双模型原生加载与项目真实 adapter smoke；
8. 锁 SHA-256 与 fingerprint；
9. 回传包路径、大小、SHA-256；
10. 明确确认：没有训练、没有改写 P03/P04 缓存、没有删除既有服务器资产。
