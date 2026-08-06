# E 服务器任务：10K 切片、恢复、融合与正式分段测速

状态：可执行工程任务；正式比赛时延结论取决于图像来源  
模板：`configs/experiments/e_10k_pipeline_cv3.template.yaml`

## 1. 任务边界

E 负责：

```text
10K 图读取完成
→ 滑窗切片
→ tile 预处理与批推理
→ tile 后处理
→ 恢复 parent 坐标并裁边
→ 跨 tile 细类 NMS
→ 同大类高 IoU 重复抑制
→ parent 结果序列化
```

本轮不训练模型、不选择 25 类阈值、不修改 M1/M3 权重。

当前主仓库 `src/rsdet/postprocess/tile_fusion.py` 仍是骨架；正式执行使用已通过
坐标/融合测试的同级 `xh-202625-model` 实现。不要用目录覆盖方式把同级仓库
整体复制回主仓库。

## 2. 冻结工程工作点

首轮固定：

```text
input size: 10000x10000
tile size: 1280
overlap: 256
stride: 1024
expected tiles for exact 10K: 100
fine-class NMS IoU: 0.55
coarse duplicate NMS IoU: 0.85
tile batch: M1=8, M3=4
candidate threshold: 0.001
```

不得一边更改 tile/overlap，一边把速度差归因于模型。后续几何消融另开任务。

## 3. 三层 smoke

### S0：纯坐标

- `generate_tiles(10000,10000,1280,256)` 恰好得到 100 个唯一 tile；
- 所有像素被覆盖；
- 最后一行/列贴边；
- tile 内已知框经恢复、裁边后回到正确全图坐标。

### S1：重复融合

- 同一目标出现在相邻 tile 时，细类 NMS 只保留一个；
- 同一目标被预测为同大类不同细类且 IoU 极高时，第二级去重只保留高分框；
- 相邻真实目标不能被错误融合；
- 空预测图正确输出空 COCO 列表。

### S2：真实 adapter

- M1 至少一张真实图、M3 若进入候选也至少一张；
- 输出 0—24 类、有限分数和原图绝对坐标；
- raw tile proposal、恢复后 proposal、fused proposal 数量守恒可解释；
- 不出现越界、负面积、NaN/Inf。

## 4. 图像来源必须明确

`benchmark_contract.json` 的 `image_source_type` 只允许：

- `real_official`：真实官方 10K 图；还必须进入
  `runtime_10k.py::OFFICIAL_IMAGE_MANIFEST_REGISTRY` 并通过代码锁，才具备
  官方声明的图像来源资格；
- `real_project_proxy`：真实但非官方测试图，只作工程近似；
- `synthetic`：程序生成；
- `stitched`：小图拼接。

后三种非正式或未注册输入即使通过 20 秒，也只能写“工程 smoke 通过”，不得
写“官方时延通过”。当前注册表刻意为空，因此本任务现阶段不会接受
`real_official` 自我声明。收到官方 10K 后，必须先独立清点、评审其 manifest
并更新代码注册表与代码锁，再另立正式测速任务。图像清单及每张图内容 SHA
必须记录。十张 measured 图必须内容互异；三次 warmup 允许复用 measured 图，
但不计入科学证据。

## 5. 冻结测速合同

创建 `benchmark_contract.json`：

```json
{
  "contract_version": "runtime_10k_benchmark_v1",
  "image_source_type": "real_project_proxy",
  "image_manifest_sha256": "<64 hex>",
  "model_key": "M1",
  "checkpoint_sha256": "<64 hex>",
  "checkpoint_provenance_sha256": "<64 hex>",
  "engineering_checkpoint_only": true,
  "config_sha256": "<64 hex>",
  "hardware_sha256": "<64 hex>",
  "tile_size": 1280,
  "overlap": 256,
  "expected_tile_count": 100,
  "timing_method": "perf_counter_with_torch_cuda_synchronize",
  "cuda_synchronized": true,
  "warmup_runs": 3,
  "minimum_measured_runs": 10
}
```

当前模板固定 `engineering_checkpoint_only=true`，因此本轮只能形成
工程证据。本任务不限制 GPU 型号，但必须记录实际型号、驱动、
CUDA 环境和显存，且 measured runs 期间不允许其他 GPU 计算进程。
不同 GPU 上的耗时只能分别报告，不得直接归因为模型或切片策略差异。
官方声明资格取决于已注册的 `real_official` manifest、
`other_gpu_processes=none`、最终冻结 checkpoint 且
`engineering_checkpoint_only=false`；不以 GPU 商标或型号作为代码门禁。

另建 `hardware.json`：

```json
{
  "gpu_name": "NVIDIA ...",
  "driver_version": "...",
  "torch_version": "...",
  "cuda_runtime": "...",
  "cudnn_version": "...",
  "python_version": "...",
  "host_id": "...",
  "power_mode": "...",
  "other_gpu_processes": "none"
}
```

每一次运行写入 `runtime_samples.jsonl`。必填：

```json
{
  "run_index": 0,
  "image_id": 1,
  "warmup": true,
  "width": 10000,
  "height": 10000,
  "image_content_sha256": "<64 hex>",
  "image_source_type": "real_project_proxy",
  "model_key": "M1",
  "checkpoint_sha256": "<64 hex>",
  "config_sha256": "<64 hex>",
  "timing_method": "perf_counter_with_torch_cuda_synchronize",
  "cuda_synchronized": true,
  "tile_size": 1280,
  "overlap": 256,
  "tile_count": 100,
  "raw_proposal_count": 0,
  "fused_proposal_count": 0,
  "peak_vram_mib": 0,
  "phases": {
    "image_read": 0,
    "tiling": 0,
    "preprocess": 0,
    "model": 0,
    "tile_postprocess": 0,
    "coordinate_restore": 0,
    "fusion": 0,
    "serialization": 0
  }
}
```

GPU 阶段必须在开始计时前、结束取时前调用
`torch.cuda.synchronize()`。每张图前重置 peak-memory stats，记录真实峰值。
预热三次不计入分位数；正式至少十次。

## 6. 用真实 adapter 采集（不是只造 JSON）

先把模板解析为无占位符的 `resolved_config.yaml`，生成
`hardware.json`，再计算配置、入选 checkpoint、checkpoint provenance、
`image_manifest.json` 和硬件记录的 SHA256，最后写入
`benchmark_contract.json`。硬件文件必须先于合同存在，禁止在测速后替换
硬件声明。图像清单每项至少包含：

```json
{
  "image_id": 1,
  "relative_path": "images/10k/example.tif",
  "width": 10000,
  "height": 10000,
  "sha256": "<image content SHA256>",
  "image_source_type": "real_project_proxy"
}
```

主仓库当前只有模型无关接口；生产 adapter、批推理、切片与 NMS 位于同级
`xh-202625-model`。因此必须让同级仓库的 `src` 排在 `PYTHONPATH` 第一位，
先按以下命令一次性生成五个冻结输入。从本节到打包必须在
同一个 Bash 会话中按顺序执行，以保留 `RUN`、`DATA_ROOT` 等变量和
清理 trap；服务器 AI 也可将本文档的 Bash 代码块按顺序合并成一个
临时脚本执行。只修改开头九个变量；`GPU_NAME` 由现场自动读取，
不手工填写：

```bash
set -euo pipefail
RUN=/workspace/results/E-10K-M1
DATA_ROOT=/workspace/data/10k
CHECKPOINT=/workspace/results/M1-CV3-OOF/fold_0/training/runs/foundation/weights/last.pt
FOLD_METADATA=/workspace/results/M1-CV3-OOF/fold_0/fold_metadata.json
OOF_METADATA=/workspace/results/M1-CV3-OOF-aggregate/oof_metadata.json
SOURCE_FOLD=0
MODEL_KEY=M1
MODEL_FAMILY=yolo
SOURCE_TYPE=real_project_proxy
source /workspace/venvs/cv3-model-cu121/bin/activate
export PYTHONNOUSERSITE=1

ASSET_ROOT=/workspace/cv3-model-assets
ASSET_SPEC=/workspace/xh-202625/configs/experiments/cv3_model_asset_env.json
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader \
  | sed -n '1{s/^[[:space:]]*//;s/[[:space:]]*$//;p;}')"
test -n "$GPU_NAME"
ASSET_GATE_DIR="$(mktemp -d /tmp/E-10K-model-asset-gate.XXXXXX)"
ASSET_LOCK="$ASSET_GATE_DIR/MODEL_ASSET_ENV_LOCK.json"
ASSET_VERIFY_REPORT="$ASSET_GATE_DIR/model_asset_env_verification.json"
cleanup_asset_gate() {
  rm -f -- "$ASSET_LOCK" "$ASSET_VERIFY_REPORT"
  rmdir -- "$ASSET_GATE_DIR" 2>/dev/null || true
}
trap cleanup_asset_gate EXIT

cd /workspace/xh-202625
sha256sum -c docs/server/CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt
sha256sum -c docs/server/XH_MODEL_INTEGRATION_CODE_SHA256.txt
PYTHONPATH=src python scripts/lock_cv3_model_assets.py create \
  --config "$ASSET_SPEC" \
  --asset-root "$ASSET_ROOT" \
  --expected-gpu "$GPU_NAME" \
  --output "$ASSET_LOCK"
PYTHONPATH=src python scripts/lock_cv3_model_assets.py verify \
  --config "$ASSET_SPEC" \
  --asset-root "$ASSET_ROOT" \
  --expected-gpu "$GPU_NAME" \
  --lock "$ASSET_LOCK" \
  --report "$ASSET_VERIFY_REPORT"

# 证明真实采集脚本导入的是同级模型仓库实现，而非主仓库同名包。
PYTHONPATH=/workspace/xh-202625-model/src python - <<'PY'
import rsdet
from pathlib import Path

actual = Path(rsdet.__file__).resolve()
expected = Path("/workspace/xh-202625-model/src/rsdet/__init__.py").resolve()
assert actual == expected, (actual, expected)
print("MODEL_RSDET_IMPORT_OK", actual)
PY

# 所有代码、环境和 adapter 门禁必须先于结果目录创建。
PYTHONPATH=src python -m pytest -q tests/test_runtime_10k.py
PYTHONPATH=src python -m ruff check \
  src/rsdet/experiments/runtime_10k.py \
  scripts/benchmark_10k_pipeline.py scripts/audit_10k_runtime.py
cd /workspace/xh-202625-model
PYTHONPATH=src python -m pytest -q \
  tests/test_ultralytics_adapter.py tests/test_inference_pipeline.py \
  tests/test_tile_fusion.py tests/test_tiling_coordinates.py
cd /workspace/xh-202625

# 正式测速要求独占 GPU；发现任何既有 compute process 即停止。
test -z "$(nvidia-smi --query-compute-apps=pid,process_name \
  --format=csv,noheader | sed '/^[[:space:]]*$/d')" || {
  echo "BLOCKED_OTHER_GPU_COMPUTE_PROCESS" >&2
  nvidia-smi >&2
  exit 2
}

test ! -e "$RUN" || {
  echo "E 结果目录已存在，禁止覆盖或混入旧测速: $RUN" >&2
  exit 2
}
test -s "$CHECKPOINT"
test -s "$FOLD_METADATA"
test -s "$OOF_METADATA"
mkdir -p "$RUN"
cp "$ASSET_VERIFY_REPORT" "$RUN/model_asset_env_verification.json"
cp "$ASSET_LOCK" "$RUN/model_asset_env_lock.json"

RUN="$RUN" DATA_ROOT="$DATA_ROOT" CHECKPOINT="$CHECKPOINT" \
FOLD_METADATA="$FOLD_METADATA" OOF_METADATA="$OOF_METADATA" \
SOURCE_FOLD="$SOURCE_FOLD" \
MODEL_KEY="$MODEL_KEY" MODEL_FAMILY="$MODEL_FAMILY" \
SOURCE_TYPE="$SOURCE_TYPE" PYTHONPATH=src python - <<'PY'
import hashlib, json, os, socket, subprocess
from pathlib import Path

import torch, ultralytics, yaml
from PIL import Image

run = Path(os.environ["RUN"]).resolve()
root = Path(os.environ["DATA_ROOT"]).resolve()
checkpoint = Path(os.environ["CHECKPOINT"]).resolve()
source_type = os.environ["SOURCE_TYPE"]
allowed_sources = {"real_official", "real_project_proxy", "synthetic", "stitched"}
assert source_type in allowed_sources
assert ultralytics.__version__ == "8.4.103", (
    "E 正式适配器锁定 ultralytics==8.4.103，"
    f"当前为 {ultralytics.__version__}"
)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

fold_metadata_path = Path(os.environ["FOLD_METADATA"]).resolve()
oof_metadata_path = Path(os.environ["OOF_METADATA"]).resolve()
source_fold = int(os.environ["SOURCE_FOLD"])
fold_metadata = json.loads(fold_metadata_path.read_text(encoding="utf-8"))
oof_metadata = json.loads(oof_metadata_path.read_text(encoding="utf-8"))
assert fold_metadata["status"] == "fold_delivery_complete"
assert fold_metadata["model_key"] == os.environ["MODEL_KEY"]
assert int(fold_metadata["held_out_fold"]) == source_fold
assert Path(fold_metadata["artifacts"]["checkpoint"]).resolve() == checkpoint
assert fold_metadata["artifacts"]["checkpoint_sha256"] == sha(checkpoint)
assert oof_metadata["status"] == "complete_downstream_ready"
assert oof_metadata["downstream_admission"] is True
assert oof_metadata["model_key"] == os.environ["MODEL_KEY"]
fold_rows = [row for row in oof_metadata["folds"] if int(row["fold"]) == source_fold]
assert len(fold_rows) == 1
assert fold_rows[0]["checkpoint_sha256"] == sha(checkpoint)
assert fold_rows[0]["metadata_sha256"] == sha(fold_metadata_path)
provenance = {
    "contract_version": "checkpoint_provenance_v1",
    "status": "checkpoint_lineage_verified",
    "engineering_checkpoint_only": True,
    "model_key": os.environ["MODEL_KEY"],
    "source_fold": source_fold,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha(checkpoint),
    "fold_metadata": str(fold_metadata_path),
    "fold_metadata_sha256": sha(fold_metadata_path),
    "oof_metadata": str(oof_metadata_path),
    "oof_metadata_sha256": sha(oof_metadata_path),
}
provenance_path = run / "checkpoint_provenance.json"
provenance_path.write_text(
    json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

samples, seen = [], set()
suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in suffixes):
    with Image.open(path) as image:
        width, height = image.size
    if (width, height) != (10000, 10000):
        continue
    digest = sha(path)
    if digest in seen:
        continue
    seen.add(digest)
    samples.append({
        "image_id": len(samples) + 1,
        "relative_path": path.relative_to(root).as_posix(),
        "width": width,
        "height": height,
        "sha256": digest,
        "image_source_type": source_type,
    })
assert len(samples) >= 10, f"至少需要10张不同内容的10K图，实际{len(samples)}"
manifest = {"version": "e_10k_image_manifest_v1", "samples": samples}
manifest_path = run / "image_manifest.json"
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

template = Path("configs/experiments/e_10k_pipeline_cv3.template.yaml")
config = yaml.safe_load(template.read_text(encoding="utf-8"))
config["model"]["family"] = os.environ["MODEL_FAMILY"]
config["model"]["checkpoint"] = str(checkpoint)
if os.environ["MODEL_KEY"] == "M3":
    config["model"]["max_detections"] = 300
    config["batch_size"] = 4
config["input"]["data_root"] = str(root)
config["input"]["manifest"] = str(manifest_path)
config["output_json"] = str(run / "capture/predictions_10k_low.json")
config_path = run / "resolved_config.yaml"
config_path.write_text(
    yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
)

def command(args):
    return subprocess.check_output(args, text=True).strip()

hardware = {
    "gpu_name": command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()[0],
    "driver_version": command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()[0],
    "torch_version": torch.__version__,
    "ultralytics_version": ultralytics.__version__,
    "cuda_runtime": str(torch.version.cuda),
    "cudnn_version": str(torch.backends.cudnn.version()),
    "python_version": command(["python", "--version"]),
    "host_id": socket.gethostname(),
    "power_mode": "recorded_in_environment_txt",
    "other_gpu_processes": command(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]) or "none",
}
hardware_path = run / "hardware.json"
hardware_path.write_text(
    json.dumps(hardware, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

contract = json.loads(Path(
    "configs/experiments/e_10k_benchmark_contract.template.json"
).read_text(encoding="utf-8"))
contract.update({
    "image_source_type": source_type,
    "image_manifest_sha256": sha(manifest_path),
    "model_key": os.environ["MODEL_KEY"],
    "checkpoint_sha256": sha(checkpoint),
    "checkpoint_provenance_sha256": sha(provenance_path),
    "engineering_checkpoint_only": True,
    "config_sha256": sha(config_path),
    "hardware_sha256": sha(hardware_path),
})
contract_path = run / "benchmark_contract.json"
contract_path.write_text(
    json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

{
  date -u +"utc=%Y-%m-%dT%H:%M:%SZ"
  git rev-parse HEAD 2>/dev/null || echo "git_commit=unavailable"
  git status --short 2>/dev/null || true
  python --version
  python -m pip freeze
  nvidia-smi
} > "$RUN/environment.txt"
```

随后执行真实采集：

```bash
cd /workspace/xh-202625
PYTHONPATH=/workspace/xh-202625-model/src \
python scripts/benchmark_10k_pipeline.py \
  --config "$RUN/resolved_config.yaml" \
  --benchmark-contract "$RUN/benchmark_contract.json" \
  --checkpoint-provenance "$RUN/checkpoint_provenance.json" \
  --image-manifest "$RUN/image_manifest.json" \
  --data-root "$DATA_ROOT" \
  --output-dir "$RUN/capture" \
  --expected-width 10000 \
  --expected-height 10000

cp "$RUN/capture/runtime_samples.jsonl" "$RUN/runtime_samples.jsonl"
cp "$RUN/capture/predictions_10k_low.json" "$RUN/predictions_10k_low.json"
sha256sum "$RUN/predictions_10k_low.json" \
  | tee "$RUN/predictions_10k_low.sha256"
```

采集入口会重新核验 manifest、resolved config、checkpoint provenance 以及
其中的 checkpoint lineage，逐张核验图像内容与来源类型，拒绝 measured 图
重复内容充数，并按冻结几何自动计算且强制 `tile_count=100`；独立审计入口还会
核验 `hardware.json` 与合同中的 SHA。它实际执行读图、切片、tile 构造、本地
批循环调用 GPU adapter、坐标恢复、两级融合和序列化。由于现有 adapter 将框
解码和 tile 内 NMS 包在 `predict()` 内，`model` 阶段包含这部分
GPU/adapter 后处理；公共输出合法性检查和计数属于 `tile_postprocess`，不重复
计入 `model`。报告中必须按此口径解释。

采集器先在结果目录同级建立 staging 目录，13 次运行全部成功并闭环校验后才
原子发布为 `capture/`；任一失败会清理 staging，不得留下可被误认为完整结果的
半成品目录。

### 6.1 采集后完整性检查

```bash
test -s "$RUN/runtime_samples.jsonl"
test -s "$RUN/predictions_10k_low.json"
test "$(wc -l < "$RUN/runtime_samples.jsonl")" -eq 13
```

行数必须等于 `warmup_runs + minimum_measured_runs`。不得手工删除慢样本，
也不得把手写的模拟时间作为正式输入。

## 7. 计时口径

同时报告：

- 各阶段 p50/p95/max；
- model-only p50/p95/max；
- `total_after_read` p50/p95/max；
- 包含读盘的 `wall_with_read` p50/p95/max；
- raw/fused proposal 总数；
- peak VRAM。

官方文字要求是“单幅图读取完成到输出不超过 20 秒”，因此硬门禁使用：

```text
每一次 measured run 的 total_after_read <= 20.0 s
```

不是只要求均值或 p95 小于 20 秒。运行：

```bash
cd /workspace/xh-202625
set +e
PYTHONPATH=src python scripts/audit_10k_runtime.py \
  --input "$RUN/runtime_samples.jsonl" \
  --hardware "$RUN/hardware.json" \
  --benchmark-contract "$RUN/benchmark_contract.json" \
  --output "$RUN/runtime_summary.json" \
  --expected-width 10000 \
  --expected-height 10000 \
  --minimum-measured-runs 10 \
  --maximum-after-read-seconds 20
AUDIT_RC=$?
set -e
test "$AUDIT_RC" -eq 0 -o "$AUDIT_RC" -eq 2
```

退出码 `0` 表示工程时限通过；`2` 表示实验正常完成但至少一幅超时；`1`
表示合同或数据错误。只有代码注册过的 `real_official` 输入、
无其他 GPU 计算进程和非工程 checkpoint 同时成立，工具才会标记
为可作正式声明。GPU 型号只记录于证据中，不作为通过/失败条件。

无论时限通过或科学失败，都保留完整记录并打包；大型 checkpoint 不重复打包：

```bash
tar -czf "$RUN-return-no-checkpoint.tar.gz" -C "$(dirname "$RUN")" "$(basename "$RUN")"
sha256sum "$RUN-return-no-checkpoint.tar.gz" \
  | tee "$RUN-return-no-checkpoint.tar.gz.sha256"
exit "$AUDIT_RC"
```

## 8. 最终交付

```text
E-10K/
├── model_asset_env_verification.json
├── model_asset_env_lock.json
├── resolved_config.yaml
├── benchmark_contract.json
├── checkpoint_provenance.json
├── hardware.json
├── image_manifest.json
├── runtime_samples.jsonl
├── runtime_summary.json
├── predictions_10k_low.json
├── predictions_10k_low.sha256
├── environment.txt
└── capture/
    ├── runtime_samples.jsonl
    ├── predictions_10k_low.json
    └── run_000_predictions.json ... run_012_predictions.json
```

必须分别给出 M1、M3（若运行）及最终组合系统的结果。不能用 model-forward
时间代替完整流水线，也不能删除超时 run。任务未创建 `logs/` 或
`failure_notes.md` 时不得把它们虚列为必交付；失败原因记录在完整终端日志与
服务器最终回报中。
