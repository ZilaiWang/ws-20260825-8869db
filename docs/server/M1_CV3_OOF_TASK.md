# M1 服务器任务：YOLO26-s/1024 正式 CV3 OOF

依赖：`CV3_OOF_COMMON_CONTRACT.md`  
状态：可执行  
模型角色：快速主检测器、P05/P06 的第一真实 OOF 来源

## 1. 唯一运行点

```text
model: YOLO26-s
family: yolo
input: 1024
seed: 42
initialization: official yolo26s.pt, same file/SHA for all folds
foundation epochs: 160（固定跑满）
checkpoint selection: last
validation during training: false
patience: 0
optimizer: AdamW
lr0: 0.002
weight_decay: 0.0005
low prediction threshold: 0.001
max detections: 500
```

只运行 `foundation`。不恢复未完成的 rare-rebalance，不接 HPR，不调
25 类阈值。

模板：

- `configs/experiments/m1_yolo26s_1024_cv3_oof.template.yaml`
- `configs/experiments/m1_yolo26s_1024_cv3_oof_infer.template.yaml`

## 2. 前置门禁

1. 主仓库本任务列出的专项测试与 Ruff 门禁通过；
2. 同级 `xh-202625-model` 本任务列出的训练、adapter、tile fusion 专项
   测试与 Ruff 门禁通过；
3. 数据 4481 张、25 类、正式 manifest SHA 正确；
4. `yolo26s.pt` 是服务器上实际文件，记录来源、大小和 SHA；
5. GPU、driver、Python、PyTorch、CUDA、Ultralytics 写入
   `environment.txt`；
6. 三折各跑一次 `scripts/train.py --dry-run`，确认 train/val 数量与
   split view 一致。
7. D00 已完成，正式数据锁 SHA 为
   `03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a`；
   每折训练进程启动前都要重新全量 verify。

held-out fold 不参与逐轮验证、early stop、checkpoint 选择或训练期调参。
每折固定跑满 160 epoch，只选择 `foundation/weights/last.pt`，随后使用该固定
checkpoint 对 held-out fold 做一次正式低阈值 OOF 推理。需要注意：
Ultralytics 即使设置 `val=false`，仍可能在最终 epoch 运行一次框架内部终局
验证；该读出只作运行记录，不得改变权重、选择 checkpoint、调整任何参数，
也不得当作本项目正式评估结果。

## 3. 准备三折

```bash
cd /workspace/xh-202625
set -euo pipefail
source /workspace/venvs/cv3-model-cu121/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

ASSET_ROOT=/workspace/cv3-model-assets
ASSET_LOCK="$ASSET_ROOT/MODEL_ASSET_ENV_LOCK.json"
ASSET_SPEC=/workspace/xh-202625/configs/experiments/cv3_model_asset_env.json
ASSET_VERIFY_REPORT="$(mktemp /tmp/M1-model-asset-verify.XXXXXX.json)"
PRETRAINED="$ASSET_ROOT/yolo26s.pt"
PRETRAINED_SIZE=20422725
PRETRAINED_SHA=646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
DATA_LOCK=/workspace/formal-detection-data/FORMAL_DETECTION_DATA_LOCK.json
DATA_LOCK_SHA=03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a
RUN_ROOT=/workspace/results/M1-CV3-OOF

# 两个仓库的关键实现必须与本任务冻结指纹完全一致。
sha256sum -c docs/server/CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt
sha256sum -c docs/server/XH_MODEL_INTEGRATION_CODE_SHA256.txt

python scripts/lock_cv3_model_assets.py verify \
  --config "$ASSET_SPEC" \
  --asset-root "$ASSET_ROOT" \
  --expected-gpu "NVIDIA GeForce RTX 4080 SUPER" \
  --lock "$ASSET_LOCK" \
  --report "$ASSET_VERIFY_REPORT"
test "$(stat -c %s "$PRETRAINED")" = "$PRETRAINED_SIZE"
echo "$PRETRAINED_SHA  $PRETRAINED" | sha256sum -c -
echo "$DATA_LOCK_SHA  $DATA_LOCK" | sha256sum -c -

python - <<'PY'
import torch
import ultralytics

assert torch.cuda.is_available(), "M1 正式训练要求 CUDA"
assert ultralytics.__version__ == "8.4.103", (
    "M1 正式适配器锁定 ultralytics==8.4.103，"
    f"当前为 {ultralytics.__version__}"
)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("ultralytics", ultralytics.__version__)
print("gpu", torch.cuda.get_device_name(0))
PY

python -m pytest -q tests/test_cv3_oof.py
python -m ruff check \
  src/rsdet/experiments/cv3_oof.py \
  scripts/prepare_cv3_oof.py scripts/materialize_cv3_oof_config.py \
  scripts/finalize_cv3_oof_fold.py scripts/audit_cv3_oof.py \
  tests/test_cv3_oof.py

cd /workspace/xh-202625-model
PYTHONPATH=src python -m pytest -q \
  tests/test_trainer_contract.py \
  tests/test_ultralytics_adapter.py \
  tests/test_inference_pipeline.py \
  tests/test_tile_fusion.py
PYTHONPATH=src python -m ruff check \
  scripts/train.py scripts/infer.py src/rsdet tests/test_ultralytics_adapter.py \
  tests/test_trainer_contract.py tests/test_inference_pipeline.py \
  tests/test_tile_fusion.py

cd /workspace/xh-202625

test ! -e "$RUN_ROOT" || {
  echo "M1 结果目录已存在，禁止覆盖或混入旧结果: $RUN_ROOT" >&2
  exit 2
}

python scripts/prepare_cv3_oof.py \
  --manifest data/splits/cv3_airport_proxy_k60_v2.json \
  --manifest-sha256 27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331 \
  --output-dir "$RUN_ROOT" \
  --model-key M1 \
  --model-family yolo \
  --model-name yolo26s \
  --seed 42 \
  --input-size 1024 \
  --foundation-epochs 160 \
  --low-score-threshold 0.001 \
  --max-detections 500 \
  --pretrained-weight "$PRETRAINED" \
  --pretrained-weight-sha256 "$PRETRAINED_SHA" \
  --detection-data-lock "$DATA_LOCK" \
  --detection-data-lock-sha256 "$DATA_LOCK_SHA"
cp "$ASSET_VERIFY_REPORT" "$RUN_ROOT/model_asset_env_verification.json"
rm -f "$ASSET_VERIFY_REPORT"
```

当前任务采用不可变整批语义，不支持同一任务 ID 的折间续跑。若在任一 fold
中断，保留原目录并回报 `blocked_partial_run`; 由负责人登记新任务后缀，在新
`RUN_ROOT` 从三折计划重新开始。不得因为已有某折 checkpoint 就删除部分目录、
覆盖或从该 checkpoint resume。此约束牺牲少量重跑成本，以避免把两次环境或
代码状态混入一份正式 OOF。

## 4. 三折完整执行、冻结与汇总

第 3 节全部通过后，严格串行运行三个 fold。`--dry-run` 只作数据/config smoke；其摘要单独
保存，正式训练会重新写入 `dry_run=false` 的最终摘要：

```bash
set -euo pipefail
P02="$(
  cat /workspace/results/CV3-DETECTION-DATA-LOCK-TASK-00/p02_path.txt
)"
echo \
  "f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e  $P02" \
  | sha256sum -c -

for FOLD in 0 1 2; do
  FOLD_DIR="$RUN_ROOT/fold_$FOLD"
  VIEW="$FOLD_DIR/split_view.json"

  cd /workspace/xh-202625
  mkdir -p "$FOLD_DIR/input-gates"
  PYTHONPATH=src python scripts/lock_formal_detection_data.py verify \
    --config configs/experiments/formal_detection_data_lock.json \
    --data-root /workspace/data \
    --cv3-manifest data/splits/cv3_airport_proxy_k60_v2.json \
    --p02-manifest "$P02" \
    --formal-crop-manifest \
      /workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv \
    --lock "$DATA_LOCK" \
    --expected-lock-sha256 "$DATA_LOCK_SHA" \
    --report \
      "$FOLD_DIR/input-gates/detection_data_lock_verification.json" \
    2>&1 | tee "$FOLD_DIR/detection-data-lock-verify.log"
  test "$(
    python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$FOLD_DIR/input-gates/detection_data_lock_verification.json"
  )" = pass

  PYTHONPATH=src python scripts/materialize_cv3_oof_config.py \
    --template configs/experiments/m1_yolo26s_1024_cv3_oof.template.yaml \
    --output "$FOLD_DIR/train_request.yaml" \
    --fold "$FOLD" \
    --data-root /workspace/data \
    --split-view "$VIEW" \
    --fold-output-dir "$FOLD_DIR/training" \
    --pretrained-weight "$PRETRAINED"

  cd /workspace/xh-202625-model
  PYTHONPATH=src python scripts/train.py \
    --config "$FOLD_DIR/train_request.yaml" \
    --dry-run 2>&1 | tee "$FOLD_DIR/train_dry_run.log"
  cp "$FOLD_DIR/training/train_summary.json" \
    "$FOLD_DIR/train_summary.dry_run.json"

  # 正式命令严禁增加 --resume。
  PYTHONPATH=src python scripts/train.py \
    --config "$FOLD_DIR/train_request.yaml" \
    2>&1 | tee "$FOLD_DIR/train.log"
  CHECKPOINT="$FOLD_DIR/training/runs/foundation/weights/last.pt"
  test -s "$CHECKPOINT"

  cd /workspace/xh-202625
  PYTHONPATH=src python scripts/materialize_cv3_oof_config.py \
    --template configs/experiments/m1_yolo26s_1024_cv3_oof_infer.template.yaml \
    --output "$FOLD_DIR/resolved_infer.yaml" \
    --fold "$FOLD" \
    --data-root /workspace/data \
    --split-view "$VIEW" \
    --fold-output-dir "$FOLD_DIR" \
    --checkpoint "$CHECKPOINT"

  cd /workspace/xh-202625-model
  PYTHONPATH=src python scripts/infer.py \
    --config "$FOLD_DIR/resolved_infer.yaml" \
    2>&1 | tee "$FOLD_DIR/infer.log"

  {
    date -u +"utc=%Y-%m-%dT%H:%M:%SZ"
    git rev-parse HEAD 2>/dev/null || echo "git_commit=unavailable"
    git status --short 2>/dev/null || true
    python --version
    python -m pip freeze
    nvidia-smi
  } > "$FOLD_DIR/environment.txt"

  cd /workspace/xh-202625
  PYTHONPATH=src python scripts/finalize_cv3_oof_fold.py \
    --plan "$RUN_ROOT/oof_run_plan.json" \
    --fold "$FOLD" \
    --train-config "$FOLD_DIR/training/resolved_config.yaml" \
    --train-summary "$FOLD_DIR/training/train_summary.json" \
    --infer-config "$FOLD_DIR/resolved_infer.yaml" \
    --environment "$FOLD_DIR/environment.txt" \
    --checkpoint "$CHECKPOINT" \
    --predictions "$FOLD_DIR/predictions_low.json" \
    --runtime "$FOLD_DIR/predictions_low.runtime.json" \
    --data-lock-verification \
      "$FOLD_DIR/input-gates/detection_data_lock_verification.json" \
    --output "$FOLD_DIR/fold_metadata.json"
done
```

正式 aggregate 一次性绑定 formal crop：

```bash
FORMAL_CROP=/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv
FORMAL_SHA=a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128
test "$(sha256sum "$FORMAL_CROP" | awk '{print $1}')" = "$FORMAL_SHA"

cd /workspace/xh-202625
PYTHONPATH=src python scripts/audit_cv3_oof.py \
  --manifest data/splits/cv3_airport_proxy_k60_v2.json \
  --manifest-sha256 27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331 \
  --plan "$RUN_ROOT/oof_run_plan.json" \
  --run-root "$RUN_ROOT" \
  --output-dir /workspace/results/M1-CV3-OOF-aggregate \
  --formal-crop-manifest "$FORMAL_CROP"

tar --exclude='*.pt' --exclude='prepared_data' -czf \
  /workspace/results/M1-CV3-OOF-return-no-checkpoints.tar.gz \
  -C /workspace/results M1-CV3-OOF M1-CV3-OOF-aggregate
sha256sum /workspace/results/M1-CV3-OOF-return-no-checkpoints.tar.gz \
  | tee /workspace/results/M1-CV3-OOF-return-no-checkpoints.tar.gz.sha256
```

## 5. 三折完成后的验收

- fold 0/1/2 预测图像数分别应为 1507/1613/1361；
- 三折 checkpoint 的输入初始化 SHA 都等于冻结常量
  `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`；
- 每折 checkpoint、训练 config、推理 config 和预测 SHA 齐全；
- 三折汇总 `image_count=4481`；
- 零预测图允许存在，但必须在 `oof_images.csv` 中有一行；
- 交付低阈值候选，不在服务器自行选择正式阈值。

本任务结束后立即进入 M1 错误分解和 P05/P06 准入判断；不能把训练日志中的
Ultralytics mAP 代替官方 Recall/FDR。

任一 fold 出现 OOM、非有限 loss 或 adapter 错误时，保留该 fold 目录并停止
整个任务；不得在同一任务 ID 下改 batch、epoch、尺寸或模型后继续。
