# P03-FORMAL-CV3-V2：tight-224 ConvNeXt 正式三折复验

## 0. 唯一目标

只运行 `tight-224 / ConvNeXt-Tiny / ImageNet V1 / full fine-tune /
natural / seed=42 / folds 0,1,2`。不运行 linear probe、336、context、
jitter、sqrt-inverse、多 seed 或新模型。

固定输入：

```text
repo         /workspace/xh-202625
data         /workspace/data
exploratory  F00 p02_manifest_path.txt 登记的精确 SHA 副本
formal dir   /workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop
formal csv   /workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv
weights      /workspace/pretrained/convnext_tiny-983f1562.pth
venv         /workspace/venvs/p03-cu121
results      /workspace/results/P03-FORMAL-CV3-V2
```

冻结 SHA：

```text
CV3       27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331
formal    a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128
P0-2      f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e
weights   983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d
```

## 1. 环境与代码门禁

```bash
set -euo pipefail
cd /workspace/xh-202625
source /workspace/venvs/p03-cu121/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
ROOT=/workspace/results/P03-FORMAL-CV3-V2
P02_PATH_REGISTER=/workspace/results/FORMAL-CV3-CROP-TASK-01/p02_manifest_path.txt
test -s "$P02_PATH_REGISTER" || {
  echo "waiting_for_FORMAL_CV3_CROP_TASK_01: P0-2 路径登记缺失" >&2
  exit 2
}
EXP="$(cat "$P02_PATH_REGISTER")"
FORMAL_DIR=/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop
FORMAL="$FORMAL_DIR/formal_crop_manifest.csv"
mkdir -p "$ROOT/logs"

# 本阶段公共实现与冻结配置必须逐项匹配；不得只凭 Git commit 近似判断。
sha256sum -c docs/server/CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/formal-stage-code-sha256.log"

{
  date -Is
  git rev-parse HEAD
  git status --short
  nvidia-smi
  python --version
  python -c "import torch,torchvision; print(torch.__version__,torchvision.__version__,torch.version.cuda)"
  df -h /workspace
} 2>&1 | tee "$ROOT/system_preflight.txt"

PYTHONPATH=src pytest -q \
  tests/test_formal_cv3.py tests/test_formal_crop.py \
  tests/test_p03_p04_formal_replay.py \
  tests/test_p03_training_utils.py tests/test_p03_summary_cli.py \
  2>&1 | tee "$ROOT/logs/pytest.log"

ruff check \
  src/rsdet/data/formal_cv3.py src/rsdet/analysis/formal_crop.py \
  src/rsdet/analysis/formal_replay.py \
  scripts/audit_p03_p04_formal_inputs.py \
  scripts/freeze_p03_formal_config.py scripts/train_crop_classifier.py \
  scripts/summarize_p03_p04_formal.py \
  tests/test_formal_cv3.py tests/test_formal_crop.py \
  tests/test_p03_p04_formal_replay.py \
  2>&1 | tee "$ROOT/logs/ruff.log"
```

任一失败立即停止。保留输出，不自行修服务器工作树后续跑。

## 2. 验收并消费 F00 的唯一 formal crop

P03 不生成 formal crop。它只允许消费公共 F00 的 `run-a` 不可变副本；
`run-b` 仅用于确定性证明，严禁训练消费。若 `run-a` 缺失、不完整或 SHA
不符，状态应为 `waiting_for_FORMAL_CV3_CROP_TASK_01` 并停止：

```bash
EXPECTED_FORMAL_SHA=a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128
FORMAL_AUDIT="$FORMAL_DIR/formal_crop_audit.json"
FORMAL_CONFIG="$FORMAL_DIR/resolved_config.yaml"

test "$FORMAL" = \
  "/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv"
test -f "$FORMAL" && test -f "$FORMAL_AUDIT" && test -f "$FORMAL_CONFIG" || {
  echo "waiting_for_FORMAL_CV3_CROP_TASK_01: run-a formal artifact 缺失或不完整" \
    | tee "$ROOT/logs/formal-upstream-waiting.log" >&2
  exit 2
}

sha256sum "$FORMAL" | tee "$ROOT/formal_crop_sha256.txt"
test "$(sha256sum "$FORMAL" | cut -d' ' -f1)" = "$EXPECTED_FORMAL_SHA"
test "$(sha256sum "$EXP" | cut -d' ' -f1)" = \
  "f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e"
printf 'F00_RUN_A_VERIFIED_AND_CONSUMED\n' \
  | tee "$ROOT/logs/formal-upstream-accepted.log"
```

必须确认 `formal_crop_audit.json`：

```text
status=formal_crop_manifest_v2_ready
formal_crop_admission=true
rows=62799
annotations=20933
pixels_read=0
fold objects=7350/7179/6404
all overlap counts=0
```

## 3. 独立 formal 输入门禁与运行配置冻结

```bash
PYTHONPATH=src python scripts/audit_p03_p04_formal_inputs.py \
  --formal-manifest "$FORMAL" \
  --exploratory-manifest "$EXP" \
  --cv3-manifest data/splits/cv3_airport_proxy_k60_v2.json \
  --output "$ROOT/formal_input_audit.json" \
  2>&1 | tee "$ROOT/logs/formal-input-audit.log"

PYTHONPATH=src python scripts/freeze_p03_formal_config.py \
  --template configs/experiments/p03_formal_cv3_v2.yaml \
  --input-audit "$ROOT/formal_input_audit.json" \
  --output "$ROOT/p03_formal_resolved.yaml" \
  2>&1 | tee "$ROOT/logs/freeze-config.log"

PYTHONPATH=src python scripts/check_p03_environment.py \
  --manifest "$FORMAL" \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --expected-manifest-sha256 a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128 \
  --verify-source-count 4481 \
  --output "$ROOT/environment_check.json" \
  2>&1 | tee "$ROOT/logs/environment-check.log"
```

`formal_input_audit.status` 必须为 `formal_replay_inputs_ready`。本步骤和第 2 节
是两套独立读取逻辑，不能只保留其中一个。

## 4. smoke

```bash
PYTHONPATH=src python scripts/train_crop_classifier.py \
  --config "$ROOT/p03_formal_resolved.yaml" \
  --manifest "$FORMAL" \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --output-dir "$ROOT/smoke-fold0" \
  --fold 0 --policy tight --resolution 224 \
  --regime fine_tune --sampler natural --seed 42 \
  --smoke --overwrite \
  2>&1 | tee "$ROOT/logs/smoke.log"
```

检查 10 项训练产物、无 NaN/OOM、全参数可训练。smoke 分数不得进入汇总。
smoke 或正式 run 一旦 OOM，保留产物并停止；正式合同不允许在同一任务 ID
下减小 batch 后继续。

## 5. 三个正式 run

```bash
for FOLD in 0 1 2; do
  RUN="$ROOT/ft-tight-224-fold${FOLD}"
  test ! -e "$RUN" || {
    test -f "$RUN/run_summary.json" && continue
    echo "发现不完整 run，停止而非覆盖: $RUN" >&2
    exit 2
  }
  PYTHONPATH=src python scripts/train_crop_classifier.py \
    --config "$ROOT/p03_formal_resolved.yaml" \
    --manifest "$FORMAL" \
    --data-root /workspace/data \
    --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
    --output-dir "$RUN" \
    --fold "$FOLD" --policy tight --resolution 224 \
    --regime fine_tune --sampler natural --seed 42 \
    2>&1 | tee "$ROOT/logs/fold${FOLD}.log"
  test "${PIPESTATUS[0]}" -eq 0 || exit 1
done
```

不得传入覆盖 epoch、LR、batch、sampler 或 checkpoint 的额外参数。每折从
同一 ImageNet 权重独立初始化。

## 6. 正式汇总

```bash
PYTHONPATH=src python scripts/summarize_p03_p04_formal.py \
  --stage p03 \
  --runs-root "$ROOT" \
  --input-audit "$ROOT/formal_input_audit.json" \
  --output "$ROOT/formal_summary.json" \
  2>&1 | tee "$ROOT/logs/formal-summary.log"
```

汇总必须：

- 恰有 3 个非 smoke run；
- 每折 `n_val=7350/7179/6404`；
- 三折 `predictions.csv` 的 UID/crop/class 必须逐行对齐 formal manifest，
  无重复、遗漏或跨折，并与保存 logits 的 labels/argmax 一致；
- 每折保存指标必须能由 logits 独立复算，输出 mean±sample std 和恰好
  20,933 个不同对象的 pooled OOF；
- resolved runtime 必须仍是固定 30 epoch、batch 96、workers 8、全参数
  fine-tune，`checkpoint_selection=fixed_epoch_last`，ImageNet 权重 SHA
  固定；held-out fold 不参与逐 epoch 选 checkpoint 或 early stop，只在训练
  完成后评估一次；
- 输出 25 类 recall/support、三大类、固定 9/8/8 support tier；
- 单列 TU-160 三折验证 support `352/8/1` 与训练 support `9/353/360`。

## 7. 回传与保留

```bash
find "$ROOT"/ft-tight-224-fold*/final_checkpoint.pt -type f -print0 \
  | sort -z | xargs -0 sha256sum > "$ROOT/CHECKPOINTS_SHA256.txt"

cd /workspace/results
tar --exclude='final_checkpoint.pt' \
  -czf P03-FORMAL-CV3-V2-results-no-checkpoints.tar.gz \
  P03-FORMAL-CV3-V2
sha256sum P03-FORMAL-CV3-V2-results-no-checkpoints.tar.gz
```

服务器保留三个 checkpoint，直到本地验收完成。最终回报需含环境、Git、
formal 三个 SHA、3/3 run、逐折与 pooled 指标、TU-160 压力折、耗时、
显存、checkpoint 和回传包 SHA。
