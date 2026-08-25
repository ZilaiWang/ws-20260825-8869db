# MAR20-GROUPING-TASK-00A：方法分列复核、calibration 包与三填充 DINO cache

## 0. 任务性质

这是 TASK-00 Phase A 的无损续跑，不重做 registry、不修改原始 Phase-A 产物。原因有二：

1. Phase A 在 `view audit` 后提前停止，原任务单中无需人工标签的 calibration pack 和 DINO cache 尚未执行；
2. 原 `manual_view_review.csv` 每图只有一组判定，无法区分 blur、local_mean、Telea。真实 contact sheets 已显示三种方法的残留/伪影行为明显不同，因此必须改用方法分列的 v2 模板。

本任务只完成无标签步骤，最终仍停在人工复核。服务器 AI 不得填写任何人工决定。

## 1. 路径

```text
repo          /workspace/xh-202625
root          /workspace/results/MAR20-GROUPING-TASK-00
MAR20         /workspace/inputs/MAR20
asset lock    /workspace/p04-assets/ASSET_LOCK.json
venv          /workspace/venvs/mar20-group-cu121
cache root    /workspace/mar20-group-cache
```

```bash
cd /workspace/xh-202625
set -o pipefail
ROOT=/workspace/results/MAR20-GROUPING-TASK-00
MAR20=/workspace/inputs/MAR20
ASSETS=/workspace/p04-assets/ASSET_LOCK.json
VENV=/workspace/venvs/mar20-group-cu121
CACHE=/workspace/mar20-group-cache
mkdir -p "$ROOT/logs" "$CACHE"
source "$VENV/bin/activate"
export PYTHONPATH=src XFORMERS_DISABLED=1
```

## 2. 代码和 Phase-A 输入门禁

```bash
sha256sum -c docs/server/MAR20_GROUPING_TASK_00A_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/task-00a-code-sha.log"

pytest -q tests/test_mar20_grouping_batch_a.py \
  2>&1 | tee "$ROOT/logs/task-00a-pytest.log"

ruff check \
  src/rsdet/grouping \
  scripts/build_mar20_view_review_template.py \
  scripts/audit_mar20_background_views.py \
  scripts/compile_mar20_view_review.py \
  scripts/build_mar20_calibration_review.py \
  scripts/compile_mar20_calibration_review.py \
  scripts/extract_mar20_place_features.py \
  scripts/analyze_mar20_descriptor_bakeoff.py \
  tests/test_mar20_grouping_batch_a.py \
  2>&1 | tee "$ROOT/logs/task-00a-ruff.log"
```

必须核验并记录：

```text
$ROOT/task_decision.json.status = waiting_for_manual_reviews
$ROOT/registry/image_registry.csv
  bcdc4b697532be0899b84abb7f218a0fc2fec2c08e5aaeb3460336eea7b7201d
$ROOT/registry/image_annotations.jsonl
  0a350c1d68e63dd134ca5f4b6a1f89c16f9a4d457f34598b72f2c1eface650f4
$ROOT/view-audit/view_audit.csv
  446b4a9c0baa2fc1c675eb38d6b54851b0d7a6e7a76f82d070074af5746f2f14
```

任一不匹配立即停止为 `failed_phase_a_input_integrity`。原始 `view-audit/manual_view_review.csv` 保留，不删除、不填写、不作为后续输入。

## 3. 生成方法分列 v2 人工模板

```bash
python scripts/build_mar20_view_review_template.py \
  --view-audit "$ROOT/view-audit/view_audit.csv" \
  --output "$ROOT/view-audit/manual_view_review_v2.csv" \
  --primary-dilation 0.15 \
  --summary-output "$ROOT/view-audit/manual_view_review_v2_summary.json" \
  2>&1 | tee "$ROOT/logs/build-view-review-v2.log"
```

确定性锚点：

```text
rows                             120
background_tile_available        114
background_tile_unavailable      6
manual_view_review_v2.csv SHA    4a6ae591c0b5f2dfef60e4cc335e84c04d0e386c99869e300574ffebfee341df
```

v2 表分别记录：

```text
blur_aircraft_remnant, blur_inpaint_artifact
local_mean_aircraft_remnant, local_mean_inpaint_artifact
telea_aircraft_remnant, telea_inpaint_artifact
background_tile_available, background_tile_aircraft
```

`background_tile_available` 是机器预填字段，人工不得修改。没有 tile 的 6 张图，其 `background_tile_aircraft` 留空。

## 4. 生成 360-pair 盲化 calibration 包

```bash
python scripts/build_mar20_calibration_review.py \
  --registry "$ROOT/registry/image_registry.csv" \
  --annotations "$ROOT/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --output-dir "$ROOT/calibration-review" \
  --near-duplicate-json reports/data/near_duplicates_mar20.json \
  --pair-count 360 \
  --duplicate-fraction 0.08 \
  --dilation-ratio 0.15 \
  --fill-method telea \
  --cards-per-sheet 4 \
  2>&1 | tee "$ROOT/logs/build-calibration-review.log"
```

要求：360 个唯一 pair、29 个左右交换顺序盲重复、重复卡实际间距至少 30、约 672 个唯一节点。实际数和所有 SHA 写入回报。

盲评阶段：

- 回传 contact sheets 和空白 `manual_calibration_decisions.csv`；
- `blind_card_mapping.csv` 留在服务器封存，不得发给人工评审者打开；
- 服务器可以读取 `blind_calibration_node_uids.txt` 做无标签特征提取，但不能读取 route 后代替人工下结论。

## 5. DINOv2-B 真实 smoke

```bash
python scripts/extract_mar20_place_features.py \
  --registry "$ROOT/registry/image_registry.csv" \
  --annotations "$ROOT/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --asset-lock "$ASSETS" \
  --output-dir "$ROOT/dino-smoke-task-00a" \
  --encoder dinov2_vitb14 \
  --scope target_only \
  --node-list "$ROOT/calibration-review/blind_calibration_node_uids.txt" \
  --max-nodes 2 \
  --view-types original,masked_inpaint,background_tiles \
  --rotations 0 \
  --input-size 518 \
  --dilation-ratio 0.15 \
  --fill-method telea \
  --layers 9,10,11 \
  --gem-powers 2,3,4 \
  --batch-size 4 \
  --shard-size 64 \
  --compute-dtype float16 \
  --storage-dtype float16 \
  --device cuda \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/dino-smoke-task-00a.log"
```

必须得到 15 个有限特征并记录维度、VRAM、行数和输入视图；mock 不得用于此门禁。

## 6. 三填充 Round-A 无标签缓存

在同一冻结 calibration node list 上分别运行三个 cache。每份都包含 original、该填充 masked、background tiles；重复计算 common views 是可接受的，换取独立缓存、简单审计和不混写。

对 `METHOD=blur local_mean telea` 逐一执行：

```bash
OUT="$CACHE/dinov2b-calibration-round-a-${METHOD}-v1"
python scripts/extract_mar20_place_features.py \
  --registry "$ROOT/registry/image_registry.csv" \
  --annotations "$ROOT/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --asset-lock "$ASSETS" \
  --output-dir "$OUT" \
  --encoder dinov2_vitb14 \
  --scope target_only \
  --node-list "$ROOT/calibration-review/blind_calibration_node_uids.txt" \
  --view-types original,masked_inpaint,background_tiles \
  --rotations 0 \
  --input-size 518 \
  --dilation-ratio 0.15 \
  --fill-method "$METHOD" \
  --tile-size 224 \
  --tile-stride 112 \
  --tile-valid-fraction 0.95 \
  --max-tiles 8 \
  --layers 9,10,11 \
  --gem-powers 2,3,4 \
  --batch-size 16 \
  --shard-size 128 \
  --compute-dtype float16 \
  --storage-dtype float16 \
  --device cuda \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/extract-round-a-${METHOD}.log"
```

服务器 AI 应使用明确的 shell loop 或逐条替换 `METHOD`，不能把上述占位符原样执行。若 batch=16 OOM，只允许新建带 `-b8`/`-b4` 后缀的独立 cache；禁止混写原 cache。每个成功 cache 用完全相同命令再跑一次，必须全部 SKIP、`computed_rows=0`、index SHA 不变。

本节不读取人工标签，不运行 descriptor 选择，因此不会造成调参泄漏。

## 7. 停止状态与回传

完成以上无标签步骤后，写：

```json
{
  "status": "waiting_for_method_specific_view_and_pair_reviews",
  "formal_grouping_admission": false,
  "descriptor_selection_admission": false,
  "required_manual_inputs": [
    "manual_view_review_v2.csv",
    "manual_calibration_decisions.csv"
  ]
}
```

回传包包含：

- `manual_view_review_v2.csv` 及 summary；
- 原 15 张 view contact sheets 可只通过 SHA 引用，不重复打包；
- calibration contact sheets、空白 decisions、summary、node list；
- DINO smoke 全部小型产物；
- 三 cache 的 meta/index/sidecar/extraction summary，不含 NPZ；
- 新日志、最终 task decision、包 SHA。

三份大 cache 和封存 mapping 留在服务器。不得删除或改写 Phase-A registry/view audit，也不得修改 P03～P07。

## 8. 回报格式

1. TASK-00A 状态和前置 Phase-A SHA；
2. v2 模板 120/114/6 计数及 SHA；
3. calibration unique pairs、重复卡、最小间距、唯一节点；
4. DINO smoke 15 特征维度、数值有限性、VRAM；
5. blur/local_mean/telea 三 cache 的行数、shard、fingerprint、大小、速度、耗时；
6. 每份 resume 是否 `computed_rows=0`；
7. 任何 OOM 和唯一 batch 变更；
8. 回传包路径、大小和 SHA；
9. 明确声明未填写人工决定、未形成 group、未自动 union。
