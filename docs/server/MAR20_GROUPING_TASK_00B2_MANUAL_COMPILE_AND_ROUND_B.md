# MAR20-GROUPING-TASK-00B2：人工证据编译与 Round-B

## 0. 任务目标与边界

本任务承接服务器已经完成的 `MAR20-GROUPING-TASK-00B1`。不重新提取 masked GeM、VLAD 或 PCA，不重新挖候选，不修改人工表。任务仅做：

1. 核验并接收两份已经冻结的人工审核表；
2. 补做 00B1 遗漏的 VLAD 全量 resume 审计；
3. 用 00B1 admitted summary 编译 patch-mask 门禁；
4. 解封既有 mapping，编译 v1.2 标定集；
5. 运行九路 Round-B 与唯一终止决策。

本任务无论结果如何，`formal_grouping_admission` 都必须保持 `false`。只有 `ready_for_task01_retrieval_and_geometry` 允许进入后续检索加几何建边，仍不代表已经得到最终机场 group_id。

## 1. 路径和人工输入

将 `MAR20-00B2-manual-input.tar.gz` 上传到服务器并解压，使下面三个文件存在：

```text
/workspace/inputs/MAR20-GROUPING-TASK-00B2/manual_patch_mask_review.csv
/workspace/inputs/MAR20-GROUPING-TASK-00B2/manual_enriched_decisions.csv
/workspace/inputs/MAR20-GROUPING-TASK-00B2/MANUAL_INPUT_SHA256.txt
```

然后执行：

```bash
cd /workspace/xh-202625
set -o pipefail
ROOT=/workspace/results/MAR20-GROUPING-TASK-00B1
OLD=/workspace/results/MAR20-GROUPING-TASK-00B
PREV=/workspace/results/MAR20-GROUPING-TASK-00
MANUAL=/workspace/inputs/MAR20-GROUPING-TASK-00B2
CACHE=/workspace/mar20-group-cache
VLAD="$CACHE/dinov2b-vlad-full-v1p2"
VLADPCA="$CACHE/dinov2b-vlad-pca512-full-v1p2"
CODEBOOK="$CACHE/dinov2b-vlad-codebooks-v1p2"
MAR20=/workspace/inputs/MAR20
ASSETS=/workspace/p04-assets/ASSET_LOCK.json
VENV=/workspace/venvs/mar20-group-cu121
mkdir -p "$ROOT/logs"
source "$VENV/bin/activate"
export PYTHONPATH=src XFORMERS_DISABLED=1
```

如果服务器原任务路径不同，只能整体修正变量并在回报中记录；不得混合不同任务目录。

## 2. 不可跳过的前置门禁

```bash
sha256sum -c docs/server/MAR20_GROUPING_TASK_00B_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/00b2-code-sha-base.log"
sha256sum -c docs/server/MAR20_GROUPING_TASK_00B1_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/00b2-code-sha-amendment.log"

cd "$MANUAL"
sha256sum -c MANUAL_INPUT_SHA256.txt \
  2>&1 | tee "$ROOT/logs/00b2-manual-input-sha.log"
cd /workspace/xh-202625

test "$(sha256sum "$MANUAL/manual_patch_mask_review.csv" | awk '{print $1}')" = \
  adba9dca47494520da3b62ea1687a3db4bb9637d4ee19e0a0e331d51d87bac6a
test "$(sha256sum "$MANUAL/manual_enriched_decisions.csv" | awk '{print $1}')" = \
  c664d1dedc8be26911ccf072bff5d918742e4a204c38b141dd0b06dc67a01eff

test -s "$OLD/patch-mask-audit/patch_mask_audit.csv"
test -s "$ROOT/low-valid-review/patch_mask_audit_summary_admitted.json"
test -s "$ROOT/enriched-review/blind_card_mapping.csv"
test -s "$PREV/calibration-compiled/calibration_pairs.csv"
test -s "$VLAD/cache/index.json"
test -s "$VLADPCA/cache/index.json"

python - <<'PY' | tee "$ROOT/logs/00b2-prerequisite-status.json"
import json
from pathlib import Path
root = Path('/workspace/results/MAR20-GROUPING-TASK-00B1')
phase = json.loads((root / 'task_decision.json').read_text())
review = json.loads((root / 'low-valid-review/continuation_decision.json').read_text())
admitted = json.loads((root / 'low-valid-review/patch_mask_audit_summary_admitted.json').read_text())
assert phase['status'] == 'waiting_for_patch_mask_and_enriched_pair_reviews'
assert phase['formal_grouping_admission'] is False
assert review['continuation_admission'] is True
assert admitted['automatic_geometry_gate'] == 'pass'
print(json.dumps({'phase': phase, 'low_valid_review': review, 'admitted_summary': admitted}, indent=2))
PY
```

任一门禁失败必须停止，不允许重新生成一套人工表、mapping 或 cache。

## 3. 补做 VLAD resume 审计

完全复用 00B1 的提取命令：

```bash
python scripts/extract_mar20_vlad_features.py \
  --registry "$PREV/registry/image_registry.csv" \
  --annotations "$PREV/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --codebook-manifest "$CODEBOOK/codebook_manifest.json" \
  --codebook-dir "$CODEBOOK" \
  --asset-lock "$ASSETS" \
  --output-dir "$VLAD" \
  --encoder dinov2_vitb14 \
  --scope full_bridge \
  --rotations 0,90,180,270 \
  --input-size 518 \
  --patch-size 14 \
  --dilation-ratio 0.15 \
  --maximum-patch-foreground-fraction 0.20 \
  --batch-size 8 \
  --shard-size 64 \
  --device cuda \
  --compute-dtype float16 \
  --storage-dtype float16 \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/00b2-vlad-resume.log"
```

门禁要求：`computed=0`、`skipped_shards=241`、`skipped_rows=15368`，cache 指纹和 index SHA 不变。任何实际重算、缺 shard、指纹变化或 nonfinite 都必须停止。

## 4. 编译 patch-mask

注意：必须引用 00B1 admitted summary，禁止引用 `$OLD/patch-mask-audit/patch_mask_audit_summary.json`，后者保留原始科学失败记录。

```bash
python scripts/compile_mar20_patch_mask_review.py \
  --audit "$OLD/patch-mask-audit/patch_mask_audit.csv" \
  --audit-summary "$ROOT/low-valid-review/patch_mask_audit_summary_admitted.json" \
  --review "$MANUAL/manual_patch_mask_review.csv" \
  --output "$ROOT/patch-mask-decision.json" \
  --minimum-valid-rate 0.95 \
  --minimum-aircraft-coverage-rate 0.95 \
  --maximum-excessive-loss-rate 0.10 \
  2>&1 | tee "$ROOT/logs/00b2-compile-patch-mask.log"

test "$(sha256sum "$ROOT/patch-mask-decision.json" | awk '{print $1}')" = \
  62a3ae8705744b8a4f096d25d96be7799bfac77c078c76fde029f01c9846d220
```

预期 `status=pass`、120 个节点、三档 coverage=1.0、excessive loss=0、`formal_patch_mask_admission=true`。

## 5. 编译 v1.2 标定集

```bash
python scripts/compile_mar20_enriched_calibration.py \
  --prior-pairs "$PREV/calibration-compiled/calibration_pairs.csv" \
  --mapping "$ROOT/enriched-review/blind_card_mapping.csv" \
  --decisions "$MANUAL/manual_enriched_decisions.csv" \
  --output-dir "$ROOT/calibration-v1p2" \
  --minimum-positive-pairs 30 \
  --recommended-positive-pairs 60 \
  --minimum-heldout-positive-pairs 5 \
  --recommended-heldout-positive-pairs 15 \
  --minimum-calibration-positive-pairs 5 \
  --minimum-repeat-agreement 0.90 \
  --heldout-fraction 0.25 \
  2>&1 | tee "$ROOT/logs/00b2-compile-enriched-calibration.log"

test "$(sha256sum "$ROOT/calibration-v1p2/calibration_pairs_v1p2.csv" | awk '{print $1}')" = \
  9d9e2a4686f7cd498c7e37a38cbae8812ab7820418e6e2dcfc7dc7250b3faa60
test "$(sha256sum "$ROOT/calibration-v1p2/calibration_compile_summary_v1p2.json" | awk '{print $1}')" = \
  87450b7930c4c3a9fb66f98ceed10315f77e16fa5ee680ae5305a32c95f0af7b
```

预期：600 pair、248 严格正例、00B1 新增 229、calibration 186、held-out 62、repeat 24/一致率1.0、strict component 194、跨 split component 0、无 conflict，状态为 `pass_recommended_evidence_target`。

## 6. Round-B 九路分析

```bash
set +e
python scripts/analyze_mar20_round_b.py \
  --routes-json configs/grouping/mar20_00b_round_b_routes_server.json \
  --calibration-pairs "$ROOT/calibration-v1p2/calibration_pairs_v1p2.csv" \
  --calibration-summary "$ROOT/calibration-v1p2/calibration_compile_summary_v1p2.json" \
  --patch-mask-decision "$ROOT/patch-mask-decision.json" \
  --output-dir "$ROOT/round-b" \
  --k-values 20,50,100 \
  --heldout-recall-target 0.95 \
  --device cuda \
  --node-batch-size 64 \
  2>&1 | tee "$ROOT/logs/00b2-analyze-round-b.log"
ROUND_B_EXIT=${PIPESTATUS[0]}
set -e
test "$ROUND_B_EXIT" -eq 0 -o "$ROUND_B_EXIT" -eq 2

python scripts/compile_mar20_00b_decision.py \
  --patch-mask-decision "$ROOT/patch-mask-decision.json" \
  --calibration-summary "$ROOT/calibration-v1p2/calibration_compile_summary_v1p2.json" \
  --round-b-decision "$ROOT/round-b/round_b_decision.json" \
  --output "$ROOT/task_decision.json" \
  2>&1 | tee "$ROOT/logs/00b2-compile-task-decision.log"
```

允许的终止状态只有：

```text
ready_for_task01_retrieval_and_geometry
needs_second_enrichment_batch
complete_00b_patch_mask_no_admission
complete_00b_calibration_no_admission
complete_00b_retrieval_no_admission
```

服务器不得自行放宽 held-out recall、Wilson 区间、negative rate 或候选负载门槛。

## 7. 回传与汇报

回传包应包含：

- `patch-mask-decision.json`；
- `calibration-v1p2/` 全部小型文件；
- `round-b/` 全部 CSV、JSON；
- 更新后的唯一 `task_decision.json`；
- 00B2 的代码、人工输入、resume、编译和 Round-B 日志；
- cache/codebook/PCA 的路径、大小、index SHA、指纹清单；
- 不含任何大 NPZ、权重或原图。

最终回报必须给出：

1. 状态与是否科学停止；
2. Git commit/dirty、两份 code SHA 门禁；
3. 两份人工输入 SHA；
4. VLAD resume 的 computed/skipped/shard/row/指纹；
5. patch-mask 五项人工指标；
6. v1.2 pair/positive/calibration/held-out/component/repeat/conflict；
7. 九条单路及所有预注册并集的 calibration 与 held-out Recall@20/50/100、Wilson 区间、negative rate、候选负载；
8. calibration-only 选中的路由、随后一次性打开的 held-out 结果；
9. 唯一 task decision 和 next action；
10. 回传包路径、大小、SHA256；
11. OOM、重试、batch 变化和未执行步骤。

所有既有 P03–P07、MAR20 v1.1/v1.2 cache、checkpoint、codebook 和 PCA 保留，等待本地验收后再决定清理。
