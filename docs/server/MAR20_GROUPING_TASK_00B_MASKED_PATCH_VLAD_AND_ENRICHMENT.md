# MAR20-GROUPING-TASK-00B：特征级掩码、VLAD 与正例富集

## 0. 任务性质

这是 Round-A `complete_round_a_no_admission` 后的独立 v1.2 修复任务。任务分为两阶段：

- Phase A：无标签 patch-mask audit、真实 DINO 提取、域内 VLAD、候选富集和盲评包；
- Phase B：人工回传后编译 patch 门禁、合并标定证据、分析 Round-B 并生成唯一终止决策。

本任务不生成最终 group ID，不生成 CV3，不训练比赛模型，不允许 DINO/VLAD/SIFT 分数自动 union。Round-A v1.1 目录和三份旧缓存均只读。

## 1. 固定路径

```text
repo             /workspace/xh-202625
task root        /workspace/results/MAR20-GROUPING-TASK-00B
previous root    /workspace/results/MAR20-GROUPING-TASK-00
MAR20            /workspace/inputs/MAR20
asset lock       /workspace/p04-assets/ASSET_LOCK.json
venv             /workspace/venvs/mar20-group-cu121
cache root       /workspace/mar20-group-cache
manual inputs    /workspace/inputs/MAR20-GROUPING-TASK-00B
```

```bash
cd /workspace/xh-202625
set -o pipefail
ROOT=/workspace/results/MAR20-GROUPING-TASK-00B
PREV=/workspace/results/MAR20-GROUPING-TASK-00
MAR20=/workspace/inputs/MAR20
ASSETS=/workspace/p04-assets/ASSET_LOCK.json
VENV=/workspace/venvs/mar20-group-cu121
CACHE=/workspace/mar20-group-cache
MANUAL=/workspace/inputs/MAR20-GROUPING-TASK-00B
mkdir -p "$ROOT/logs" "$CACHE" "$MANUAL"
source "$VENV/bin/activate"
export PYTHONPATH=src XFORMERS_DISABLED=1
```

## 2. 代码、环境与上游门禁

```bash
sha256sum -c docs/server/MAR20_GROUPING_TASK_00B_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/code-sha256.log"

python - <<'PY' 2>&1 | tee "$ROOT/logs/environment.log"
import cv2, numpy, PIL, sklearn, torch, yaml
print("python ok")
print("numpy", numpy.__version__)
print("Pillow", PIL.__version__)
print("opencv", cv2.__version__)
print("sklearn", sklearn.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("PyYAML", yaml.__version__)
assert numpy.__version__ == "1.26.4"
assert PIL.__version__ == "10.4.0"
assert cv2.__version__ == "4.10.0"
assert sklearn.__version__ == "1.5.2"
assert torch.__version__ == "2.5.1+cu121"
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
PY

pytest -q tests/test_mar20_grouping_batch_a.py tests/test_mar20_grouping_00b.py \
  2>&1 | tee "$ROOT/logs/pytest.log"

ruff check \
  src/rsdet/grouping \
  scripts/audit_mar20_patch_masks.py \
  scripts/compile_mar20_patch_mask_review.py \
  scripts/extract_mar20_masked_patch_features.py \
  scripts/fit_mar20_vlad_codebooks.py \
  scripts/extract_mar20_vlad_features.py \
  scripts/project_mar20_vlad_cache.py \
  scripts/mine_mar20_enriched_candidates.py \
  scripts/build_mar20_enriched_calibration_review.py \
  scripts/compile_mar20_enriched_calibration.py \
  scripts/analyze_mar20_round_b.py \
  scripts/compile_mar20_00b_decision.py \
  tests/test_mar20_grouping_00b.py \
  2>&1 | tee "$ROOT/logs/ruff.log"
```

服务器环境安装了 scikit-learn，因此 scoped pytest 不得出现 skip。

上游固定文件：

```bash
test -f "$PREV/registry/image_registry.csv"
test -f "$PREV/registry/image_annotations.jsonl"
test -f "$PREV/calibration-compiled/calibration_pairs.csv"
test -f "$PREV/task_decision.json"
test -d "$MAR20/JPEGImages"
test -f "$ASSETS"

printf '%s  %s\n' \
  bcdc4b697532be0899b84abb7f218a0fc2fec2c08e5aaeb3460336eea7b7201d \
  "$PREV/registry/image_registry.csv" \
  0a350c1d68e63dd134ca5f4b6a1f89c16f9a4d457f34598b72f2c1eface650f4 \
  "$PREV/registry/image_annotations.jsonl" \
  257a9489289f7ab1e608270d420c11683710e325e590a93a14209920013e080b \
  "$PREV/calibration-compiled/calibration_pairs.csv" \
  | sha256sum -c - 2>&1 | tee "$ROOT/logs/upstream-sha256.log"

python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("/workspace/results/MAR20-GROUPING-TASK-00/task_decision.json").read_text())
assert p["status"] == "complete_round_a_no_admission"
assert p["formal_grouping_admission"] is False
print(json.dumps(p, indent=2))
PY
```

若服务器的最终 `task_decision.json` 保存在旧 root 的子目录，允许将 `PREV` 改为包含 registry、calibration-compiled 和该决策文件的真实根目录；必须在 `system_preflight.txt` 记录，不允许混用两套上游。

## 3. Phase A-1：patch-mask 自动与人工审计包

```bash
python scripts/audit_mar20_patch_masks.py \
  --registry "$PREV/registry/image_registry.csv" \
  --annotations "$PREV/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --output-dir "$ROOT/patch-mask-audit" \
  --sample-count 120 \
  --dilation-ratios 0.10,0.15,0.20 \
  --input-size 518 \
  --patch-size 14 \
  --maximum-patch-foreground-fraction 0.20 \
  --minimum-valid-patch-fraction 0.25 \
  --cards-per-sheet 4 \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/patch-mask-audit.log"
```

必须得到：

- 120 行 audit 和空白 review；
- 30 张 contact sheets；
- 10%/15%/20% 三组可追溯 mask SHA；
- `automatic_geometry_gate=pass`；
- `formal_patch_mask_admission=false`，因为尚未人工审核。

若自动门禁 exit 2，保留全部产物并停止 GPU 提取，不改 20%/25% 阈值重跑。

## 4. Phase A-2：真实 DINO masked-patch smoke

```bash
python scripts/extract_mar20_masked_patch_features.py \
  --registry "$PREV/registry/image_registry.csv" \
  --annotations "$PREV/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --asset-lock "$ASSETS" \
  --output-dir "$ROOT/dino-masked-smoke" \
  --encoder dinov2_vitb14 \
  --scope target_only \
  --max-nodes 2 \
  --rotations 0,90,180,270 \
  --layers 9,10,11 \
  --gem-powers 2,3,4 \
  --input-size 518 \
  --patch-size 14 \
  --dilation-ratio 0.15 \
  --maximum-patch-foreground-fraction 0.20 \
  --minimum-valid-patch-fraction 0.25 \
  --patch-samples-per-node 16 \
  --batch-size 8 \
  --shard-size 8 \
  --device cuda \
  --compute-dtype float16 \
  --storage-dtype float16 \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/dino-masked-smoke.log"
```

要求 8 行、12 个 768D 特征、NaN/Inf=0、low-valid=0、两节点各 16 个 0° 有效 patch sample。mock 结果不能代替本门禁。

## 5. Phase A-3：3,842 图全量 masked GeM 和等额 token

```bash
MASKED="$CACHE/dinov2b-masked-patch-full-v1p2"
python scripts/extract_mar20_masked_patch_features.py \
  --registry "$PREV/registry/image_registry.csv" \
  --annotations "$PREV/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --asset-lock "$ASSETS" \
  --output-dir "$MASKED" \
  --encoder dinov2_vitb14 \
  --scope full_bridge \
  --rotations 0,90,180,270 \
  --layers 9,10,11 \
  --gem-powers 2,3,4 \
  --input-size 518 \
  --patch-size 14 \
  --dilation-ratio 0.15 \
  --maximum-patch-foreground-fraction 0.20 \
  --minimum-valid-patch-fraction 0.25 \
  --patch-samples-per-node 16 \
  --batch-size 16 \
  --shard-size 128 \
  --device cuda \
  --compute-dtype float16 \
  --storage-dtype float16 \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/extract-masked-full.log"
```

必须是 3,842 节点、15,368 行、12 个 768D 描述子、121 个 cache shard，low-valid=0。完全相同命令立即复跑一次，必须 `computed_rows=0`、`skipped_rows=15368`、index SHA 不变；patch-sample shard 也必须全部跳过。

首次 OOM 时唯一允许的调整是 batch 16→8→4；每个变体使用新目录后缀，不得混写。

## 6. Phase A-4：拟合域内 PCA128 + VLAD16/32

```bash
CODEBOOK="$CACHE/dinov2b-vlad-codebooks-v1p2"
python scripts/fit_mar20_vlad_codebooks.py \
  --patch-sample-dir "$MASKED/patch_samples" \
  --extraction-summary "$MASKED/extraction_summary.json" \
  --output-dir "$CODEBOOK" \
  --layers 9,10,11 \
  --cluster-counts 16,32 \
  --local-dimension 128 \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/fit-vlad-codebooks.log"
```

必须得到 6 份 codebook，每层的 token 数必须等于 `3842×16=61472`；manifest 中 `image_balanced_samples=true`。

## 7. Phase A-5：全量 VLAD 与 PCA512

```bash
VLAD="$CACHE/dinov2b-vlad-full-v1p2"
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
  2>&1 | tee "$ROOT/logs/extract-vlad-full.log"
```

必须得到 15,368 行、6 个 VLAD 描述子、241 shards、NaN/Inf=0。完全同命令复跑必须全部 SKIP。

```bash
VLADPCA="$CACHE/dinov2b-vlad-pca512-full-v1p2"
python scripts/project_mar20_vlad_cache.py \
  --input-cache "$VLAD/cache" \
  --output-dir "$VLADPCA" \
  --output-dimension 512 \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/project-vlad-pca512.log"
```

每个 PCA 只用 3,842 个 0° 行无标签拟合，全部 15,368 行投影到 512D 并 L2 归一。

## 8. Phase A-6：多路候选与 SIFT 富集

先确认旧 Telea cache 确实保留：

```bash
test -d "$CACHE/dinov2b-calibration-round-a-telea-v1/cache"
```

它只用于 candidate discovery。不得将其加入 Round-B 正式 routes。

```bash
python scripts/mine_mar20_enriched_candidates.py \
  --registry "$PREV/registry/image_registry.csv" \
  --annotations "$PREV/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --routes-json configs/grouping/mar20_00b_candidate_routes_server.json \
  --existing-pairs "$PREV/calibration-compiled/calibration_pairs.csv" \
  --output-dir "$ROOT/enriched-candidates" \
  --top-k-per-route 12 \
  --row-search-multiplier 8 \
  --retrieval-batch-size 256 \
  --device cuda \
  --pregeometry-limit 1600 \
  --phash-top-k 5 \
  --sift-max-dimension 1024 \
  --sift-features 2000 \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/mine-enriched-candidates.log"
```

要求：旧 360 pair 全部去重；每条路由对全库检索；前 1,600 对获得 SIFT/RANSAC 字段；`formal_edge_admission=false`。

```bash
python scripts/build_mar20_enriched_calibration_review.py \
  --candidates "$ROOT/enriched-candidates/enriched_candidate_pairs.csv" \
  --candidate-summary "$ROOT/enriched-candidates/candidate_mining_summary.json" \
  --registry "$PREV/registry/image_registry.csv" \
  --mar20-root "$MAR20" \
  --output-dir "$ROOT/enriched-review" \
  --pair-count 240 \
  --duplicate-fraction 0.10 \
  --minimum-target-target-fraction 0.75 \
  --cards-per-sheet 4 \
  2>&1 | tee "$ROOT/logs/build-enriched-review.log"
```

必须是 240 个唯一 pair、24 个盲重复、至少 180 个 target-target，并生成 66 张左右 contact sheet。

## 9. Phase A 正常停止与回传

人工文件尚不存在时，写入：

```json
{
  "status": "waiting_for_patch_mask_and_enriched_pair_reviews",
  "protocol_version": "mar20-source-grouping-v1.2",
  "formal_grouping_admission": false,
  "task01_retrieval_admission": false,
  "required_manual_inputs": [
    "manual_patch_mask_review.csv",
    "manual_enriched_decisions.csv"
  ]
}
```

回传包必须包含：

- `patch-mask-audit` 的 summary、CSV、overlays、contact sheets；
- `enriched-review` 的 contact sheets、空白 decisions 和 summary；
- 所有 extraction/projection/candidate summary；
- cache meta/index/sidecar 和 codebook/PCA SHA 清单，不包含大 NPZ cache；
- 所有日志、pip freeze、GPU/耗时/峰值显存记录和 task decision。

`blind_card_mapping.csv` 保留服务器封存，Phase A 回传包不得包含。所有大 cache、codebook 和 PCA 保留服务器，不删除。

## 10. 人工回传合同

上传至：

```text
/workspace/inputs/MAR20-GROUPING-TASK-00B/manual_patch_mask_review.csv
/workspace/inputs/MAR20-GROUPING-TASK-00B/manual_enriched_decisions.csv
```

patch review 字段均为 0/1：

- `valid`；
- `dilation_0p10_aircraft_covered`；
- `dilation_0p15_aircraft_covered`；
- `dilation_0p20_aircraft_covered`；
- `dilation_0p15_excessive_background_loss`。

pair review 的 label 只能是：

```text
same_frame
geometric_overlap
same_local_site
likely_same_airport
not_same_local_site
different_airport
uncertain
```

`confidence` 必须是 `[0,1]`。人工不得在审核前打开 mapping；服务器 AI 不得自动填写人工文件。

两份文件不齐时必须保持 waiting，不运行 Phase B。

## 11. Phase B-1：编译 patch 和富集标定集

```bash
test -s "$MANUAL/manual_patch_mask_review.csv"
test -s "$MANUAL/manual_enriched_decisions.csv"

set +e
python scripts/compile_mar20_patch_mask_review.py \
  --audit "$ROOT/patch-mask-audit/patch_mask_audit.csv" \
  --audit-summary "$ROOT/patch-mask-audit/patch_mask_audit_summary.json" \
  --review "$MANUAL/manual_patch_mask_review.csv" \
  --output "$ROOT/patch-mask-decision.json" \
  --minimum-valid-rate 0.95 \
  --minimum-aircraft-coverage-rate 0.95 \
  --maximum-excessive-loss-rate 0.10 \
  2>&1 | tee "$ROOT/logs/compile-patch-mask-review.log"
PATCH_EXIT=${PIPESTATUS[0]}
set -e
test "$PATCH_EXIT" -eq 0 -o "$PATCH_EXIT" -eq 2

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
  2>&1 | tee "$ROOT/logs/compile-enriched-calibration.log"
```

任何盲重复冲突、新旧标签冲突或 strict component 跨 split 都是技术/人工证据失败，必须停止并回传冲突明细。

## 12. Phase B-2：Round-B 分析与最终决策

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
  2>&1 | tee "$ROOT/logs/analyze-round-b.log"
ROUND_B_EXIT=${PIPESTATUS[0]}
set -e
test "$ROUND_B_EXIT" -eq 0 -o "$ROUND_B_EXIT" -eq 2

python scripts/compile_mar20_00b_decision.py \
  --patch-mask-decision "$ROOT/patch-mask-decision.json" \
  --calibration-summary "$ROOT/calibration-v1p2/calibration_compile_summary_v1p2.json" \
  --round-b-decision "$ROOT/round-b/round_b_decision.json" \
  --output "$ROOT/task_decision.json" \
  2>&1 | tee "$ROOT/logs/compile-task-decision.log"
```

允许的终止状态只有：

```text
ready_for_task01_retrieval_and_geometry
needs_second_enrichment_batch
complete_00b_patch_mask_no_admission
complete_00b_calibration_no_admission
complete_00b_retrieval_no_admission
```

无论哪种状态，`formal_grouping_admission=false`。只有第一种允许进入 TASK-01。

## 13. Phase B 回传

回传：

- `patch-mask-decision.json`；
- `calibration-v1p2` 全部小型文件；
- `round-b` 全部表格、JSON 和日志；
- `task_decision.json`；
- 新旧证据 SHA；
- cache/codebook/PCA 路径、大小、SHA 清单；
- 全部 Phase A/B 日志和完整执行回报。

不打包大 cache NPZ，服务器上所有 P03–P07、MAR20 v1.1/v1.2 cache、checkpoint、codebook 和 PCA 均保留，等待本地验收后再决定清理。

## 14. 最终执行回报格式

1. 状态和是否触发科学停止；
2. Git commit/dirty 及 code SHA 门禁；
3. GPU、driver、Python、torch/CUDA、numpy/Pillow/OpenCV/sklearn；
4. 上游 registry/annotation/calibration SHA；
5. patch audit 数量、自动门禁和人工门禁；
6. masked GeM cache 行数、shard、指纹、有效 patch 分布、时间、显存和 resume；
7. 6 份 codebook 的 token 数、维度、K 和 SHA；
8. VLAD/PCA cache 行数、维度、指纹、时间和 resume；
9. 候选并集量、SIFT 处理量、几何支持量；
10. 盲评 pair/card/target-target/重复数；
11. 新标定集总正例、新增正例、calibration/held-out、component 数和一致率；
12. Round-B 单路/并集的 Recall@20/50/100、Wilson 区间、negative rate 和候选负载；
13. calibration-only 选出的路由与 held-out 结果；
14. 唯一 `task_decision.json` 状态和 next action；
15. 回传包路径、大小、SHA256；
16. 失败、OOM、重试、batch 变更和未执行步骤。
