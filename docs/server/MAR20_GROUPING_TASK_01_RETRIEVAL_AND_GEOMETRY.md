# MAR20-GROUPING-TASK-01：正式检索、局部几何与盲审包

## 0. 任务目标和终止边界

本任务复用同一服务器上 TASK-00B1/00B2 的 DINO、VLAD、PCA 和 codebook，不重新拟合描述子。依次完成：

1. 两条入选 VLAD 路由的全量 K=50 正式检索和 K=100 审计索引；
2. 最多 6,000 pair 的局部几何队列，强制纳入全部 600 个冻结控制 pair；
3. block11 PCA-128 patch overlap 与背景 SIFT/RANSAC；
4. calibration-only 人工复核排序；
5. 生成匿名审查包。

正常状态必须是：

```text
waiting_for_blind_pair_review
formal_grouping_admission=false
```

本任务不生成最终 group，不运行 LightGlue/RoMa，不允许除像素等价外的自动 union。

## 1. 路径

```bash
cd /workspace/xh-202625
set -euo pipefail

ROOT=/workspace/results/MAR20-GROUPING-TASK-01
PREV=/workspace/results/MAR20-GROUPING-TASK-00
B2=/workspace/results/MAR20-GROUPING-TASK-00B1
CACHE=/workspace/mar20-group-cache
VLADPCA="$CACHE/dinov2b-vlad-pca512-full-v1p2"
CODEBOOK="$CACHE/dinov2b-vlad-codebooks-v1p2"
PATCHCACHE="$CACHE/dinov2b-task01-patch-overlap-v1"
MAR20=/workspace/inputs/MAR20
ASSETS=/workspace/p04-assets/ASSET_LOCK.json
VENV=/workspace/venvs/mar20-group-cu121

mkdir -p "$ROOT/logs"
source "$VENV/bin/activate"
export PYTHONPATH=src XFORMERS_DISABLED=1
```

若实际服务器统一位于 `/root/autodl-tmp/workspace`，只允许把 `/workspace` 前缀整体替换一次，并把最终变量写入 `$ROOT/logs/resolved_paths.txt`。禁止混用两套上游目录。

```bash
printf '%s\n' \
  "ROOT=$ROOT" "PREV=$PREV" "B2=$B2" "CACHE=$CACHE" \
  "VLADPCA=$VLADPCA" "CODEBOOK=$CODEBOOK" "PATCHCACHE=$PATCHCACHE" \
  "MAR20=$MAR20" "ASSETS=$ASSETS" "VENV=$VENV" \
  | tee "$ROOT/logs/resolved_paths.txt"
```

## 2. 代码、环境和上游门禁

```bash
sha256sum -c docs/server/MAR20_GROUPING_TASK_01_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/code-sha256.log"

pytest -q \
  tests/test_mar20_grouping_task01.py \
  tests/test_mar20_grouping_00b.py \
  tests/test_mar20_grouping_00b1.py \
  tests/test_mar20_grouping_batch_a.py \
  2>&1 | tee "$ROOT/logs/pytest.log"

ruff check \
  src/rsdet/grouping \
  scripts/retrieve_mar20_task01_candidates.py \
  scripts/build_mar20_geometry_queue.py \
  scripts/extract_mar20_patch_overlap_cache.py \
  scripts/verify_mar20_task01_geometry.py \
  scripts/analyze_mar20_task01_geometry.py \
  scripts/build_mar20_task01_blind_review.py \
  tests/test_mar20_grouping_task01.py \
  2>&1 | tee "$ROOT/logs/ruff.log"

python - <<'PY' 2>&1 | tee "$ROOT/logs/environment.log"
import cv2, numpy, PIL, sklearn, torch, yaml
assert numpy.__version__ == '1.26.4'
assert PIL.__version__ == '10.4.0'
assert cv2.__version__ == '4.10.0'
assert sklearn.__version__ == '1.5.2'
assert torch.__version__ == '2.5.1+cu121'
assert torch.cuda.is_available()
print('GPU', torch.cuda.get_device_name(0))
print('numpy', numpy.__version__, 'Pillow', PIL.__version__, 'opencv', cv2.__version__)
print('sklearn', sklearn.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)
print('PyYAML', yaml.__version__)
PY
```

服务器 scoped pytest 不得出现 skip。逐项核验上游：

```bash
test -s "$PREV/registry/image_registry.csv"
test -s "$PREV/registry/image_annotations.jsonl"
test -s "$B2/calibration-v1p2/calibration_pairs_v1p2.csv"
test -s "$B2/calibration-v1p2/calibration_compile_summary_v1p2.json"
test -s "$B2/round-b/round_b_decision.json"
test -s "$B2/task_decision.json"
test -s "$VLADPCA/cache/index.json"
test -s "$CODEBOOK/codebook_manifest.json"
test -s "$ASSETS"

printf '%s  %s\n' \
  bcdc4b697532be0899b84abb7f218a0fc2fec2c08e5aaeb3460336eea7b7201 \
  "$PREV/registry/image_registry.csv" \
  0a350c1d68e63dd134ca5f4b6a1f89c16f9a4d457f34598b72f2c1eface650f4 \
  "$PREV/registry/image_annotations.jsonl" \
  9d9e2a4686f7cd498c7e37a38cbae8812ab7820418e6e2dcfc7dc7250b3faa60 \
  "$B2/calibration-v1p2/calibration_pairs_v1p2.csv" \
  87450b7930c4c3a9fb66f98ceed10315f77e16fa5ee680ae5305a32c95f0af7b \
  "$B2/calibration-v1p2/calibration_compile_summary_v1p2.json" \
  d46205a27cf55d74c2cdcf56b9d0c4c98cb584b6261c3289731fc24687c1444f \
  "$B2/round-b/round_b_decision.json" \
  deea8b6831e7b9e03302abea3fd73eadb83816d02d3236b79e99736ca9f70334 \
  "$B2/task_decision.json" \
  dfd88a6b25c028990f1eb90fe944902e29369ba4b5445ba2d41fc2250df8d837 \
  configs/grouping/mar20_00b_round_b_routes_server.json \
  | sha256sum -c - 2>&1 | tee "$ROOT/logs/upstream-sha256.log"

python - <<'PY' 2>&1 | tee "$ROOT/logs/upstream-decision.json"
import json
from pathlib import Path
b2 = Path('/workspace/results/MAR20-GROUPING-TASK-00B1')
round_b = json.loads((b2/'round-b/round_b_decision.json').read_text())
task = json.loads((b2/'task_decision.json').read_text())
assert round_b['formal_descriptor_selection_admission'] is True
assert round_b['selection_uses_heldout'] is False
assert round_b['selected_routes'] == [
    'masked_block10_vlad_k32_pca512',
    'masked_block11_vlad_k32_pca512',
]
assert task['status'] == 'ready_for_task01_retrieval_and_geometry'
assert task['task01_retrieval_admission'] is True
assert task['formal_grouping_admission'] is False
print(json.dumps({'round_b': round_b, 'task': task}, indent=2))
PY
```

若使用了整体路径替换，上面 Python 的 `b2` 同步替换。任一失败立即停止，不得重拟合或改 K。

## 3. 全量 K=50 检索与 K=100 审计

```bash
python scripts/retrieve_mar20_task01_candidates.py \
  --registry "$PREV/registry/image_registry.csv" \
  --mar20-root "$MAR20" \
  --routes-json configs/grouping/mar20_00b_round_b_routes_server.json \
  --round-b-decision "$B2/round-b/round_b_decision.json" \
  --task-00b2-decision "$B2/task_decision.json" \
  --calibration-pairs "$B2/calibration-v1p2/calibration_pairs_v1p2.csv" \
  --output-dir "$ROOT/retrieval" \
  --formal-k 50 \
  --audit-k 100 \
  --device cuda \
  --batch-size 64 \
  2>&1 | tee "$ROOT/logs/retrieval.log"

python - <<'PY' 2>&1 | tee "$ROOT/logs/retrieval-gate.json"
import json
from pathlib import Path
p=Path('/workspace/results/MAR20-GROUPING-TASK-01/retrieval')
s=json.loads((p/'retrieval_summary.json').read_text())
d=json.loads((p/'retrieval_decision.json').read_text())
assert s['status']=='pass' and d['formal_retrieval_admission'] is True
assert s['node_count']==3842
assert s['formal_k_per_route']==50 and s['audit_k_per_route']==100
assert s['saturation']['held_out_audit__recall_at_50'] >= 0.95
assert d['formal_grouping_admission'] is False
print(json.dumps({'summary':s,'decision':d},indent=2))
PY
```

## 4. 冻结 6,000 pair 几何队列

```bash
python scripts/build_mar20_geometry_queue.py \
  --formal-candidates "$ROOT/retrieval/candidate_edges_full_bridge_k50.csv" \
  --retrieval-decision "$ROOT/retrieval/retrieval_decision.json" \
  --calibration-pairs "$B2/calibration-v1p2/calibration_pairs_v1p2.csv" \
  --registry "$PREV/registry/image_registry.csv" \
  --mar20-root "$MAR20" \
  --output-dir "$ROOT/geometry-queue" \
  --maximum-pairs 6000 \
  2>&1 | tee "$ROOT/logs/build-geometry-queue.log"

python - <<'PY' 2>&1 | tee "$ROOT/logs/geometry-queue-gate.json"
import json
from pathlib import Path
p=Path('/workspace/results/MAR20-GROUPING-TASK-01/geometry-queue/geometry_queue_summary.json')
s=json.loads(p.read_text())
assert s['status']=='pass'
assert s['queue_pair_count']==6000
assert s['calibration_control_count']==600
assert s['new_candidate_count']==5400
assert s['formal_grouping_admission'] is False
print(json.dumps(s,indent=2))
PY
```

## 5. block11 局部 patch cache

只提取 geometry queue 涉及的节点。使用 00B1 已冻结的 block11/K32 local PCA-128，不重新拟合 PCA。

```bash
python scripts/extract_mar20_patch_overlap_cache.py \
  --geometry-queue "$ROOT/geometry-queue/geometry_queue.csv" \
  --geometry-queue-summary "$ROOT/geometry-queue/geometry_queue_summary.json" \
  --registry "$PREV/registry/image_registry.csv" \
  --annotations "$PREV/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --asset-lock "$ASSETS" \
  --codebook-manifest "$CODEBOOK/codebook_manifest.json" \
  --codebook-dir "$CODEBOOK" \
  --output-dir "$PATCHCACHE" \
  --batch-size 12 \
  --shard-size 16 \
  --device cuda \
  --compute-dtype float16 \
  2>&1 | tee "$ROOT/logs/extract-patch-overlap.log"

python scripts/extract_mar20_patch_overlap_cache.py \
  --geometry-queue "$ROOT/geometry-queue/geometry_queue.csv" \
  --geometry-queue-summary "$ROOT/geometry-queue/geometry_queue_summary.json" \
  --registry "$PREV/registry/image_registry.csv" \
  --annotations "$PREV/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --asset-lock "$ASSETS" \
  --codebook-manifest "$CODEBOOK/codebook_manifest.json" \
  --codebook-dir "$CODEBOOK" \
  --output-dir "$PATCHCACHE" \
  --batch-size 12 \
  --shard-size 16 \
  --device cuda \
  --compute-dtype float16 \
  2>&1 | tee "$ROOT/logs/extract-patch-overlap-resume.log"

python - <<'PY' 2>&1 | tee "$ROOT/logs/patch-cache-gate.json"
import json
from pathlib import Path
p=Path('/workspace/mar20-group-cache/dinov2b-task01-patch-overlap-v1')
s=json.loads((p/'patch_overlap_extraction_summary.json').read_text())
assert s['status']=='pass'
assert s['computed_nodes']==0 and s['skipped_nodes']==s['node_count']
assert s['cache']['nonfinite_count']==0
assert s['cache']['feature_dimensions']['block11_coarse_tokens_flat']==19*19*128
assert s['cache']['feature_dimensions']['coarse_valid']==19*19
print(json.dumps(s,indent=2))
PY
```

OOM 时只允许 batch `12→8→4`，必须继续使用相同 cache 合同；不可改变 grid、PCA、mask 或 storage dtype。

## 6. SIFT、RANSAC 与 patch overlap

```bash
python scripts/verify_mar20_task01_geometry.py \
  --geometry-queue "$ROOT/geometry-queue/geometry_queue.csv" \
  --geometry-queue-summary "$ROOT/geometry-queue/geometry_queue_summary.json" \
  --patch-cache "$PATCHCACHE/cache" \
  --patch-summary "$PATCHCACHE/patch_overlap_extraction_summary.json" \
  --registry "$PREV/registry/image_registry.csv" \
  --annotations "$PREV/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --output-dir "$ROOT/geometry" \
  --sift-max-dimension 1024 \
  --sift-features 2500 \
  --sift-ratio 0.75 \
  --ransac-repeat-count 20 \
  --pair-shard-size 250 \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/verify-geometry.log"

# 完全相同命令复跑一次，验证 pair shards 和 SIFT cache resume。
python scripts/verify_mar20_task01_geometry.py \
  --geometry-queue "$ROOT/geometry-queue/geometry_queue.csv" \
  --geometry-queue-summary "$ROOT/geometry-queue/geometry_queue_summary.json" \
  --patch-cache "$PATCHCACHE/cache" \
  --patch-summary "$PATCHCACHE/patch_overlap_extraction_summary.json" \
  --registry "$PREV/registry/image_registry.csv" \
  --annotations "$PREV/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --output-dir "$ROOT/geometry" \
  --sift-max-dimension 1024 \
  --sift-features 2500 \
  --sift-ratio 0.75 \
  --ransac-repeat-count 20 \
  --pair-shard-size 250 \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/verify-geometry-resume.log"

python - <<'PY' 2>&1 | tee "$ROOT/logs/geometry-gate.json"
import json
from pathlib import Path
p=Path('/workspace/results/MAR20-GROUPING-TASK-01/geometry')
s=json.loads((p/'geometry_verification_summary.json').read_text())
assert s['status']=='pass' and s['pair_count']==6000
assert s['computed_pairs']==0 and s['skipped_pairs']==6000
assert s['nonfinite_count']==0
assert s['formal_grouping_admission'] is False
print(json.dumps(s,indent=2))
PY
```

## 7. calibration-only 排序与盲审包

```bash
python scripts/analyze_mar20_task01_geometry.py \
  --pair-evidence "$ROOT/geometry/pair_evidence.csv" \
  --geometry-summary "$ROOT/geometry/geometry_verification_summary.json" \
  --output-dir "$ROOT/geometry-analysis" \
  --minimum-q1-calibration-precision 0.90 \
  --minimum-q1-calibration-count 20 \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/analyze-geometry.log"

python scripts/build_mar20_task01_blind_review.py \
  --pair-evidence "$ROOT/geometry/pair_evidence.csv" \
  --geometry-summary "$ROOT/geometry/geometry_verification_summary.json" \
  --assignments "$ROOT/geometry-analysis/geometry_queue_assignments.csv" \
  --geometry-decision "$ROOT/geometry-analysis/geometry_calibration_decision.json" \
  --registry "$PREV/registry/image_registry.csv" \
  --annotations "$PREV/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --output-dir "$ROOT/blind-review" \
  --new-pair-count 300 \
  --control-pair-count 48 \
  --duplicate-fraction 0.08 \
  --cards-per-sheet 4 \
  2>&1 | tee "$ROOT/logs/build-blind-review.log"

python - <<'PY' 2>&1 | tee "$ROOT/logs/final-gate.json"
import json
from pathlib import Path
p=Path('/workspace/results/MAR20-GROUPING-TASK-01/blind-review')
s=json.loads((p/'blind_review_summary.json').read_text())
d=json.loads((p/'task_decision.json').read_text())
assert s['status']=='waiting_for_blind_pair_review'
assert s['new_pair_count']==300 and s['control_pair_count']==48
assert s['blind_duplicate_count'] >= 27
assert d['status']=='waiting_for_blind_pair_review'
assert d['formal_grouping_admission'] is False
assert d['next_action']=='complete_manual_review_then_compile_strict_core'
print(json.dumps({'summary':s,'decision':d},indent=2))
PY
```

`heldout_metrics` 无论好坏都只能如实回报，不得据此重拟合模型、改阈值或重新选择路由。

## 8. 回传包

大 cache 留服务器：

- `$VLADPCA`；
- `$CODEBOOK`；
- `$PATCHCACHE/cache`；
- `$ROOT/geometry/sift_cache`；
- `$ROOT/geometry/pair_shards`。

回传：

```bash
tar -czf /workspace/results/MAR20-GROUPING-TASK-01-return.tar.gz \
  -C "$ROOT" \
  logs \
  retrieval/retrieval_summary.json \
  retrieval/retrieval_decision.json \
  retrieval/candidate_edges_target_k50.csv \
  geometry-queue/geometry_queue.csv \
  geometry-queue/geometry_queue_summary.json \
  geometry/pair_evidence.csv \
  geometry/geometry_verification_summary.json \
  geometry-analysis \
  blind-review

sha256sum /workspace/results/MAR20-GROUPING-TASK-01-return.tar.gz \
  | tee "$ROOT/logs/return-package-sha256.txt"
```

若回传包过大，只允许从包中移除 `candidate_edges_target_k50.csv`；不得移除 `pair_evidence.csv`、盲审表、contact sheets、summary 或日志。

## 9. 最终回报格式

必须报告：

1. 技术状态与科学状态；
2. 代码 SHA、pytest、ruff、环境；
3. 三个 K=50/K=100 candidate 文件的行数、SHA 和 relation 构成；
4. calibration/held-out 的 R@20/50/100 与 Wilson 区间；
5. geometry queue 的 600 控制 + 5,400 新候选构成；
6. patch cache node/shard/维度/指纹、首次计算和 resume；
7. SIFT cache node 数、pair shard 数、6000/6000 完整性和 resume；
8. calibration 与 held-out 几何排序指标、Q0～Q4 数量；
9. 盲审包 300 新 pair、48 控制、重复数、卡片和 sheet 数；
10. 唯一 task decision；
11. 运行时间、峰值显存、磁盘、OOM/重试/batch 变化；
12. 回传包路径、大小和 SHA256。

不得宣称已经得到机场真值或正式 `group_id`。
