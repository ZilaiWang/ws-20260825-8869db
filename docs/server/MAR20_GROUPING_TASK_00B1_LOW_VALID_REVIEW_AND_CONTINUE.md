# MAR20-GROUPING-TASK-00B1：低背景支持复核与续跑

## 0. 任务性质

本任务在 `00B` 原始 `blocked_low_valid_patch_fraction` 上执行显式修订。不得覆盖原始 summary，不重新提取已完成的 masked GeM。先复核现有缓存和19个低背景节点，再直接续跑 VLAD、候选富集和盲评包。

`00B1` 最多授予继续候选发现的权限；`formal_grouping_admission` 始终为 `false`。

## 1. 路径

```bash
cd /workspace/xh-202625
set -o pipefail
ROOT=/workspace/results/MAR20-GROUPING-TASK-00B1
OLD=/workspace/results/MAR20-GROUPING-TASK-00B
PREV=/workspace/results/MAR20-GROUPING-TASK-00
CACHE=/workspace/mar20-group-cache
MASKED="$CACHE/dinov2b-masked-patch-full-v1p2"
CODEBOOK="$CACHE/dinov2b-vlad-codebooks-v1p2"
VLAD="$CACHE/dinov2b-vlad-full-v1p2"
VLADPCA="$CACHE/dinov2b-vlad-pca512-full-v1p2"
MAR20=/workspace/inputs/MAR20
ASSETS=/workspace/p04-assets/ASSET_LOCK.json
VENV=/workspace/venvs/mar20-group-cu121
mkdir -p "$ROOT/logs"
source "$VENV/bin/activate"
export PYTHONPATH=src XFORMERS_DISABLED=1
```

如果旧任务实际 root 不同，只能整体修改 `OLD/PREV` 并在 preflight 记录，禁止混用两套上游。

## 2. 代码、环境与原始失败门禁

```bash
sha256sum -c docs/server/MAR20_GROUPING_TASK_00B1_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/code-sha256.log"

pytest -q \
  tests/test_mar20_grouping_batch_a.py \
  tests/test_mar20_grouping_00b.py \
  tests/test_mar20_grouping_00b1.py \
  2>&1 | tee "$ROOT/logs/pytest.log"

ruff check \
  src/rsdet/grouping \
  scripts/review_mar20_low_valid_patch_fraction.py \
  scripts/compile_mar20_00b1_phase_a_decision.py \
  scripts/fit_mar20_vlad_codebooks.py \
  scripts/extract_mar20_vlad_features.py \
  scripts/project_mar20_vlad_cache.py \
  scripts/mine_mar20_enriched_candidates.py \
  scripts/build_mar20_enriched_calibration_review.py \
  tests/test_mar20_grouping_00b1.py \
  2>&1 | tee "$ROOT/logs/ruff.log"

python - <<'PY' 2>&1 | tee "$ROOT/logs/environment.log"
import cv2, numpy, PIL, sklearn, torch, yaml
print('numpy', numpy.__version__, 'Pillow', PIL.__version__)
print('opencv', cv2.__version__, 'sklearn', sklearn.__version__)
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('PyYAML', yaml.__version__)
assert numpy.__version__ == '1.26.4'
assert PIL.__version__ == '10.4.0'
assert cv2.__version__ == '4.10.0'
assert sklearn.__version__ == '1.5.2'
assert torch.__version__ == '2.5.1+cu121'
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
PY
```

服务器 scoped pytest 不得出现 skip。

```bash
test -f "$MASKED/extraction_summary.json"
test -f "$MASKED/cache/cache_meta.json"
test -f "$MASKED/cache/index.json"
test -d "$MASKED/patch_samples"
test -f "$OLD/patch-mask-audit/patch_mask_audit.csv"
test -f "$OLD/patch-mask-audit/patch_mask_audit_summary.json"
test -f "$PREV/registry/image_registry.csv"
test -f "$PREV/registry/image_annotations.jsonl"
test -f "$ASSETS"

printf '%s  %s\n' \
  ffa3973e167ba6876e7397b024714ea90d45e3116bc916c3243b14a87e6ac8fb \
  "$MASKED/extraction_summary.json" \
  f56a7174bb85e4cb092e54190c60a443fe4ab28730769069d62a572e838abc23 \
  "$MASKED/cache/cache_meta.json" \
  929e859c267196bf8d6896289eed6b38ace6a93d86567d61cc76d642fda44e25 \
  "$OLD/patch-mask-audit/patch_mask_audit.csv" \
  4de5499ad101ccd3b84bf75a8a0bf25a8c32f3398aae4a15458fd05832b64d76 \
  "$OLD/patch-mask-audit/patch_mask_audit_summary.json" \
  bcdc4b697532be0899b84abb7f218a0fc2fec2c08e5aaeb3460336eea7b7201d \
  "$PREV/registry/image_registry.csv" \
  0a350c1d68e63dd134ca5f4b6a1f89c16f9a4d457f34598b72f2c1eface650f4 \
  "$PREV/registry/image_annotations.jsonl" \
  | sha256sum -c - 2>&1 | tee "$ROOT/logs/upstream-sha256.log"

test "$(sha256sum "$MASKED/cache/index.json" | awk '{print $1}')" = \
  34b72e1859fdd8f967dbd644878db49248e591f3a8a8f7ca8b3c836c99142f89

python - <<'PY' | tee "$ROOT/logs/original-failure-check.json"
import json
from pathlib import Path
root = Path('/workspace/mar20-group-cache/dinov2b-masked-patch-full-v1p2')
value = json.loads((root / 'extraction_summary.json').read_text())
assert value['status'] == 'fail_low_valid_patch_fraction'
assert len(value['low_valid_patch_nodes']) == 19
assert value['cache']['row_count'] == 15368
assert value['actual_shards'] == 121
assert value['cache']['nonfinite_count'] == 0
print(json.dumps(value, indent=2))
PY
```

任何 SHA 或原始失败状态不一致都必须停止，不能重新提取制造另一套输入。

## 3. 00B1 复核

```bash
python scripts/review_mar20_low_valid_patch_fraction.py \
  --extraction-summary "$MASKED/extraction_summary.json" \
  --cache-dir "$MASKED/cache" \
  --registry "$PREV/registry/image_registry.csv" \
  --patch-audit "$OLD/patch-mask-audit/patch_mask_audit.csv" \
  --patch-audit-summary "$OLD/patch-mask-audit/patch_mask_audit_summary.json" \
  --output-dir "$ROOT/low-valid-review" \
  --quality-threshold 0.25 \
  --very-low-threshold 0.10 \
  --maximum-low-node-fraction 0.01 \
  --maximum-audit-primary-low-fraction 0.01 \
  --primary-dilation-key dilation_0p15 \
  2>&1 | tee "$ROOT/logs/review-low-valid.log"
```

必须精确得到：

```text
status=accepted_for_continuation_with_low_support_flags
continuation_admission=true
node_count=3842
low_background_support_count=19
very_low_background_support_count=4
target/bridge low=15/4
minimum_valid_patch_count=71
audit_primary_low_count=1
formal_patch_mask_admission=false
formal_grouping_admission=false
```

`extraction_summary_admitted.json` 必须同时包含 `status=pass` 和 `source_status=fail_low_valid_patch_fraction`。原始 summary SHA 必须保持不变。

若复核 exit 2，停止并回传，不调阈值。

## 4. 复用现有 patch samples 拟合词典

```bash
python scripts/fit_mar20_vlad_codebooks.py \
  --patch-sample-dir "$MASKED/patch_samples" \
  --extraction-summary "$ROOT/low-valid-review/extraction_summary_admitted.json" \
  --output-dir "$CODEBOOK" \
  --layers 9,10,11 \
  --cluster-counts 16,32 \
  --local-dimension 128 \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/fit-vlad-codebooks.log"
```

要求6份词典；每层输入 `3842×16=61472` 个等额token；manifest必须引用 admitted summary SHA，不能引用或覆盖原始失败文件。

## 5. 第二遍 VLAD 与 PCA512

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
  2>&1 | tee "$ROOT/logs/extract-vlad-full.log"
```

要求15,368行、6个VLAD描述子、241 shards、NaN/Inf=0。完全相同命令复跑必须全部SKIP。首次OOM仅允许batch 8→4，使用新目录后缀。

```bash
python scripts/project_mar20_vlad_cache.py \
  --input-cache "$VLAD/cache" \
  --output-dir "$VLADPCA" \
  --output-dimension 512 \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/project-vlad-pca512.log"
```

要求6个512D特征、15,368行、0°的3,842行无标签拟合、全部输出L2归一。

## 6. 候选富集与盲评包

```bash
test -d "$CACHE/dinov2b-calibration-round-a-telea-v1/cache"

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

python scripts/compile_mar20_00b1_phase_a_decision.py \
  --continuation-decision "$ROOT/low-valid-review/continuation_decision.json" \
  --quality-review "$ROOT/low-valid-review/low_valid_patch_fraction_review.json" \
  --codebook-manifest "$CODEBOOK/codebook_manifest.json" \
  --vlad-summary "$VLAD/extraction_summary.json" \
  --projection-summary "$VLADPCA/projection_summary.json" \
  --candidate-summary "$ROOT/enriched-candidates/candidate_mining_summary.json" \
  --enriched-review-summary "$ROOT/enriched-review/enriched_review_summary.json" \
  --output "$ROOT/task_decision.json" \
  2>&1 | tee "$ROOT/logs/compile-phase-a-decision.log"
```

必须得到240个唯一pair、24个对换盲重复、至少180个target-target和66张contact sheet。DINO/VLAD/SIFT仅排序人工候选，不自动成边。

## 7. 正常停止

Phase A 完成后写入：

```json
{
  "status": "waiting_for_patch_mask_and_enriched_pair_reviews",
  "protocol_version": "mar20-source-grouping-v1.2-00b1-quality-amendment",
  "source_status": "blocked_low_valid_patch_fraction",
  "low_valid_review_status": "accepted_for_continuation_with_low_support_flags",
  "formal_grouping_admission": false,
  "task01_retrieval_admission": false,
  "required_manual_inputs": [
    "manual_patch_mask_review.csv",
    "manual_enriched_decisions.csv"
  ]
}
```

回传包包含：

- `low-valid-review` 全部CSV/JSON；
- patch-mask 30张contact sheet和空白review；
- enriched-review 66张contact sheet、空白decisions和summary；
- codebook、VLAD、PCA、候选的所有小型meta/index/summary；
- 全部日志和执行回报；
- 不包含大NPZ cache和封存的 `blind_card_mapping.csv`。

大缓存、codebook、PCA和blind mapping全部留在服务器。正常终止状态不是实验失败。

## 8. 后续人工回传的重要变化

人工审核合同沿用00B。Phase B 编译 patch review 时，`--audit-summary` 必须使用：

```text
$ROOT/low-valid-review/patch_mask_audit_summary_admitted.json
```

不得使用原始 fail summary，也不得删除其中 `source_automatic_geometry_gate=fail` 和 quality review SHA。

## 9. 最终回报格式

1. 00B1状态、Git/dirty、代码SHA和环境；
2. 原始失败summary/cache/audit SHA；
3. 复核的19/4/15/4/71等精确结果及分位数；
4. 原始失败是否保持不变；
5. 6份词典的输入token、维度、K、SHA；
6. VLAD/PCA行数、shard、维度、指纹、耗时、显存、resume；
7. 候选并集、每路贡献、SIFT处理和强几何支持量；
8. 盲评pair/card/target-target/重复数量；
9. 唯一task decision；
10. 回传包路径、大小、SHA及服务器保留资产；
11. OOM、重试、batch变化和未执行步骤。
