# MAR20-GROUPING-TASK-01A：退化几何证据清洗与可审计重编译

## 0. 目标、原因和边界

TASK-01 已经完成 6,000 对检索候选的局部几何计算，但 29 对中的 OpenCV affine RANSAC 退化，每对留下 1 个非有限矩阵和 2 个非有限误差字段，共 87 个非有限字段。原 driver 因此停止，后续分析和盲审产物虽已存在，但没有形成一条可复现的成功日志链。

本修订任务只做三件事：

1. 校验原始 `pair_evidence.csv` 的完整性；
2. 将退化模型的 `NaN/Inf` 转为显式的“缺失证据”，不当作 0 误差或正证据；
3. 使用清洗后的 6,000 行证据重跑 calibration-only 排序和盲审包生成。

严格禁止：重跑 DINO/VLAD、重跑 SIFT/RANSAC、改变检索 K、重拟合描述子、改变校准集、根据 held-out 指标调参、覆盖 TASK-01 原始产物。

预期最终状态是：

```text
technical_status=pass
scientific_status=waiting_for_blind_pair_review
formal_grouping_admission=false
```

## 1. 路径与不可变输入

```bash
cd /workspace/xh-202625
set -euo pipefail

SOURCE=/workspace/results/MAR20-GROUPING-TASK-01
ROOT=/workspace/results/MAR20-GROUPING-TASK-01A
PREV=/workspace/results/MAR20-GROUPING-TASK-00
MAR20=/workspace/inputs/MAR20
VENV=/workspace/venvs/mar20-group-cu121
RAW="$SOURCE/geometry/pair_evidence.csv"
RAW_SHA=97920b081d1137e72817191ccca1dea90955045e460ace9af03cc7253efa2051

mkdir -p "$ROOT/logs"
source "$VENV/bin/activate"
export PYTHONPATH=src XFORMERS_DISABLED=1

printf '%s\n' \
  "SOURCE=$SOURCE" "ROOT=$ROOT" "PREV=$PREV" "MAR20=$MAR20" \
  "VENV=$VENV" "RAW=$RAW" "RAW_SHA=$RAW_SHA" \
  | tee "$ROOT/logs/resolved_paths.txt"
```

如服务器实际使用 `/root/autodl-tmp/workspace`，必须将所有 `/workspace` 前缀统一替换，不得混用。

## 2. 代码与环境门禁

```bash
sha256sum -c docs/server/MAR20_GROUPING_TASK_01A_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/code-sha256.log"

pytest -q \
  tests/test_mar20_grouping_task01.py \
  tests/test_mar20_grouping_00b.py \
  tests/test_mar20_grouping_00b1.py \
  tests/test_mar20_grouping_batch_a.py \
  2>&1 | tee "$ROOT/logs/pytest.log"

ruff check \
  src/rsdet/grouping \
  scripts/sanitize_mar20_task01_geometry.py \
  scripts/analyze_mar20_task01_geometry.py \
  scripts/build_mar20_task01_blind_review.py \
  tests/test_mar20_grouping_task01.py \
  2>&1 | tee "$ROOT/logs/ruff.log"

python - <<'PY' 2>&1 | tee "$ROOT/logs/environment.log"
import cv2, numpy, PIL, sklearn
assert numpy.__version__ == '1.26.4'
assert PIL.__version__ == '10.4.0'
assert cv2.__version__ == '4.10.0'
assert sklearn.__version__ == '1.5.2'
print('numpy', numpy.__version__, 'Pillow', PIL.__version__)
print('opencv', cv2.__version__, 'sklearn', sklearn.__version__)
PY
```

本任务是 CPU 任务，不得因 GPU 空闲情况停止。

## 3. 原始证据独立审计

```bash
test -s "$RAW"
printf '%s  %s\n' "$RAW_SHA" "$RAW" \
  | sha256sum -c - 2>&1 | tee "$ROOT/logs/raw-evidence-sha256.log"

python - "$RAW" <<'PY' 2>&1 | tee "$ROOT/logs/raw-evidence-audit.json"
import csv, json, math, sys
from collections import Counter
from pathlib import Path

path=Path(sys.argv[1])
with path.open(encoding='utf-8', newline='') as f:
    rows=list(csv.DictReader(f))
assert len(rows)==6000
assert len({r['pair_uid'] for r in rows})==6000
fields=Counter()
pairs=set()
for row in rows:
    for key,text in row.items():
        if not text:
            continue
        if key.endswith('_matrix'):
            try: values=[float(part) for part in text.split(';') if part]
            except ValueError: continue
            if values and not all(math.isfinite(value) for value in values):
                fields[key]+=1
                pairs.add(row['pair_uid'])
            continue
        try: value=float(text)
        except ValueError: continue
        if not math.isfinite(value):
            fields[key]+=1
            pairs.add(row['pair_uid'])
assert fields==Counter({'affine_matrix':29, 'affine_median_error':29,
                       'affine_p95_error':29}), fields
assert len(pairs)==29
print(json.dumps({'status':'pass','rows':len(rows),'nonfinite_fields':fields,
                  'affected_pairs':len(pairs)},indent=2))
PY
```

若 SHA、行数、影响对数或字段类型不匹配，立即停止并回报；不得放宽期望值。

## 4. 清洗为显式缺失证据

```bash
python scripts/sanitize_mar20_task01_geometry.py \
  --raw-evidence "$RAW" \
  --expected-raw-sha256 "$RAW_SHA" \
  --output-dir "$ROOT/geometry-sanitized" \
  --expected-rows 6000 \
  --expected-nonfinite-fields 87 \
  --expected-affected-pairs 29 \
  2>&1 | tee "$ROOT/logs/sanitize-geometry.log"

SANITIZED="$ROOT/geometry-sanitized/pair_evidence_sanitized.csv"
SANITIZED_SUMMARY="$ROOT/geometry-sanitized/geometry_sanitization_summary.json"

python - "$SANITIZED" "$SANITIZED_SUMMARY" \
  2>&1 <<'PY' | tee "$ROOT/logs/sanitized-evidence-gate.json"
import csv, hashlib, json, math, sys
from pathlib import Path

evidence=Path(sys.argv[1]); summary=Path(sys.argv[2])
s=json.loads(summary.read_text())
rows=list(csv.DictReader(evidence.open(encoding='utf-8',newline='')))
sha=hashlib.sha256(evidence.read_bytes()).hexdigest()
assert s['status']=='pass'
assert s['row_count']==6000 and len(rows)==6000
assert s['raw_nonfinite_field_count']==87
assert s['affected_pair_count']==29
assert s['remaining_nonfinite_count']==0
assert s['pair_evidence_sha256']==sha==s['sanitized_evidence_sha256']
changed=[r for r in rows if r['sanitized_nonfinite_fields']]
assert len(changed)==29
assert all(r['affine_fit_valid']=='0' for r in changed)
for row in rows:
    for key,text in row.items():
        if not text or key=='sanitized_nonfinite_fields' or key.endswith('_matrix'):
            continue
        try: value=float(text)
        except ValueError: continue
        assert math.isfinite(value), (row['pair_uid'],key,text)
print(json.dumps({'status':'pass','rows':len(rows),'sha256':sha,
                  'sanitized_degenerate_affine_pairs':29},indent=2))
PY
```

## 5. 重编译 calibration-only 排序

```bash
python scripts/analyze_mar20_task01_geometry.py \
  --pair-evidence "$SANITIZED" \
  --geometry-summary "$SANITIZED_SUMMARY" \
  --output-dir "$ROOT/geometry-analysis" \
  --minimum-q1-calibration-precision 0.90 \
  --minimum-q1-calibration-count 20 \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/analyze-geometry.log"

python - "$ROOT/geometry-analysis/geometry_calibration_decision.json" \
         "$ROOT/geometry-analysis/geometry_queue_assignments.csv" \
  2>&1 <<'PY' | tee "$ROOT/logs/geometry-analysis-gate.json"
import hashlib, json, math, sys
from pathlib import Path

decision=Path(sys.argv[1]); assignments=Path(sys.argv[2])
d=json.loads(decision.read_text())
assert d['status']=='ready_for_blind_review_pack'
assert d['protocol']=='calibration_only_logistic_review_ranking_v1'
assert d['selection_uses_heldout'] is False and d['heldout_is_audit_only'] is True
assert d['q1_is_review_priority_not_automatic_edge'] is True
assert d['formal_grouping_admission'] is False
assert d['queue_counts']=={'Q1':4564,'Q2':518,'Q3':602,'Q4':316}
assert d['calibration_metrics']['count']==402
assert d['calibration_metrics']['positive_count']==186
assert d['calibration_metrics']['predicted_positive_count']==205
assert math.isclose(d['calibration_metrics']['precision'],0.9024390243902439)
assert math.isclose(d['calibration_metrics']['recall'],0.9946236559139785)
assert d['heldout_metrics']['count']==133
assert d['heldout_metrics']['positive_count']==62
assert d['heldout_metrics']['predicted_positive_count']==68
assert math.isclose(d['heldout_metrics']['precision'],0.9117647058823529)
assert math.isclose(d['heldout_metrics']['recall'],1.0)
assignment_sha=hashlib.sha256(assignments.read_bytes()).hexdigest()
assert assignment_sha==d['assignment_sha256']
assert assignment_sha=='3212899813530a5dcfe5bf111aa3664b91a3e9b43e1c845034c568fc8bdaa6f7'
print(json.dumps({'status':'pass','decision':d},indent=2))
PY
```

本步预期与临时 TASK-01 读出数值相同，因为排序特征不使用 affine error；这个一致性是对修订边界的额外校验，不是新的调参。

## 6. 重建盲审包

```bash
python scripts/build_mar20_task01_blind_review.py \
  --pair-evidence "$SANITIZED" \
  --geometry-summary "$SANITIZED_SUMMARY" \
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

python - "$ROOT/blind-review" \
  2>&1 <<'PY' | tee "$ROOT/logs/final-gate.json"
import csv, hashlib, json, sys
from pathlib import Path

p=Path(sys.argv[1])
s=json.loads((p/'blind_review_summary.json').read_text())
d=json.loads((p/'task_decision.json').read_text())
manual=list(csv.DictReader((p/'manual_review_decisions.csv').open(encoding='utf-8',newline='')))
mapping=list(csv.DictReader((p/'blind_mapping_private.csv').open(encoding='utf-8',newline='')))
assert s['status']=='waiting_for_blind_pair_review'
assert (s['new_pair_count'],s['control_pair_count'],s['base_pair_count'])==(300,48,348)
assert (s['blind_duplicate_count'],s['card_count'],s['contact_sheet_count'])==(28,376,94)
assert s['new_pair_grade_counts']=={'Q1':180,'Q2':80,'Q3':40}
assert len(manual)==len(mapping)==376
assert len({r['card_id'] for r in manual})==376
assert all(not r['label'] and not r['confidence'] for r in manual)
assert d['status']=='waiting_for_blind_pair_review'
assert d['formal_grouping_admission'] is False
assert d['next_action']=='complete_manual_review_then_compile_strict_core'
for item in s['artifacts']['contact_sheets']:
    path=p/item['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
print(json.dumps({'technical_status':'pass',
                  'scientific_status':d['status'],
                  'formal_grouping_admission':d['formal_grouping_admission'],
                  'summary':s},indent=2))
PY
```

## 7. 回传包

```bash
tar -czf /workspace/results/MAR20-GROUPING-TASK-01A-return.tar.gz \
  -C "$ROOT" \
  logs \
  geometry-sanitized \
  geometry-analysis \
  blind-review

sha256sum /workspace/results/MAR20-GROUPING-TASK-01A-return.tar.gz \
  | tee "$ROOT/logs/return-package-sha256.txt"
```

回传包不得删除 `pair_evidence_sanitized.csv`、两个决策 JSON、匿名映射、空白审核表、contact sheets 或日志。TASK-01 原始文件和大 cache 保留不动。

## 8. 最终回报要求

必须报告：

1. 原始 SHA、6,000 行、87 个退化字段/29 对的审计结果；
2. 清洗策略、清洗后 SHA、剩余非有限数量；
3. calibration/held-out 指标、Q1～Q4 数量和 assignment SHA；
4. 盲审包的 300/48/28/376/94 完整性；
5. 唯一科学状态 `waiting_for_blind_pair_review`；
6. 回传包路径、大小和 SHA256。

不得宣称已生成正式 `group_id`，也不得将 Q1 自动 union。
