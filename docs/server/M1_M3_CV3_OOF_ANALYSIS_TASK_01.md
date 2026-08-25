# M1-M3-OOF-ANALYSIS-TASK-01 服务器任务单

状态：`waiting_for_both_formal_oof_aggregates`  
资源：CPU；禁止占用 GPU  
科学方案：`reports/experiments/M1_M3_CV3_OOF_POSTPROCESS_ANALYSIS_PLAN_v1.md`

## 1. 前置条件

本任务只有在以下目录均已由各自任务单完成后才启动：

```text
/workspace/results/M1-CV3-OOF-aggregate
/workspace/results/M3-CV3-OOF-aggregate
/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv
```

缺少任一输入时，写 `waiting_for_both_formal_oof_aggregates` 并停止；不得用
smoke、单折预测或 diagnostic aggregate 补位。

## 2. 代码与环境门禁

```bash
set -euo pipefail
cd /workspace/xh-202625
source /workspace/venvs/cv3-model-cu121/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

TASK_ID=M1-M3-OOF-ANALYSIS-TASK-01
RESULT=/workspace/results/$TASK_ID
FORMAL=/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv
M1=/workspace/results/M1-CV3-OOF-aggregate
M3=/workspace/results/M3-CV3-OOF-aggregate
FORMAL_SHA=a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128

for PATH_REQUIRED in \
  "$FORMAL" \
  "$M1/oof_metadata.json" "$M1/oof_images.csv" \
  "$M1/oof_proposals.csv" "$M1/predictions_oof_low.json" \
  "$M3/oof_metadata.json" "$M3/oof_images.csv" \
  "$M3/oof_proposals.csv" "$M3/predictions_oof_low.json"; do
  if [ ! -s "$PATH_REQUIRED" ]; then
    mkdir -p /workspace/results/$TASK_ID-waiting
    printf 'status=waiting_for_both_formal_oof_aggregates\nmissing=%s\n' \
      "$PATH_REQUIRED" \
      > /workspace/results/$TASK_ID-waiting/task_status.txt
    exit 20
  fi
done

test ! -e "$RESULT" || {
  echo "结果目录已存在，禁止覆盖: $RESULT" >&2
  exit 2
}
echo "$FORMAL_SHA  $FORMAL" | sha256sum -c -
sha256sum -c docs/server/CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt
sha256sum -c \
  docs/server/M1_M3_CV3_OOF_ANALYSIS_TASK_01_CODE_SHA256.txt

python -m pytest -q \
  tests/test_official_metric.py \
  tests/test_threshold_sweep.py \
  tests/test_oof_detection_analysis.py
python -m ruff check \
  src/rsdet/evaluation/official_metric.py \
  src/rsdet/analysis/oof_detection.py \
  scripts/analyze_cv3_oof_models.py \
  tests/test_official_metric.py \
  tests/test_oof_detection_analysis.py
```

## 3. 输入 metadata 快速门禁

```bash
python - "$M1/oof_metadata.json" "$M3/oof_metadata.json" <<'PY'
import json
import sys

expected = ("M1", "M3")
for path, model_key in zip(sys.argv[1:], expected):
    payload = json.load(open(path, encoding="utf-8"))
    assert payload["contract_version"] == "cv3_oof_v1"
    assert payload["status"] == "complete_downstream_ready"
    assert payload["downstream_admission"] is True
    assert payload["model_key"] == model_key
    assert payload["source_manifest_sha256"] == (
        "27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331"
    )
    assert payload["image_count"] == 4481
    assert payload["low_score_threshold"] == 0.001
    assert payload["formal_crop_manifest"]["sha256"] == (
        "a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128"
    )
print("OOF_AGGREGATE_METADATA_PREFLIGHT_PASS")
PY
```

该段只作快速错误提示；正式脚本还会重算 aggregate 三件套的 SHA、检查
4481 行 image ledger、model key、candidate floor 和 image ID 集合。

## 4. 正式运行

```bash
mkdir -p "$RESULT"
{
  date -Is
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true
  python --version
  python -m pip freeze
} > "$RESULT/environment.txt"

python scripts/analyze_cv3_oof_models.py \
  --config configs/experiments/m1_m3_cv3_oof_analysis_v1.yaml \
  --project-config configs/project.yaml \
  --formal-crop-manifest "$FORMAL" \
  --m1-aggregate "$M1" \
  --m3-aggregate "$M3" \
  --output-dir "$RESULT/analysis" \
  2>&1 | tee "$RESULT/run.log"
```

注意：脚本要求 output directory 原先不存在或为空。本任务先创建的是
`$RESULT`，不是 `$RESULT/analysis`，因此不会破坏不可覆盖门禁。

## 5. 自动验收

```bash
python - "$RESULT/analysis/analysis_metadata.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["analysis_version"] == "m1_m3_cv3_oof_analysis_v1"
assert payload["status"] == "complete_formal_oof_descriptive_analysis"
assert payload["counts"] == {
    "images": 4481,
    "ground_truth_objects": 20933,
}
scope = payload["scientific_claim_scope"]
assert scope["official_matching_metrics"] is True
assert scope["error_decomposition_is_diagnostic"] is True
assert scope["same_oof_selected_threshold_is_final"] is False
assert scope["oracle_union_is_deployable"] is False
for model in ("M1", "M3"):
    assert payload["models"][model]["curve_parity"]["status"] == "pass"
    assert (
        payload["models"][model]["aggregate_cross_file_audit"]["status"]
        == "pass"
    )
    assert payload["models"][model]["relocated_artifacts"] == []
    assert (
        payload["models"][model]["exploratory_workpoint"]
        ["deployment_admission"]
        is False
    )
print("M1_M3_OOF_ANALYSIS_ACCEPTANCE_PASS")
PY

test "$(wc -l < \
  "$RESULT/analysis/paired/object_outcomes_candidate_floor.csv")" = 20934
test "$(wc -l < \
  "$RESULT/analysis/paired/object_outcomes_exploratory_workpoints.csv")" = 20934
```

## 6. 打包

```bash
tar -czf /workspace/results/$TASK_ID-return.tar.gz \
  -C /workspace/results "$TASK_ID"
sha256sum /workspace/results/$TASK_ID-return.tar.gz \
  | tee /workspace/results/$TASK_ID-return.tar.gz.sha256
```

本任务不含 checkpoint 或特征 cache，整个结果包应完整回传。

## 7. 服务器最终回报格式

1. 状态；
2. Git commit/dirty、task code SHA、pytest 数、ruff；
3. M1/M3 aggregate metadata 与 prediction SHA；
4. formal crop SHA、图像数、GT 数；
5. 两个模型 candidate-floor Recall/FDR；
6. 两个模型描述性工作点阈值、Recall/FDR，明确
   `deployment_admission=false`；
7. 两组 `FP_DUP/FP_CLS/FP_LOC/FP_BG` 和
   `FN_CLS/FN_LOC/FN_MISS`，并确认守恒；
8. candidate floor 与描述性工作点的 both/M1-only/M3-only/neither 和
   oracle recall；
9. 所有停止条件、失败与重试；
10. 回传包路径、大小与 SHA。

禁止把 oracle recall 写成集成模型成绩；禁止把 same-OOF 工作点写成最终无偏
阈值。
