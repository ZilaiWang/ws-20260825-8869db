#!/usr/bin/env bash
# Y2：正式 P2 强基线三折 OOF（材料19 第二优先 / 材料18 C 分工 V1-FULL-P2）
# 只增加 stride-4 P2/P3/P4/P5 检测路径，其余与 M1 完全一致，作单因素对照。
# 依据：configs/experiments/y2_yolo26s_p2_1024_cv3_oof.template.yaml
# 用法：cd /workspace/xh-202625-n2cfg && bash scripts/server/run_y2_p2.sh
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/xh-202625-n2cfg}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results}"

RUN_ROOT="${RESULTS_ROOT}/Y2-P2-CV3-OOF"
PRETRAINED="${PRETRAINED:-/workspace/cv3-model-assets/yolo26s.pt}"
PRETRAINED_SHA="646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
DATA_LOCK="/workspace/results/CV3-DETECTION-DATA-LOCK-TASK-00/formal_detection_data_lock.json"
DATA_LOCK_SHA="03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a"
P02="/root/autodl-tmp/P0-2-exploratory/crop_manifest.csv"
P02_SHA="f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e"
FORMAL_CROP="/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv"
CV3_MANIFEST="data/splits/cv3_airport_proxy_k60_v2.json"
CV3_SHA="27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1

echo "=== 前置 SHA 校验 ==="
test "$(sha256sum "${PRETRAINED}" | awk '{print $1}')" = "${PRETRAINED_SHA}"
test "$(sha256sum "${DATA_LOCK}" | awk '{print $1}')" = "${DATA_LOCK_SHA}"
test "$(sha256sum "${P02}" | awk '{print $1}')" = "${P02_SHA}"
test -f "${RUN_ROOT}/oof_run_plan.json" || {
  echo "缺少 oof_run_plan.json，先跑 prepare_cv3_oof.py（--model-key Y2 --model-family yolo）" >&2
  exit 2
}

for FOLD in 0 1 2; do
  FOLD_DIR="${RUN_ROOT}/fold_${FOLD}"
  VIEW="${FOLD_DIR}/split_view.json"
  echo "=== fold ${FOLD}: 数据锁验证 ==="
  mkdir -p "${FOLD_DIR}/input-gates"
  "${PYTHON_BIN}" scripts/lock_formal_detection_data.py verify \
    --config configs/experiments/formal_detection_data_lock.json \
    --data-root "${DATA_ROOT}" --cv3-manifest "${CV3_MANIFEST}" \
    --p02-manifest "${P02}" --formal-crop-manifest "${FORMAL_CROP}" \
    --lock "${DATA_LOCK}" --expected-lock-sha256 "${DATA_LOCK_SHA}" \
    --report "${FOLD_DIR}/input-gates/detection_data_lock_verification.json" \
    2>&1 | tee "${FOLD_DIR}/detection-data-lock-verify.log"

  echo "=== fold ${FOLD}: materialize 训练配置 ==="
  "${PYTHON_BIN}" scripts/materialize_cv3_oof_config.py \
    --template configs/experiments/y2_yolo26s_p2_1024_cv3_oof.template.yaml \
    --output "${FOLD_DIR}/train_request.yaml" \
    --fold "${FOLD}" --data-root "${DATA_ROOT}" --split-view "${VIEW}" \
    --fold-output-dir "${FOLD_DIR}/training" \
    --pretrained-weight "${PRETRAINED}"

  echo "=== fold ${FOLD}: dry-run ==="
  "${PYTHON_BIN}" scripts/train_cv3_oof.py \
    --config "${FOLD_DIR}/train_request.yaml" --dry-run \
    2>&1 | tee "${FOLD_DIR}/train_dry_run.log"

  echo "=== fold ${FOLD}: 正式训练（160 epoch，长跑）==="
  "${PYTHON_BIN}" scripts/train_cv3_oof.py \
    --config "${FOLD_DIR}/train_request.yaml" \
    2>&1 | tee "${FOLD_DIR}/train.log"
  CHECKPOINT="${FOLD_DIR}/training/runs/foundation/weights/last.pt"
  test -s "${CHECKPOINT}"

  echo "=== fold ${FOLD}: materialize 推理配置 ==="
  "${PYTHON_BIN}" scripts/materialize_cv3_oof_config.py \
    --template configs/experiments/y2_yolo26s_p2_1024_cv3_oof_infer.template.yaml \
    --output "${FOLD_DIR}/resolved_infer.yaml" \
    --fold "${FOLD}" --data-root "${DATA_ROOT}" --split-view "${VIEW}" \
    --fold-output-dir "${FOLD_DIR}" --checkpoint "${CHECKPOINT}"

  echo "=== fold ${FOLD}: 低阈值推理 ==="
  "${PYTHON_BIN}" scripts/infer_cv3_oof.py \
    --config "${FOLD_DIR}/resolved_infer.yaml" \
    2>&1 | tee "${FOLD_DIR}/infer.log"

  {
    date -u +"utc=%Y-%m-%dT%H:%M:%SZ"
    git rev-parse HEAD 2>/dev/null || echo "git_commit=unavailable"
    "${PYTHON_BIN}" --version
    nvidia-smi
  } > "${FOLD_DIR}/environment.txt"

  echo "=== fold ${FOLD}: finalize ==="
  "${PYTHON_BIN}" scripts/finalize_cv3_oof_fold.py \
    --plan "${RUN_ROOT}/oof_run_plan.json" --fold "${FOLD}" \
    --train-config "${FOLD_DIR}/training/resolved_config.yaml" \
    --train-summary "${FOLD_DIR}/training/train_summary.json" \
    --infer-config "${FOLD_DIR}/resolved_infer.yaml" \
    --environment "${FOLD_DIR}/environment.txt" \
    --checkpoint "${CHECKPOINT}" \
    --predictions "${FOLD_DIR}/predictions_low.json" \
    --runtime "${FOLD_DIR}/predictions_low.runtime.json" \
    --data-lock-verification "${FOLD_DIR}/input-gates/detection_data_lock_verification.json" \
    --output "${FOLD_DIR}/fold_metadata.json"
done

echo "=== aggregate 审计 ==="
"${PYTHON_BIN}" scripts/audit_cv3_oof.py \
  --manifest "${CV3_MANIFEST}" --manifest-sha256 "${CV3_SHA}" \
  --plan "${RUN_ROOT}/oof_run_plan.json" --run-root "${RUN_ROOT}" \
  --output-dir "${RESULTS_ROOT}/Y2-P2-CV3-OOF-aggregate" \
  --formal-crop-manifest "${FORMAL_CROP}"

echo "=== 统一评估 + 错误诊断（材料19 第6节统一输出）==="
# 把 aggregate 的 oof_proposals.csv（xywh）转成 evaluate_experiment 的 list JSON
"${PYTHON_BIN}" - "${RESULTS_ROOT}/Y2-P2-CV3-OOF-aggregate/oof_proposals.csv" \
  "${RUN_ROOT}/oof_predictions_list.json" <<'PY'
import csv, json, sys
src, dst = sys.argv[1], sys.argv[2]
rows = []
with open(src, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        x, y, w, h = float(r["x"]), float(r["y"]), float(r["width"]), float(r["height"])
        rows.append({
            "image_id": int(r["image_id"]),
            "category_id": int(r["category_id"]),
            "score": float(r["score"]),
            "bbox_xyxy": [x, y, x + w, y + h],
        })
json.dump(rows, open(dst, "w"), ensure_ascii=False)
print(f"converted {len(rows)} predictions -> {dst}")
PY
"${PYTHON_BIN}" scripts/evaluate_experiment.py \
  --predictions "${RUN_ROOT}/oof_predictions_list.json" \
  --model-key Y2-P2 --output "${RUN_ROOT}/evaluate_Y2-P2.json" > /dev/null
"${PYTHON_BIN}" scripts/analyze_experiment_errors.py \
  --cases "${RUN_ROOT}/evaluate_Y2-P2.cases.json" \
  --output "${RUN_ROOT}/diagnose_Y2-P2.json"

echo "=== 回传包 ==="
tar --exclude='*.pt' --exclude='prepared_data' -czf \
  "${RESULTS_ROOT}/Y2-P2-CV3-OOF-return-no-checkpoints.tar.gz" \
  -C "${RESULTS_ROOT}" Y2-P2-CV3-OOF Y2-P2-CV3-OOF-aggregate
sha256sum "${RESULTS_ROOT}/Y2-P2-CV3-OOF-return-no-checkpoints.tar.gz" \
  | tee "${RESULTS_ROOT}/Y2-P2-CV3-OOF-return-no-checkpoints.tar.gz.sha256"
echo "Y2_P2_CV3_OOF_COMPLETE"
