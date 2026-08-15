#!/usr/bin/env bash
# M3 RT-DETR-L/1024 正式 CV3 三折 OOF（正规补齐训练/推理引擎后）
# 依据：docs/server/M3_CV3_OOF_TASK.md（路径已适配本机 N2-CFG 环境）
# 用法：cd /workspace/xh-202625-n2cfg && bash scripts/server/run_m3_cv3.sh
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/xh-202625-n2cfg}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results}"

RUN_ROOT="${RESULTS_ROOT}/M3-CV3-OOF"
PRETRAINED="${PRETRAINED:-/workspace/cv3-model-assets/rtdetr-l.pt}"
PRETRAINED_SHA="6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f"
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
  echo "缺少 oof_run_plan.json，先跑 prepare_cv3_oof.py" >&2
  exit 2
}

for FOLD in 0 1 2; do
  FOLD_DIR="${RUN_ROOT}/fold_${FOLD}"
  VIEW="${FOLD_DIR}/split_view.json"
  echo "=== fold ${FOLD}: 数据锁验证 ==="
  mkdir -p "${FOLD_DIR}/input-gates"
  "${PYTHON_BIN}" scripts/lock_formal_detection_data.py verify \
    --config configs/experiments/formal_detection_data_lock.json \
    --data-root "${DATA_ROOT}" \
    --cv3-manifest "${CV3_MANIFEST}" \
    --p02-manifest "${P02}" \
    --formal-crop-manifest "${FORMAL_CROP}" \
    --lock "${DATA_LOCK}" \
    --expected-lock-sha256 "${DATA_LOCK_SHA}" \
    --report "${FOLD_DIR}/input-gates/detection_data_lock_verification.json" \
    2>&1 | tee "${FOLD_DIR}/detection-data-lock-verify.log"

  echo "=== fold ${FOLD}: materialize 训练配置 ==="
  "${PYTHON_BIN}" scripts/materialize_cv3_oof_config.py \
    --template configs/experiments/m3_rtdetr_l_1024_cv3_oof.template.yaml \
    --output "${FOLD_DIR}/train_request.yaml" \
    --fold "${FOLD}" --data-root "${DATA_ROOT}" \
    --split-view "${VIEW}" \
    --fold-output-dir "${FOLD_DIR}/training" \
    --pretrained-weight "${PRETRAINED}"

  echo "=== fold ${FOLD}: dry-run ==="
  "${PYTHON_BIN}" scripts/train_cv3_oof.py \
    --config "${FOLD_DIR}/train_request.yaml" --dry-run \
    2>&1 | tee "${FOLD_DIR}/train_dry_run.log"

  echo "=== fold ${FOLD}: 正式训练（120 epoch，长跑）==="
  "${PYTHON_BIN}" scripts/train_cv3_oof.py \
    --config "${FOLD_DIR}/train_request.yaml" \
    2>&1 | tee "${FOLD_DIR}/train.log"
  CHECKPOINT="${FOLD_DIR}/training/runs/foundation/weights/last.pt"
  test -s "${CHECKPOINT}"

  echo "=== fold ${FOLD}: materialize 推理配置 ==="
  "${PYTHON_BIN}" scripts/materialize_cv3_oof_config.py \
    --template configs/experiments/m3_rtdetr_l_1024_cv3_oof_infer.template.yaml \
    --output "${FOLD_DIR}/resolved_infer.yaml" \
    --fold "${FOLD}" --data-root "${DATA_ROOT}" \
    --split-view "${VIEW}" \
    --fold-output-dir "${FOLD_DIR}" \
    --checkpoint "${CHECKPOINT}"

  echo "=== fold ${FOLD}: 低阈值推理 ==="
  "${PYTHON_BIN}" scripts/infer_cv3_oof.py \
    --config "${FOLD_DIR}/resolved_infer.yaml" \
    2>&1 | tee "${FOLD_DIR}/infer.log"

  {
    date -u +"utc=%Y-%m-%dT%H:%M:%SZ"
    git rev-parse HEAD 2>/dev/null || echo "git_commit=unavailable"
    git status --short 2>/dev/null || true
    "${PYTHON_BIN}" --version
    "${PYTHON_BIN}" -m pip freeze
    nvidia-smi
  } > "${FOLD_DIR}/environment.txt"

  echo "=== fold ${FOLD}: finalize ==="
  "${PYTHON_BIN}" scripts/finalize_cv3_oof_fold.py \
    --plan "${RUN_ROOT}/oof_run_plan.json" \
    --fold "${FOLD}" \
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
  --manifest "${CV3_MANIFEST}" \
  --manifest-sha256 "${CV3_SHA}" \
  --plan "${RUN_ROOT}/oof_run_plan.json" \
  --run-root "${RUN_ROOT}" \
  --output-dir "${RESULTS_ROOT}/M3-CV3-OOF-aggregate" \
  --formal-crop-manifest "${FORMAL_CROP}"

echo "=== 回传包 ==="
tar --exclude='*.pt' --exclude='prepared_data' -czf \
  "${RESULTS_ROOT}/M3-CV3-OOF-return-no-checkpoints.tar.gz" \
  -C "${RESULTS_ROOT}" M3-CV3-OOF M3-CV3-OOF-aggregate
sha256sum "${RESULTS_ROOT}/M3-CV3-OOF-return-no-checkpoints.tar.gz" \
  | tee "${RESULTS_ROOT}/M3-CV3-OOF-return-no-checkpoints.tar.gz.sha256"
echo "M3_CV3_OOF_COMPLETE"
