#!/usr/bin/env bash
# E 组合系统（M1 + M3 串行）10K 正式测速：预算加总口径。
#
# 组合系统 = M1（YOLO26s）+ M3（RT-DETR-L）串行流水线，同一张 10K 图
# 依次过两模型后融合。组合 after-read 预算 = M1 after-read + M3 after-read
# （逐 run 相加），门禁 p50/p95/max <= 20s。
#
# 流程：
#   1. MODEL_KEY=M1 跑 run_e_benchmark.sh（3 warmup + 10 measured，正式合同）；
#   2. MODEL_KEY=M3 跑 run_e_benchmark.sh（同图集，正式合同）；
#   3. audit_combined_runtime.py 逐图加总 → 组合审计；
#   4. 打组合回传包。
#
# 用法（服务器，GPU 独占）：
#   bash scripts/server/run_e_benchmark_combined.sh
#
# 前置：M3 资产环境变量（同 run_e_benchmark.sh M3 模式）：
#   CHECKPOINT_M3 / FOLD_METADATA_M3 / OOF_METADATA_M3 及对应 EXPECTED_*_SHA
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/xh-202625-n2cfg}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"

COMBINED_RUN_ID="E-FORMAL-BENCHMARK-COMBINED"
COMBINED_ROOT="${RESULTS_ROOT}/${COMBINED_RUN_ID}"
M1_ROOT="${RESULTS_ROOT}/E-FORMAL-BENCHMARK-M1"
M3_ROOT="${RESULTS_ROOT}/E-FORMAL-BENCHMARK-M3"
mkdir -p "${COMBINED_ROOT}/logs"
log() { echo "[E-COMBINED] $*" | tee -a "${COMBINED_ROOT}/logs/run.log"; }

log "=== [1/4] M1 正式测速（复用 run_e_benchmark.sh）==="
MODEL_KEY=M1 bash "${PROJECT_ROOT}/scripts/server/run_e_benchmark.sh"

log "=== [2/4] M3 正式测速（换 checkpoint + rtdetr family）==="
MODEL_KEY=M3 \
  CHECKPOINT="${CHECKPOINT_M3:?}" \
  FOLD_METADATA="${FOLD_METADATA_M3:?}" \
  OOF_METADATA="${OOF_METADATA_M3:?}" \
  EXPECTED_CHECKPOINT_SHA="${EXPECTED_CHECKPOINT_SHA_M3:?}" \
  EXPECTED_FOLD_META_SHA="${EXPECTED_FOLD_META_SHA_M3:?}" \
  EXPECTED_OOF_META_SHA="${EXPECTED_OOF_META_SHA_M3:?}" \
  bash "${PROJECT_ROOT}/scripts/server/run_e_benchmark.sh"

log "=== [3/4] 组合审计（逐图 M1+M3 after-read 加总）==="
"${PYTHON_BIN}" scripts/audit_combined_runtime.py \
  --m1-samples "${M1_ROOT}/capture/runtime_samples.jsonl" \
  --m3-samples "${M3_ROOT}/capture/runtime_samples.jsonl" \
  --output "${COMBINED_ROOT}/audit_combined.json" \
  2>&1 | tee "${COMBINED_ROOT}/logs/combined-audit.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo "组合审计未过（>20s）"; exit 1; }

log "=== [4/4] 组合回传包 ==="
cp "${M1_ROOT}/benchmark_contract.json" "${COMBINED_ROOT}/contract_M1.json"
cp "${M3_ROOT}/benchmark_contract.json" "${COMBINED_ROOT}/contract_M3.json"
cp "${M1_ROOT}/hardware.json" "${COMBINED_ROOT}/hardware_M1.json"
cp "${M3_ROOT}/hardware.json" "${COMBINED_ROOT}/hardware_M3.json"
echo "complete" > "${COMBINED_ROOT}/status.txt"
RETURN_PACKAGE="${RESULTS_ROOT}/${COMBINED_RUN_ID}-return.tar.gz"
tar -C "${RESULTS_ROOT}" -czf "${RETURN_PACKAGE}" \
  "${COMBINED_RUN_ID}/audit_combined.json" \
  "${COMBINED_RUN_ID}/contract_M1.json" "${COMBINED_RUN_ID}/contract_M3.json" \
  "${COMBINED_RUN_ID}/hardware_M1.json" "${COMBINED_RUN_ID}/hardware_M3.json" \
  "${COMBINED_RUN_ID}/logs" "${COMBINED_RUN_ID}/status.txt"
sha256sum "${RETURN_PACKAGE}" > "${RETURN_PACKAGE}.sha256"
log "组合测速完成: ${RETURN_PACKAGE}"
"${PYTHON_BIN}" -c "
import json
d = json.load(open('${COMBINED_ROOT}/audit_combined.json'))
print(json.dumps(d, ensure_ascii=False, indent=2)[:1200])
"
