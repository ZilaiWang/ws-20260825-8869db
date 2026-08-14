#!/usr/bin/env bash
# N2-CFG：粗类条件式前景门控 三折训练 + S0/S1/S2 快筛（一键）
# 来源：reports/experiments/N2_CFG_BACKGROUND_GATE_PLAN_20260814.md（2026-08-14 冻结）
# 权威路线：《改进方案 1》第 2、3、5 节
# 用法：cd /workspace/xh-202625 && bash scripts/server/run_n2_cfg.sh
#
# 前置（本地完成，B 回传 CSV 后）：
#   1. 本地 compile_fp_bg_review.py 编译 clear_background_whitelist.csv（一致率>=0.85）；
#   2. 把 whitelist 上传到本脚本 WHITELIST 指向的服务器路径；
#   3. 确认 DATA_ROOT 指向 4,481 张源图所在目录。
#
# 本脚本只做 background_reject；舰船/车辆正式门控，飞机 shadow 旁路。
set -Eeuo pipefail

# ===== 路径约定（可用环境变量覆盖）=====
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/xh-202625}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p03-cu121/bin/python}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the 4481 source images dir}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results}"

RUN_ID="N2-CFG-BACKGROUND-GATE-V1"
RUN_ROOT="${RESULTS_ROOT}/${RUN_ID}"
STATUS_PATH="${RESULTS_ROOT}/${RUN_ID}.status"
LOCK_DIR="${RESULTS_ROOT}/.${RUN_ID}.lock"

CONFIG="${PROJECT_ROOT}/configs/experiments/n2_cfg_background_gate_v1.yaml"
PROJECT_CONFIG="${PROJECT_ROOT}/configs/project.yaml"
CODE_LOCK="${PROJECT_ROOT}/docs/server/N2_CFG_CODE_SHA256.txt"

# ===== 输入资产（相对 PROJECT_ROOT，SHA 与 n2_cfg 合同一致）=====
SELECTED_PREDICTIONS="${PROJECT_ROOT}/outputs/R1-6-POST-RERANK-NMS/selected_predictions_xyxy.json"
FORMAL_CROP_MANIFEST="${PROJECT_ROOT}/outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv"
IMAGE_LEDGER="${PROJECT_ROOT}/outputs/M1-CV3-OOF-return-no-checkpoints-extracted-20260725/M1-CV3-OOF-aggregate/oof_images.csv"
CONVNEXT_WEIGHTS="${PROJECT_ROOT}/weights/convnext_tiny_imagenet1k_v1.pth"
WHITELIST="${PROJECT_ROOT}/outputs/N0-FP-BG-AUDIT-R1-6-V3/compiled/clear_background_whitelist.csv"

SELECTED_SHA="d07f43e5a1b610d44cdedbedd844719e003aa6e978380fb2257093648975d047"
MANIFEST_SHA="a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128"
LEDGER_SHA="fc2aa7ca947f71d841700b656ece5e90f3112746e0c3f592a78a231d958a750c"
WEIGHTS_SHA="983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "N2-CFG lock already exists: ${LOCK_DIR}" >&2
  exit 1
fi
cleanup_lock() { rmdir "${LOCK_DIR}" 2>/dev/null || true; }
trap cleanup_lock EXIT
on_error() {
  local code=$?
  echo "failed_exit_${code}" > "${STATUS_PATH}"
  exit "${code}"
}
trap on_error ERR

mkdir -p "${RESULTS_ROOT}"
if [[ -e "${RUN_ROOT}" ]]; then
  echo "N2-CFG run root exists; refuse overwrite: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/manifest" "${RUN_ROOT}/train" \
  "${RUN_ROOT}/infer" "${RUN_ROOT}/evaluation"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== [1/7] 环境 preflight + 代码 SHA + pytest + ruff ==="
{
  date -Is
  git rev-parse HEAD
  git status --short
  nvidia-smi
  "${PYTHON_BIN}" --version
  "${PYTHON_BIN}" -c "import torch,torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda)"
  df -h /workspace
} 2>&1 | tee "${RUN_ROOT}/logs/system_preflight.txt"

sha256sum --check "${CODE_LOCK}" 2>&1 | tee "${RUN_ROOT}/logs/code-sha256.log"

"${PYTHON_BIN}" -m pytest -q tests/test_background_gate.py tests/test_background_gate_fit.py \
  2>&1 | tee "${RUN_ROOT}/logs/pytest.log"

"${PYTHON_BIN}" -m ruff check \
  src/rsdet/analysis/background_gate.py \
  src/rsdet/analysis/background_gate_manifest.py \
  src/rsdet/models/background_gate_classifier.py \
  scripts/build_bg_gate_manifest.py scripts/train_bg_gate.py \
  scripts/infer_bg_gate_logits.py scripts/evaluate_bg_gate.py \
  tests/test_background_gate.py tests/test_background_gate_fit.py \
  2>&1 | tee "${RUN_ROOT}/logs/ruff.log"

echo "=== [2/7] 输入资产 SHA 门禁 + 白名单检查 ==="
for asset in \
  "${SELECTED_PREDICTIONS}:${SELECTED_SHA}" \
  "${FORMAL_CROP_MANIFEST}:${MANIFEST_SHA}" \
  "${IMAGE_LEDGER}:${LEDGER_SHA}" \
  "${CONVNEXT_WEIGHTS}:${WEIGHTS_SHA}"; do
  path="${asset%%:*}"
  expected="${asset##*:}"
  [[ -f "${path}" ]] || { echo "missing required asset: ${path}" >&2; exit 1; }
  actual="$(sha256sum "${path}" | cut -d' ' -f1)"
  [[ "${actual}" = "${expected}" ]] || {
    echo "SHA mismatch for ${path}: expected=${expected} actual=${actual}" >&2
    exit 1
  }
done
if [[ ! -f "${WHITELIST}" ]]; then
  echo "waiting_for_clear_background_whitelist: ${WHITELIST}" >&2
  echo "本地尚未编译 N0 盲审白名单（等 B 回传 manual_review_decisions.csv 后执行 compile_fp_bg_review.py）" >&2
  exit 2
fi
echo "INPUT_ASSETS_VERIFIED" | tee "${RUN_ROOT}/logs/input-assets-accepted.log"

echo "=== [3/7] 构建 foreground gate manifest ==="
echo "building_manifest" > "${STATUS_PATH}"
"${PYTHON_BIN}" scripts/build_bg_gate_manifest.py \
  --config "${CONFIG}" --project-config "${PROJECT_CONFIG}" \
  --clear-background-whitelist "${WHITELIST}" \
  --output "${RUN_ROOT}/manifest/gate_manifest.csv" \
  2>&1 | tee "${RUN_ROOT}/logs/build-manifest.log"
MANIFEST="${RUN_ROOT}/manifest/gate_manifest.csv"
[[ -s "${MANIFEST}" ]] || { echo "gate manifest 为空" >&2; exit 1; }

echo "=== [4/7] 三折训练（freeze_backbone, 5 epoch, seed 202625）==="
for fold in 0 1 2; do
  echo "training_fold_${fold}" > "${STATUS_PATH}"
  "${PYTHON_BIN}" scripts/train_bg_gate.py \
    --manifest "${MANIFEST}" --data-root "${DATA_ROOT}" \
    --convnext-weights "${CONVNEXT_WEIGHTS}" \
    --output-dir "${RUN_ROOT}/train" \
    --held-out-fold "${fold}" \
    --freeze freeze_backbone \
    --epochs 5 --batch-size 64 --batches-per-epoch 200 \
    --learning-rate 0.001 --seed 202625 \
    --device cuda --verify-weight-sha256 \
    2>&1 | tee "${RUN_ROOT}/logs/train-fold${fold}.log"
  test "${PIPESTATUS[0]}" -eq 0 || exit 1
done

echo "=== [5/7] 三折前景 logit 推理 ==="
echo "inferring_fg_logits" > "${STATUS_PATH}"
"${PYTHON_BIN}" scripts/infer_bg_gate_logits.py \
  --predictions "${SELECTED_PREDICTIONS}" \
  --formal-crop-manifest "${FORMAL_CROP_MANIFEST}" \
  --image-ledger "${IMAGE_LEDGER}" \
  --data-root "${DATA_ROOT}" \
  --convnext-weights "${CONVNEXT_WEIGHTS}" \
  --checkpoint-dir "${RUN_ROOT}/train" \
  --output "${RUN_ROOT}/infer/fg_logits.json" \
  --batch-size 256 --device cuda --freeze freeze_backbone \
  2>&1 | tee "${RUN_ROOT}/logs/infer.log"
FG_LOGITS="${RUN_ROOT}/infer/fg_logits.json"

echo "=== [6/7] S0 / S1 / S2 cross-fit 快筛（recall_budget=0）==="
for mode in S0 S1 S2; do
  echo "evaluating_${mode}" > "${STATUS_PATH}"
  "${PYTHON_BIN}" scripts/evaluate_bg_gate.py \
    --predictions "${SELECTED_PREDICTIONS}" \
    --fg-logits "${FG_LOGITS}" \
    --formal-crop-manifest "${FORMAL_CROP_MANIFEST}" \
    --image-ledger "${IMAGE_LEDGER}" \
    --project-config "${PROJECT_CONFIG}" \
    --output-dir "${RUN_ROOT}/evaluation" \
    --recall-budget 0 --mode "${mode}" \
    2>&1 | tee "${RUN_ROOT}/logs/evaluate-${mode}.log"
  test "${PIPESTATUS[0]}" -eq 0 || exit 1
done

echo "=== [7/7] 门禁判定（S2 须在相同 Recall 约束下优于 S0）==="
echo "running_admission" > "${STATUS_PATH}"
"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
eval_dir = root / "evaluation"
modes = ("S0", "S1", "S2")
data = {}
for mode in modes:
    data[mode] = json.loads((eval_dir / f"evaluate_{mode}.json").read_text())

def pooled_removed_fp_bg(mode_json: dict) -> int:
    return sum(fold["fit"]["removed_fp_bg"] for fold in mode_json["per_fold"])

def max_removed_tp(mode_json: dict) -> int:
    return max(fold["fit"]["removed_tp"] for fold in mode_json["per_fold"])

fp_bg_removed = {mode: pooled_removed_fp_bg(data[mode]) for mode in modes}
admission = {
    "fp_bg_removed_pooled_inner_crossfit": fp_bg_removed,
    "s2_vs_s0_delta": fp_bg_removed["S2"] - fp_bg_removed["S0"],
    "s2_beats_s0": fp_bg_removed["S2"] > fp_bg_removed["S0"],
    "zero_tp_loss_all_modes": all(max_removed_tp(data[mode]) == 0 for mode in modes),
    "note": (
        "removed_fp_bg 为 leave-one-fold-out 的 inner 拟合删除数（零 TP 损失预算）；"
        "正式 pooled/逐类/来源稳健性门禁须由本地基于 per_fold 完整 JSON 复算。"
    ),
}
(out := (eval_dir / "admission.json")).write_text(
    json.dumps(admission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(admission, ensure_ascii=False, indent=2))
PY

echo "complete" > "${STATUS_PATH}"

find "${RUN_ROOT}/train" -type f -name 'bg_gate_fold*_final.pt' -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/CHECKPOINTS_SHA256.txt"

RETURN_PACKAGE="${RESULTS_ROOT}/${RUN_ID}-return-no-checkpoints.tar.gz"
tar -C "${RESULTS_ROOT}" --exclude='bg_gate_fold*_final.pt' -czf "${RETURN_PACKAGE}" "${RUN_ID}"
sha256sum "${RETURN_PACKAGE}" > "${RETURN_PACKAGE}.sha256"
echo "N2-CFG complete: ${RETURN_PACKAGE}"
