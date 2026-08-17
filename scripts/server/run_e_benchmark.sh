#!/usr/bin/env bash
# E 正式测速一键脚本：M1 正式 last.pt + 合成 10K 图 + 3 warmup + 10 measured。
#
# 用法：
#   bash scripts/server/run_e_benchmark.sh
#
# 前置（本脚本第 2 步逐项校验）：
#   $CHECKPOINT        M1 fold0 正式 last.pt（SHA d403ca0d…，先 scp 到服务器）
#   $FOLD_METADATA     fold_0/fold_metadata.json（SHA b2bd717d…）
#   $OOF_METADATA      oof_metadata.json（SHA 53b35f2c…）
#   $DATA_ROOT         合成 10K 图目录（>=10 张 10000x10000，内容互异）
#
# 产出（$RUN_ROOT）：
#   resolved_config.yaml / image_manifest.json / checkpoint_provenance.json
#   hardware.json / benchmark_contract.json / runtime_samples.jsonl
#   audit.json / predictions_10k_low.json + 回传包
#
# 口径：image_source_type=synthetic → 结论为「工程正式化 smoke」
# （3 warmup + 10 measured + 全冻结合同），非 real_official 官方时延。
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/xh-202625-n2cfg}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/p06-cu121}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results}"
PYTHON_BIN="${VENV_DIR}/bin/python"
RUN_ID="E-FORMAL-BENCHMARK"
RUN_ROOT="${RESULTS_ROOT}/${RUN_ID}"
CAPTURE_DIR="${RUN_ROOT}/capture"
STATUS_PATH="${RUN_ROOT}/status.txt"

CHECKPOINT="${CHECKPOINT:-/workspace/cv3-model-assets/m1_fold0_last.pt}"
FOLD_METADATA="${FOLD_METADATA:-/workspace/cv3-model-assets/fold_0_fold_metadata.json}"
OOF_METADATA="${OOF_METADATA:-/workspace/cv3-model-assets/oof_metadata.json}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/10k}"
MODEL_KEY="M1"
TILE_SIZE=1280
OVERLAP=256
EXPECTED_TILE_COUNT=100
WARMUP_RUNS=3
MEASURED_RUNS=10

EXPECTED_CHECKPOINT_SHA="d403ca0d5f760f8d7271af33c467426a3bd8ce8095b6d13e5dc63a6d8e19501d"
EXPECTED_FOLD_META_SHA="b2bd717d1766ab4fbf870901e56ef6327b81e4d56dbcb6958657ac1c0261af81"
EXPECTED_OOF_META_SHA="53b35f2cff801ce23d9bd211ead9a0ef896e3a01f974b5da060fd2a4ac4bf4c6"

mkdir -p "${RUN_ROOT}/logs"
log() { echo "[E-BENCH] $*" | tee -a "${RUN_ROOT}/logs/run.log"; }

log "=== [1/9] 环境 preflight ==="
echo "running_preflight" > "${STATUS_PATH}"
test -x "${PYTHON_BIN}" || { echo "venv 不存在: ${PYTHON_BIN}"; exit 1; }
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src"
export PYTHONNOUSERSITE=1
"${PYTHON_BIN}" -c "import torch, ultralytics, yaml, PIL, numpy; print('deps OK', 'torch', torch.__version__, 'ultralytics', ultralytics.__version__)"

log "=== [2/9] 输入资产 SHA 校验 + checkpoint 就位 ==="
test -f "${CHECKPOINT}" || { echo "缺少 checkpoint: ${CHECKPOINT}"; exit 1; }
test -f "${FOLD_METADATA}" || { echo "缺少 fold_metadata: ${FOLD_METADATA}"; exit 1; }
test -f "${OOF_METADATA}" || { echo "缺少 oof_metadata: ${OOF_METADATA}"; exit 1; }
CKPT_SHA=$("${PYTHON_BIN}" -c "import hashlib;print(hashlib.sha256(open('${CHECKPOINT}','rb').read()).hexdigest())")
FOLD_SHA=$("${PYTHON_BIN}" -c "import hashlib;print(hashlib.sha256(open('${FOLD_METADATA}','rb').read()).hexdigest())")
OOF_SHA=$("${PYTHON_BIN}" -c "import hashlib;print(hashlib.sha256(open('${OOF_METADATA}','rb').read()).hexdigest())")
log "checkpoint SHA: ${CKPT_SHA:0:12}… (期望 ${EXPECTED_CHECKPOINT_SHA:0:12}…)"
log "fold_meta SHA:  ${FOLD_SHA:0:12}… (期望 ${EXPECTED_FOLD_META_SHA:0:12}…)"
log "oof_meta SHA:   ${OOF_SHA:0:12}… (期望 ${EXPECTED_OOF_META_SHA:0:12}…)"
test "${CKPT_SHA}" = "${EXPECTED_CHECKPOINT_SHA}" || { echo "checkpoint SHA 不符"; exit 1; }
test "${FOLD_SHA}" = "${EXPECTED_FOLD_META_SHA}" || { echo "fold_metadata SHA 不符"; exit 1; }
test "${OOF_SHA}" = "${EXPECTED_OOF_META_SHA}" || { echo "oof_metadata SHA 不符"; exit 1; }
# benchmark 校验要求 fold_metadata.artifacts.checkpoint 解析后 == 传入 checkpoint。
# fold_metadata 的 SHA 是冻结的（内容不可改），因此把校验过的 checkpoint 复制到
# fold_metadata 记录的原始路径，作为正式测速 checkpoint。
CKPT_TARGET=$("${PYTHON_BIN}" -c "
import json
print(json.load(open('${FOLD_METADATA}'))['artifacts']['checkpoint'])
")
test -n "${CKPT_TARGET}" || { echo "无法从 fold_metadata 解析 checkpoint 路径"; exit 1; }
mkdir -p "$(dirname "${CKPT_TARGET}")"
cp "${CHECKPOINT}" "${CKPT_TARGET}"
CHECKPOINT="${CKPT_TARGET}"
log "checkpoint 就位于 fold_metadata 记录路径: ${CHECKPOINT}"

log "=== [3/9] GPU 独占检查 + 合成图检查 ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
# 禁止训练进程占 GPU（measured 期间不允许其他 GPU 计算进程）
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qE '[0-9]+'; then
  echo "检测到其他 GPU 计算进程，正式测速要求独占 GPU。请先停止训练。"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi
N_10K=$("${PYTHON_BIN}" -c "
from PIL import Image
from pathlib import Path
import glob
n = 0
for p in sorted(Path('${DATA_ROOT}').rglob('*')):
    if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        w, h = Image.open(p).size
        if (w, h) == (10000, 10000):
            n += 1
print(n)
")
log "data_root=${DATA_ROOT} 下 10000x10000 图数: ${N_10K}（需 >= ${MEASURED_RUNS}）"
test "${N_10K}" -ge "${MEASURED_RUNS}" || { echo "合成 10K 图不足"; exit 1; }

log "=== [4/9] 生成 resolved config + image manifest + provenance ==="
"${PYTHON_BIN}" - "${RUN_ROOT}" "${DATA_ROOT}" "${CHECKPOINT}" <<'PY'
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
data_root = Path(sys.argv[2])
checkpoint = sys.argv[3]
template = Path("/workspace/xh-202625-n2cfg/configs/experiments/e_10k_pipeline_cv3.template.yaml")
text = template.read_text(encoding="utf-8")
text = text.replace("__MODEL_FAMILY__", "yolo")
text = text.replace("__SELECTED_CHECKPOINT__", checkpoint)
text = text.replace("__10K_DATA_ROOT__", str(data_root))
text = text.replace("__10K_MANIFEST__", str(run_root / "image_manifest.json"))
text = text.replace("__OUTPUT_DIR__", str(run_root / "capture"))
(run_root / "resolved_config.yaml").write_text(text, encoding="utf-8")
print("resolved_config.yaml written")
PY
"${PYTHON_BIN}" scripts/build_e_10k_manifest.py \
  --data-root "${DATA_ROOT}" --source-type synthetic \
  --min-count "${MEASURED_RUNS}" --output "${RUN_ROOT}/image_manifest.json"
"${PYTHON_BIN}" scripts/build_e_checkpoint_provenance.py \
  --checkpoint "${CHECKPOINT}" --fold-metadata "${FOLD_METADATA}" \
  --oof-metadata "${OOF_METADATA}" --model-key "${MODEL_KEY}" --fold 0 \
  --output "${RUN_ROOT}/checkpoint_provenance.json"

log "=== [5/9] 现场采集 hardware.json ==="
"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json
import subprocess
import sys
import torch
from pathlib import Path

run_root = Path(sys.argv[1])
def smi(*query_fields):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + ",".join(query_fields), "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]
    return [part.strip() for part in out.split(",")]

gpu_name, driver_version, _ = smi("name", "driver_version", "memory.total")
apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip()
hardware = {
    "gpu_name": gpu_name,
    "driver_version": driver_version,
    "torch_version": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "other_gpu_processes": apps if apps else "none",
    "collected_at_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                       capture_output=True, text=True).stdout.strip(),
}
(run_root / "hardware.json").write_text(json.dumps(hardware, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(hardware, ensure_ascii=False, indent=2))
PY

log "=== [6/9] 生成 benchmark contract ==="
"${PYTHON_BIN}" scripts/build_e_benchmark_contract.py \
  --image-manifest "${RUN_ROOT}/image_manifest.json" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-provenance "${RUN_ROOT}/checkpoint_provenance.json" \
  --config "${RUN_ROOT}/resolved_config.yaml" \
  --hardware "${RUN_ROOT}/hardware.json" \
  --model-key "${MODEL_KEY}" --image-source-type synthetic \
  --tile-size "${TILE_SIZE}" --overlap "${OVERLAP}" \
  --expected-tile-count "${EXPECTED_TILE_COUNT}" \
  --warmup-runs "${WARMUP_RUNS}" --minimum-measured-runs "${MEASURED_RUNS}" \
  --output "${RUN_ROOT}/benchmark_contract.json"

log "=== [7/9] 正式采集（3 warmup + 10 measured，独占 GPU）==="
echo "running_capture" > "${STATUS_PATH}"
"${PYTHON_BIN}" scripts/benchmark_10k_pipeline.py \
  --config "${RUN_ROOT}/resolved_config.yaml" \
  --benchmark-contract "${RUN_ROOT}/benchmark_contract.json" \
  --checkpoint-provenance "${RUN_ROOT}/checkpoint_provenance.json" \
  --image-manifest "${RUN_ROOT}/image_manifest.json" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${CAPTURE_DIR}" \
  --expected-width 10000 --expected-height 10000 \
  2>&1 | tee "${RUN_ROOT}/logs/capture.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo "采集失败"; exit 1; }

log "=== [8/9] audit 审计 ==="
"${PYTHON_BIN}" scripts/audit_10k_runtime.py \
  --input "${CAPTURE_DIR}/runtime_samples.jsonl" \
  --hardware "${RUN_ROOT}/hardware.json" \
  --benchmark-contract "${RUN_ROOT}/benchmark_contract.json" \
  --output "${CAPTURE_DIR}/audit.json" \
  --expected-width 10000 --expected-height 10000 \
  --minimum-measured-runs "${MEASURED_RUNS}" \
  --maximum-after-read-seconds 20.0 \
  2>&1 | tee "${RUN_ROOT}/logs/audit.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo "audit 失败"; exit 1; }

log "=== [9/9] 回传包 ==="
echo "complete" > "${STATUS_PATH}"
RETURN_PACKAGE="${RESULTS_ROOT}/${RUN_ID}-return.tar.gz"
tar -C "${RESULTS_ROOT}" -czf "${RETURN_PACKAGE}" \
  "${RUN_ID}/resolved_config.yaml" "${RUN_ID}/image_manifest.json" \
  "${RUN_ID}/checkpoint_provenance.json" "${RUN_ID}/hardware.json" \
  "${RUN_ID}/benchmark_contract.json" \
  "${RUN_ID}/capture/runtime_samples.jsonl" "${RUN_ID}/capture/audit.json" \
  "${RUN_ID}/logs" "${RUN_ID}/status.txt"
sha256sum "${RETURN_PACKAGE}" > "${RETURN_PACKAGE}.sha256"
log "E 正式测速完成: ${RETURN_PACKAGE}"
log "audit 摘要:"
"${PYTHON_BIN}" -c "
import json
d = json.load(open('${CAPTURE_DIR}/audit.json'))
print(json.dumps(d, ensure_ascii=False, indent=2)[:2000])
"
