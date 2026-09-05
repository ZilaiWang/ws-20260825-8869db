#!/usr/bin/env bash
set -Eeuo pipefail

# Run on the 3-GPU host.  As soon as the sanitized full hierarchy asset and its
# validated config exist, copy them directly to the RTX 3090 host and start a
# latency/parity diagnostic there.  This never builds a Docker image or submits.
SOURCE=${SOURCE:-/root/autodl-tmp/results/P40-HIER-S256V128-AIRCRAFT-D4-FULL-VALIDATION-V1}
DEST_HOST=${DEST_HOST:-connect.nmb2.seetacloud.com}
DEST_PORT=${DEST_PORT:-19864}
DEST_KEY=${DEST_KEY:-/root/.ssh/hier_handoff_ed25519}
DEST_OUT=${DEST_OUT:-/root/autodl-tmp/results/P40-HIER-S256V128-AIRCRAFT-D4-FULL-3090-V1}
REMOTE_SOURCE=${REMOTE_SOURCE:-${SOURCE}}
LOCAL_STATUS=${SOURCE}.handoff_status.txt

failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${LOCAL_STATUS}"; exit "${code}"; }
trap failed ERR INT TERM
printf 'waiting_for_sanitized_candidate\n' >"${LOCAL_STATUS}"
while [[ ! -f "${SOURCE}/candidate_runtime_config.json" || \
         ! -f "${SOURCE}/hier_full_sanitized.pt" ]]; do
  current=$(cat "${SOURCE}/status.txt" 2>/dev/null || true)
  [[ "${current}" != failed_* ]] || exit 5
  sleep 10
done

ssh_cmd=(ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
  -i "${DEST_KEY}" -p "${DEST_PORT}")
rsync_cmd=(rsync -a -e "ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -i ${DEST_KEY} -p ${DEST_PORT}")

printf 'transferring_candidate_to_3090\n' >"${LOCAL_STATUS}"
"${ssh_cmd[@]}" "root@${DEST_HOST}" "mkdir -p '${REMOTE_SOURCE}'"
"${rsync_cmd[@]}" "${SOURCE}/hier_full_sanitized.pt" \
  "root@${DEST_HOST}:${REMOTE_SOURCE}/hier_full_sanitized.pt"
"${rsync_cmd[@]}" "${SOURCE}/candidate_runtime_config.json" \
  "root@${DEST_HOST}:${REMOTE_SOURCE}/candidate_runtime_config.json"

printf 'starting_3090_runtime_diagnostic\n' >"${LOCAL_STATUS}"
"${ssh_cmd[@]}" "root@${DEST_HOST}" bash -s -- "${REMOTE_SOURCE}" "${DEST_OUT}" <<'REMOTE'
set -Eeuo pipefail
SOURCE=$1
OUT=$2
PROJECT=/root/autodl-tmp/xh-apex-v1
PY=/workspace/venvs/p06-cu121/bin/python
if screen -list | grep -q 'hier-combo-3090'; then
  echo '3090 diagnostic already running' >&2
  exit 4
fi
if [[ -e "${OUT}" ]]; then
  echo "refusing non-fresh output: ${OUT}" >&2
  exit 5
fi
mkdir -p "${OUT}/logs"
cat >"${OUT}/run.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src"
failed() { code=\$?; printf 'failed_exit_%s\\n' "\${code}" >"${OUT}/status.txt"; exit "\${code}"; }
trap failed ERR INT TERM
for condition in hard sentinel; do
  if [[ "\${condition}" == hard ]]; then root=/root/autodl-tmp/pseudo10k-trial-mix-local; else root=/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1; fi
  printf 'running_%s_exact_runtime\\n' "\${condition}" >"${OUT}/status.txt"
  mkdir -p "${OUT}/\${condition}"
  "${PY}" -u scripts/run_competition_runtime_coco.py \\
    --config "${SOURCE}/candidate_runtime_config.json" --pseudo-root "\${root}" \\
    --device cuda:0 --predictions "${OUT}/\${condition}/predictions.json" \\
    --summary "${OUT}/\${condition}/runtime_summary.json" \\
    >"${OUT}/logs/\${condition}.log" 2>&1
  "${PY}" scripts/evaluate_fixed_score_threshold.py \\
    --gt "\${root}/ground_truth.json" --pred "${OUT}/\${condition}/predictions.json" \\
    --threshold 0 --output "${OUT}/\${condition}/metrics.json"
done
trap - ERR INT TERM
printf 'complete\\n' >"${OUT}/status.txt"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
EOF
chmod +x "${OUT}/run.sh"
screen -dmS hier-combo-3090 bash "${OUT}/run.sh"
REMOTE

trap - ERR INT TERM
printf 'complete_3090_diagnostic_started\n' >"${LOCAL_STATUS}"
