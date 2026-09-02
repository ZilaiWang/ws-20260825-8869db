#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
ROOT=${ROOT:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for key in s1024 s1280 m1024 m1280; do
  test "$(cat "${ROOT}/${key}/status.txt")" = complete
  test -f "${ROOT}/${key}/frontier.json"
done
cd "${PROJECT}"
"${PY}" scripts/summarize_yolo_capacity_scale_screen.py \
  --s1024 "${ROOT}/s1024/frontier.json" \
  --s1280 "${ROOT}/s1280/frontier.json" \
  --m1024 "${ROOT}/m1024/frontier.json" \
  --m1280 "${ROOT}/m1280/frontier.json" \
  --output "${ROOT}/screening_result.json"
sha256sum "${ROOT}"/*/frontier.json "${ROOT}/screening_result.json" \
  >"${ROOT}/RESULT_SHA256.txt"
printf 'complete\n' >"${ROOT}/status.txt"
