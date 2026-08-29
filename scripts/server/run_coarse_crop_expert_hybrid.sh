#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${GT:?set GT}"
: "${REFERENCE:?set REFERENCE}"
: "${NATURAL:?set NATURAL}"
: "${BALANCED:?set BALANCED}"
: "${WAIT_STATUS:?set WAIT_STATUS}"
: "${OUT:?set OUT}"

mkdir -p "${OUT}"
while [[ ! -f "${WAIT_STATUS}" ]] || [[ $(<"${WAIT_STATUS}") != complete ]]; do
  printf 'waiting_for_natural_evidence\n' >"${OUT}/status.txt"
  sleep 20
done

cd "${REPO}"
printf 'routing_evidence\n' >"${OUT}/status.txt"
"${PYTHON_BIN}" scripts/merge_coarse_crop_verifier_experts.py \
  --reference "${REFERENCE}" \
  --expert natural "${NATURAL}" \
  --expert balanced "${BALANCED}" \
  --ship natural \
  --aircraft natural \
  --vehicle balanced \
  --output "${OUT}/hybrid_crop_predictions.json" \
  --summary "${OUT}/hybrid_crop_summary.json" \
  >"${OUT}/hybrid_crop.log" 2>&1

"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${GT}" \
  --pred "${OUT}/hybrid_crop_predictions.json" \
  --output "${OUT}/hybrid_direct_frontier.json" \
  >"${OUT}/hybrid_direct_frontier.log" 2>&1

printf 'pixel_oer\n' >"${OUT}/status.txt"
"${PYTHON_BIN}" scripts/train_pseudo_pixel_oer.py \
  --ground-truth "${GT}" \
  --crop-predictions "${OUT}/hybrid_crop_predictions.json" \
  --output-dir "${OUT}/pixel_oer" \
  --nms-iou 0.60 \
  >"${OUT}/pixel_oer.log" 2>&1

for variant in identity dual_hypothesis; do
  "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "${GT}" \
    --pred "${OUT}/pixel_oer/${variant}_predictions.json" \
    --output "${OUT}/pixel_oer/${variant}_frontier.json" \
    >"${OUT}/pixel_oer/${variant}_frontier.log" 2>&1
done

(
  cd "${OUT}"
  find . -type f ! -name SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum >SHA256SUMS.txt
)
printf 'complete\n' >"${OUT}/status.txt"
