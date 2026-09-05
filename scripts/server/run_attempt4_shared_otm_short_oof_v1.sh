#!/usr/bin/env bash
set -Eeuo pipefail

# Re-evaluate the only positive Sprint20 attack candidate through the exact
# shared-forward path used by deployment. This fresh scientific contract does
# not erase or reinterpret the historical native-vs-shared parity failure.

PROJECT=${PROJECT:-/root/autodl-tmp/xh-sprint20-ab51106}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
INPUT=${INPUT:-/root/autodl-tmp/results/HERA-SPRINT20-20260905/p40_short_oof}
IMAGE_ROOT=${IMAGE_ROOT:-/root/autodl-tmp/data}
COCO=${COCO:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1/aggregate/ground_truth.json}
GROUP_MANIFEST=${GROUP_MANIFEST:-/root/autodl-tmp/capscale-assets/split_view.json}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-SHARED-OTM-SHORT-OOF-V1}
STATUS=${OUT}/status.txt

COCO_SHA=c4290b542ffdafe62d5dbcb575f0b3431d46721bbcb366f8ef05291653fcb975
GROUPS_SHA=a647ce030fa832aadc6a6c286a3f6464ac1783f71797a52cc598ec340f128943
CONFIG_SHAS=(
  142108de4a19b4389aa45aea256133c17eec8a6ea643f8f01e0fb121f905463e
  cc8a09403afffdb9cb2d87a2e7e4b142bcae1a2abb68699abfce9b6e98eba535
  92605162a55dacd5f7063cf4a83f9a2e5d369bfa9df08bb33d18cad7d1cb3f5d
)
GT_SHAS=(
  d1021e813a407738dfd2403b6338f8e2a455dfe44241121257de52f6c2ecd87b
  027a707a7264a325d38a187e62cf8ff4917e5c8112ab9804a1fce24c743bc89e
  b238ce65ef1a7a7d90f40cb8879b4fdca032519057791dda8aba8d821aaaf0a3
)
WEIGHT_SHAS=(
  48434e5206058ed767abea1b2f1fbd5252daf131e09b095a68678c6e39ead5c1
  1d65e9fde45e5f7d2c7909ef02baa81628f117453ed1d713cb9b96a433f97281
  09787a83a6564dbcb084a2d90ca9f7468c19528d7d19203588c8e528f4f8decb
)

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}"
exec 9>"${OUT}.lock"
flock -n 9 || exit 4
failed() {
  code=$?
  printf 'failed_exit_%s\n' "${code}" >"${STATUS}"
  exit "${code}"
}
trap failed ERR INT TERM

sha_is() {
  local path=$1 expected=$2
  test -f "${path}"
  test "$(sha256sum "${path}" | awk '{print $1}')" = "${expected}"
}

printf 'preflight\n' >"${STATUS}"
test -d "${PROJECT}"
test -d "${IMAGE_ROOT}"
sha_is "${COCO}" "${COCO_SHA}"
sha_is "${GROUP_MANIFEST}" "${GROUPS_SHA}"
for fold in 0 1 2; do
  sha_is "${INPUT}/fold_${fold}/config.json" "${CONFIG_SHAS[$fold]}"
  sha_is "${INPUT}/fold_${fold}/ground_truth.json" "${GT_SHAS[$fold]}"
  weight=$(${PY} -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["weight_path"])' "${INPUT}/fold_${fold}/config.json")
  sha_is "${weight}" "${WEIGHT_SHAS[$fold]}"
done

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
for fold in 0 1 2; do
  printf 'probe_fold_%s\n' "${fold}" >"${STATUS}"
  mkdir -p "${OUT}/fold_${fold}"
  "${PY}" -m sprint20.cli probe \
    --config "${INPUT}/fold_${fold}/config.json" \
    --coco "${INPUT}/fold_${fold}/ground_truth.json" \
    --image-root "${IMAGE_ROOT}" \
    --out "${OUT}/fold_${fold}/shared.json" \
    --head shared --role outer_oof_short \
    >"${OUT}/fold_${fold}/shared.log" 2>&1
done

printf 'aggregate\n' >"${STATUS}"
"${PY}" scripts/aggregate_sprint20_head_caches.py \
  --source shared \
  --input-dir "${OUT}" \
  --aggregate-coco "${COCO}" \
  --output-oto "${OUT}/shared_oto.json" \
  --output-otm "${OUT}/shared_otm.json" \
  >"${OUT}/aggregate.log" 2>&1

printf 'analyze\n' >"${STATUS}"
"${PY}" scripts/analyze_sprint20_oof_routing.py \
  --coco "${COCO}" \
  --oto-cache "${OUT}/shared_oto.json" \
  --otm-cache "${OUT}/shared_otm.json" \
  --output "${OUT}/crossfit_routing_shared_v1.json" \
  --step 0.001 \
  --primary-threshold 0.536 \
  --groups "${GROUP_MANIFEST}" \
  --bootstrap 3000 \
  >"${OUT}/analysis.log" 2>&1

find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
printf 'complete\n' >"${STATUS}"
