#!/usr/bin/env bash
set -Eeuo pipefail

# Persistent single-GPU queue.  It starts only after the already queued plan-15
# short-chain has reached a verified terminal status.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
SHORT_FIXED=${SHORT_FIXED:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE-FIXED-BENCHMARKS-V1}
PROGRESSIVE40=${PROGRESSIVE40:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1}
PROGRESSIVE40_FIXED=${PROGRESSIVE40_FIXED:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-FIXED-BENCHMARKS-V1}
OUT=${OUT:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-LONG-QUEUE-V1}
STATUS=${OUT}/status.txt

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}"
exec 9>"${OUT}.lock"
flock -n 9 || exit 4
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${STATUS}"; exit "${code}"; }
trap failed ERR INT TERM

printf 'waiting_for_progressive20_fixed\n' >"${STATUS}"
while true; do
  upstream=$(cat "${SHORT_FIXED}/status.txt" 2>/dev/null || true)
  if [[ "${upstream}" = complete ]]; then break; fi
  if [[ "${upstream}" = failed* ]]; then
    printf 'blocked_by_progressive20_fixed_%s\n' "${upstream}" >"${STATUS}"
    exit 11
  fi
  sleep 30
done

cd "${PROJECT}"
printf 'progressive40_cv3\n' >"${STATUS}"
bash scripts/server/run_scaleroute_plan15_progressive40_cv3.sh

printf 'progressive40_fixed_benchmarks\n' >"${STATUS}"
PROGRESSIVE="${PROGRESSIVE40}" OUT="${PROGRESSIVE40_FIXED}" \
  bash scripts/server/run_scaleroute_plan15_progressive_fixed_benchmarks.sh

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
