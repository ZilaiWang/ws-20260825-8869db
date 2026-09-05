#!/usr/bin/env bash
set -Eeuo pipefail

# Engineering-only full-image soak for the selected P40 + hierarchy Vehicle
# + Aircraft-D4 workpoint.  It verifies coverage, finite outputs and ordinary
# image latency on all official training images; it is not a score estimate.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
CONFIG=${CONFIG:-/root/autodl-tmp/results/P40-HIER-S256V128-AIRCRAFT-D4-VEHICLE-THRESHOLD-LOW-V1/0p420/runtime_config.json}
IMAGE_ROOT=${IMAGE_ROOT:-/root/autodl-tmp/data/images/train}
OUT=${OUT:-/root/autodl-tmp/results/P40-HIER-S256V128-T042-AIRCRAFT-D4-FULL-SOAK-V1}
STATUS=${OUT}/status.txt

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}/logs"
exec 9>"${OUT}.lock"
flock -n 9 || exit 4
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${STATUS}"; exit "${code}"; }
trap failed ERR INT TERM
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
test -f "${CONFIG}"
test -d "${IMAGE_ROOT}"

printf 'running_three_shard_exact_runtime_soak\n' >"${STATUS}"
for index in 0 1 2; do
  CUDA_VISIBLE_DEVICES="${index}" "${PY}" -u scripts/run_competition_runtime_soak.py \
    --config "${CONFIG}" --image-root "${IMAGE_ROOT}" --device cuda:0 \
    --shard-index "${index}" --shard-count 3 --output "${OUT}/shard_${index}" \
    >"${OUT}/logs/shard_${index}.log" 2>&1 &
  pids[$index]=$!
done
wait "${pids[@]}"

printf 'summarizing\n' >"${STATUS}"
"${PY}" - "${OUT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = [json.loads((out / f"shard_{i}" / "summary.json").read_text()) for i in range(3)]
if any(row.get("status") != "pass" for row in rows):
    raise SystemExit("one or more soak shards failed")
count = sum(int(row["image_count"]) for row in rows)
if count != 4481:
    raise SystemExit(f"expected 4481 unique assigned images, got {count}")
objects = sum(int(row["object_count"]) for row in rows)
weighted_mean = sum(
    float(row["mean_image_seconds"]) * int(row["image_count"]) for row in rows
) / count
digest = hashlib.sha256()
for row in rows:
    digest.update(str(row["prediction_digest_sha256"]).encode())
payload = {
    "status": "pass",
    "role": "engineering_runtime_soak_not_score_estimation",
    "image_count": count,
    "object_count": objects,
    "weighted_mean_image_seconds": weighted_mean,
    "max_shard_p95_image_seconds": max(float(row["p95_image_seconds"]) for row in rows),
    "combined_shard_digest_sha256": digest.hexdigest(),
    "shards": rows,
}
(out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
