#!/usr/bin/env bash
# Fresh paired inference only. Evaluation decides whether to run Sentinel.
set -Eeuo pipefail
PROJECT=${PROJECT:-/root/autodl-tmp/xh-p40-zoom-v1}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
P40=${P40:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1}
OUT=${OUT:-/root/autodl-tmp/results/P40-VEHICLE-ZOOM-RESCUE-V1}
CONFIG=${CONFIG:-configs/experiments/p40_vehicle_zoom_rescue_v1.json}
CONDITION=${1:-hard}
case "$CONDITION" in
  hard) ROOT=/root/autodl-tmp/pseudo10k-trial-mix-local ;;
  sentinel) ROOT=/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1 ;;
  *) exit 2 ;;
esac
mkdir -p "$OUT"
exec 9>"$OUT.lock"
flock -n 9 || exit 4
test ! -e "$OUT/$CONDITION"
mkdir "$OUT/$CONDITION"
STATUS="$OUT/$CONDITION/status.txt"
trap 'code=$?; printf "failed_exit_%s\n" "$code" >"$STATUS"; exit "$code"' ERR INT TERM
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src:$PROJECT"
export OMP_NUM_THREADS=4
WEIGHTS=()
for fold in 0 1 2; do
  WEIGHTS+=("$P40/fold_$fold/adaptation/runs/resolution_adaptation/weights/last.pt")
done
"$PY" - "$P40" "$ROOT" "$OUT/$CONDITION" "$CONDITION" "$CONFIG" <<'PY'
import hashlib, json, pathlib, sys
root, data, out = map(pathlib.Path, sys.argv[1:4])
condition = sys.argv[4]
config_path = pathlib.Path(sys.argv[5])
contract = json.loads(config_path.read_text())
expected = {root/'aggregate/crossfit_frontier.json': contract['frontier_sha256'],
            data/'ground_truth.json': contract['ground_truth_sha256'][condition]}
for fold, sha in enumerate(contract['weights_sha256']):
    expected[root/f'fold_{fold}/adaptation/runs/resolution_adaptation/weights/last.pt'] = sha
observed = {}
for path, sha in expected.items():
    actual = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() if sys.version_info >= (3, 11) else hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != sha:
        raise ValueError(f'asset SHA mismatch: {path}: {actual}')
    observed[str(path)] = actual
frontier = json.loads((root/'aggregate/crossfit_frontier.json').read_text())
thresholds = frontier['frontiers'][contract['fdr_level']]['crossfit_thresholds']
if set(thresholds) != {'0','1','2'}:
    raise ValueError('incomplete threshold source')
code = {}
for directory in ('src', 'scripts', 'configs'):
    for path in sorted(pathlib.Path(directory).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            code[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
(out/'preflight.json').write_text(json.dumps({'contract': contract, 'contract_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(), 'assets': observed, 'thresholds': thresholds, 'code_sha256': code}, indent=2)+'\n')
PY
AUX_VIEW=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("auxiliary_view", "zoom"))' "$CONFIG")
for view in base "$AUX_VIEW"; do
  read -r TILE OVERLAP ROTATION < <("$PY" -c 'import json,sys; c=json.load(open(sys.argv[1]))[sys.argv[2]]; print(c["tile_size"], c["overlap"], c.get("tile_rotation",0))' "$CONFIG" "$view")
  printf 'running_%s_%s\n' "$CONDITION" "$view" >"$STATUS"
  "$PY" scripts/run_multifamily_cv3_pseudo_eval.py \
    --pseudo-root "$ROOT" --family yolo --weights "${WEIGHTS[@]}" \
    --output-dir "$OUT/$CONDITION/$view" --score-floor 0.001 \
    --batch-size 4 --device cuda:0 --imgsz 1280 --tile-size "$TILE" --overlap "$OVERLAP" --tile-rotation "$ROTATION" \
    >"$OUT/$CONDITION/$view.log" 2>&1
done
printf 'inference_complete_waiting_analysis\n' >"$STATUS"
trap - ERR INT TERM
