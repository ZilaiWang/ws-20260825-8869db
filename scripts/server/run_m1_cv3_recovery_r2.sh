#!/usr/bin/env bash
set -Eeuo pipefail

umask 022

MAIN=/workspace/xh-202625
MODEL=/workspace/xh-202625-model
VENV=/workspace/venvs/cv3-model-cu121
RESULTS=/workspace/results
RUN_ROOT="$RESULTS/M1-CV3-OOF"
FOLD_DIR="$RUN_ROOT/fold_2"
AGG_ROOT="$RESULTS/M1-CV3-OOF-aggregate"
OPS_ROOT="$RESULTS/M1-CV3-OOF-ops"
INTERRUPTED_ROOT="$RESULTS/M1-CV3-OOF-INTERRUPTED-20260725-FOLD2-E134"
RETURN_ARCHIVE="$RESULTS/M1-CV3-OOF-return-no-checkpoints.tar.gz"
PROTOCOL_SOURCE="$MAIN/docs/server/M1_CV3_OOF_RECOVERY_R2_POWER_INTERRUPTION.md"

PRETRAINED=/workspace/cv3-model-assets/yolo26s.pt
PRETRAINED_SHA=646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
ORIGINAL_ASSET_LOCK=/workspace/cv3-model-assets/MODEL_ASSET_ENV_LOCK.json
RECOVERY_ASSET_LOCK="$OPS_ROOT/RECOVERY_R2_MODEL_ASSET_ENV_LOCK.json"
ASSET_SPEC="$MAIN/configs/experiments/cv3_model_asset_env.json"
DATA_LOCK=/workspace/formal-detection-data/FORMAL_DETECTION_DATA_LOCK.json
DATA_LOCK_SHA=03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a
FORMAL_CROP="$RESULTS/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv"
FORMAL_CROP_SHA=a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128
P02_REGISTER="$RESULTS/CV3-DETECTION-DATA-LOCK-TASK-00/p02_path.txt"
P02_SHA=f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e

export PATH="$VENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ULTRALYTICS_OFFLINE=true

mkdir -p "$OPS_ROOT"
exec 9>"$OPS_ROOT/recovery_r2_execution.lock"
flock -n 9 || {
  printf '%s\n' "Another M1 CV3 recovery is already active." >&2
  exit 2
}

RECOVERY_LOG="$OPS_ROOT/recovery-r2-$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$RECOVERY_LOG") 2>&1

on_error() {
  local exit_code=$?
  {
    printf 'status=failed\n'
    printf 'exit_code=%s\n' "$exit_code"
    printf 'failed_at=%s\n' "$(date -Is)"
    printf 'recovery_log=%s\n' "$RECOVERY_LOG"
  } > "$OPS_ROOT/recovery_r2_status.txt"
  exit "$exit_code"
}
trap on_error ERR

require_sha() {
  local expected=$1
  local path=$2
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

record_environment() {
  {
    printf 'captured_at=%s\n' "$(date -Is)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'main_repo_commit='
    git -C "$MAIN" rev-parse HEAD 2>/dev/null || printf 'unavailable\n'
    printf 'model_repo_commit=unavailable_empty_git_metadata\n'
    printf 'formal_code_lock_sha256=%s\n' \
      "$(sha256sum "$MAIN/docs/server/CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt" \
        | awk '{print $1}')"
    printf 'model_code_lock_sha256=%s\n' \
      "$(sha256sum "$MAIN/docs/server/XH_MODEL_INTEGRATION_CODE_SHA256.txt" \
        | awk '{print $1}')"
    printf 'original_asset_lock_sha256=%s\n' \
      "$(sha256sum "$ORIGINAL_ASSET_LOCK" | awk '{print $1}')"
    printf 'recovery_asset_lock_sha256=%s\n' \
      "$(sha256sum "$RECOVERY_ASSET_LOCK" | awk '{print $1}')"
    printf 'data_lock_sha256=%s\n' \
      "$(sha256sum "$DATA_LOCK" | awk '{print $1}')"
    python --version
    python -m pip freeze
    nvidia-smi
    df -h /workspace
  } > "$FOLD_DIR/environment.txt"
}

printf 'M1 CV3 recovery R2 started at %s\n' "$(date -Is)"
test -d "$RUN_ROOT"
test -d "$FOLD_DIR"
test -s "$PROTOCOL_SOURCE"
test ! -e "$INTERRUPTED_ROOT"
test ! -e "$AGG_ROOT"
test ! -e "$RETURN_ARCHIVE"
test ! -e "$RETURN_ARCHIVE.sha256"

require_sha "$PRETRAINED_SHA" "$PRETRAINED"
require_sha "$DATA_LOCK_SHA" "$DATA_LOCK"
require_sha "$FORMAL_CROP_SHA" "$FORMAL_CROP"
test "$(stat -c '%s' "$PRETRAINED")" = 20422725

cd "$MAIN"
sha256sum -c docs/server/CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt
sha256sum -c docs/server/XH_MODEL_INTEGRATION_CODE_SHA256.txt
if test ! -e "$RECOVERY_ASSET_LOCK"; then
  PYTHONPATH=src python scripts/lock_cv3_model_assets.py create \
    --config "$ASSET_SPEC" \
    --asset-root /workspace/cv3-model-assets \
    --expected-gpu "NVIDIA GeForce RTX 4080 SUPER" \
    --output "$RECOVERY_ASSET_LOCK"
fi
PYTHONPATH=src python scripts/lock_cv3_model_assets.py verify \
  --config "$ASSET_SPEC" \
  --asset-root /workspace/cv3-model-assets \
  --expected-gpu "NVIDIA GeForce RTX 4080 SUPER" \
  --lock "$RECOVERY_ASSET_LOCK" \
  --report "$OPS_ROOT/recovery_r2_asset_env_verification.json"

python - "$ORIGINAL_ASSET_LOCK" "$RECOVERY_ASSET_LOCK" <<'PY'
import copy
import json
import sys
from pathlib import Path

original = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
recovery = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
original_uuid = original["environment"]["nvidia_smi"]["uuid"]
recovery_uuid = recovery["environment"]["nvidia_smi"]["uuid"]
assert original_uuid != recovery_uuid, (original_uuid, recovery_uuid)

for payload in (original, recovery):
    payload.pop("lock_fingerprint")
original["environment"]["nvidia_smi"]["uuid"] = recovery_uuid
assert original == recovery, "克隆环境除 GPU UUID 外仍有其他漂移"

Path(
    "/workspace/results/M1-CV3-OOF-ops/recovery_r2_environment_delta.json"
).write_text(
    json.dumps(
        {
            "status": "accepted_clone_gpu_uuid_only",
            "original_gpu_uuid": original_uuid,
            "recovery_gpu_uuid": recovery_uuid,
            "all_other_locked_fields_equal": True,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

PYTHONPATH=src python -m pytest -q tests/test_cv3_oof.py
PYTHONPATH=src python -m ruff check \
  src/rsdet/experiments/cv3_oof.py \
  scripts/materialize_cv3_oof_config.py \
  scripts/finalize_cv3_oof_fold.py scripts/audit_cv3_oof.py

cd "$MODEL"
PYTHONPATH=src python -m pytest -q \
  tests/test_trainer_contract.py \
  tests/test_infer_evaluation.py \
  tests/test_ultralytics_adapter.py \
  tests/test_inference_pipeline.py \
  tests/test_tile_fusion.py
PYTHONPATH=src python -m ruff check \
  scripts/train.py scripts/infer.py \
  tests/test_trainer_contract.py tests/test_infer_evaluation.py

python - "$RUN_ROOT" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
expected_rows = {0: 160, 1: 160, 2: 134}
for fold, expected in expected_rows.items():
    csv_path = root / f"fold_{fold}" / "runs/foundation/results.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == expected, (fold, len(rows), expected)
    assert int(float(rows[-1]["epoch"])) == expected, (fold, rows[-1]["epoch"])

for fold in (0, 1):
    metadata = json.loads(
        (root / f"fold_{fold}" / "fold_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "fold_delivery_complete", (fold, metadata)
    assert (root / f"fold_{fold}" / "predictions_low.json").stat().st_size > 0

fold2 = root / "fold_2"
assert not (fold2 / "fold_metadata.json").exists()
assert not (fold2 / "predictions_low.json").exists()
checkpoint = torch.load(
    fold2 / "runs/foundation/weights/last.pt",
    map_location="cpu",
    weights_only=False,
)
assert checkpoint["epoch"] == 133, checkpoint["epoch"]
assert checkpoint["optimizer"] is not None
assert checkpoint["ema"] is not None
assert checkpoint["train_args"].get("resume") is False

log = (fold2 / "train.log").read_text(encoding="utf-8", errors="replace")
plain = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", log.replace("\r", "\n"))
assert "Traceback" not in plain
assert "CUDA out of memory" not in plain
assert "135/160" in plain
assert "77/260" in plain
PY

FROZEN_BEFORE="$OPS_ROOT/recovery_r2_fold01_before.sha256"
: > "$FROZEN_BEFORE"
for fold in 0 1; do
  fold_dir="$RUN_ROOT/fold_$fold"
  for rel in \
    split_view.json \
    resolved_config.yaml \
    train_summary.json \
    runs/foundation/results.csv \
    runs/foundation/weights/last.pt \
    resolved_infer.yaml \
    predictions_low.json \
    predictions_low.runtime.json \
    fold_metadata.json \
    input-gates/detection_data_lock_verification.json
  do
    test -s "$fold_dir/$rel"
    sha256sum "$fold_dir/$rel" >> "$FROZEN_BEFORE"
  done
done

mkdir "$INTERRUPTED_ROOT"
cp -a "$FOLD_DIR" "$INTERRUPTED_ROOT/fold_2"
ARCHIVE_INDEX_TMP="$OPS_ROOT/recovery_r2_interrupted_archive.sha256.tmp"
find "$INTERRUPTED_ROOT/fold_2" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$ARCHIVE_INDEX_TMP"
mv "$ARCHIVE_INDEX_TMP" "$INTERRUPTED_ROOT/ARCHIVE_SHA256.txt"
cp "$PROTOCOL_SOURCE" "$RUN_ROOT/recovery_r2_protocol.md"

python - "$RUN_ROOT" "$INTERRUPTED_ROOT" "$RECOVERY_LOG" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_root = Path(sys.argv[1])
archive_root = Path(sys.argv[2])
recovery_log = Path(sys.argv[3])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "status": "recovery_prepared",
    "scientific_status": "formal_with_power_interruption_resume_amendment",
    "prepared_at": datetime.now(timezone.utc).isoformat(),
    "failure_type": "external_power_interruption",
    "completed_folds_reused": [0, 1],
    "resumed_fold": 2,
    "interrupted_completed_epochs": 134,
    "interrupted_partial_epoch": 135,
    "interrupted_partial_batch": "77/260",
    "resume_used": True,
    "resume_checkpoint": str(
        run_root / "fold_2/runs/foundation/weights/last.pt"
    ),
    "resume_checkpoint_sha256": sha256(
        run_root / "fold_2/runs/foundation/weights/last.pt"
    ),
    "original_initialization_sha256": (
        "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
    ),
    "interrupted_archive": str(archive_root),
    "interrupted_archive_index_sha256": sha256(
        archive_root / "ARCHIVE_SHA256.txt"
    ),
    "recovery_log": str(recovery_log),
}
(run_root / "recovery_r2_protocol.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

{
  printf 'status=resuming_fold2_from_epoch134\n'
  printf 'started_at=%s\n' "$(date -Is)"
  printf 'resume_used=true\n'
  printf 'interrupted_archive=%s\n' "$INTERRUPTED_ROOT"
  printf 'recovery_log=%s\n' "$RECOVERY_LOG"
} > "$OPS_ROOT/recovery_r2_status.txt"

P02="$(cat "$P02_REGISTER")"
require_sha "$P02_SHA" "$P02"
cd "$MAIN"
PYTHONPATH=src python scripts/lock_formal_detection_data.py verify \
  --config configs/experiments/formal_detection_data_lock.json \
  --data-root /workspace/data \
  --cv3-manifest data/splits/cv3_airport_proxy_k60_v2.json \
  --p02-manifest "$P02" \
  --formal-crop-manifest "$FORMAL_CROP" \
  --lock "$DATA_LOCK" \
  --expected-lock-sha256 "$DATA_LOCK_SHA" \
  --report "$FOLD_DIR/input-gates/detection_data_lock_verification.json" \
  2>&1 | tee "$FOLD_DIR/detection-data-lock-resume-verify.log"

RESUME_CHECKPOINT="$FOLD_DIR/runs/foundation/weights/last.pt"
RESUME_CHECKPOINT_SHA="$(sha256sum "$RESUME_CHECKPOINT" | awk '{print $1}')"
cd "$MODEL"
PYTHONPATH=src python scripts/train.py \
  --config "$FOLD_DIR/train_config.yaml" \
  --resume "$RESUME_CHECKPOINT" \
  2>&1 | tee "$FOLD_DIR/train_resume.log"

cp "$FOLD_DIR/train_summary.json" "$FOLD_DIR/train_summary.resume_segment.json"
python - "$FOLD_DIR" "$PRETRAINED" "$RESUME_CHECKPOINT_SHA" <<'PY'
import json
import sys
from pathlib import Path

fold_dir = Path(sys.argv[1])
pretrained = str(Path(sys.argv[2]).resolve())
resume_checkpoint_sha = sys.argv[3]
summary_path = fold_dir / "train_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
stage = summary["stages"][0]
assert summary["dry_run"] is False
assert stage["arguments"].get("resume") is True
assert Path(summary["initial_weights"]).resolve() == (
    fold_dir / "runs/foundation/weights/last.pt"
).resolve()

summary["initial_weights"] = pretrained
stage["input_weights"] = pretrained
stage["arguments"].pop("resume", None)
summary["recovery_amendment"] = {
    "type": "external_power_interruption_resume",
    "resume_used": True,
    "resumed_fold": 2,
    "completed_epochs_before_interruption": 134,
    "resume_checkpoint_sha256": resume_checkpoint_sha,
    "raw_resume_segment_summary": str(
        fold_dir / "train_summary.resume_segment.json"
    ),
    "interrupted_archive": (
        "/workspace/results/"
        "M1-CV3-OOF-INTERRUPTED-20260725-FOLD2-E134/fold_2"
    ),
}
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

python - "$FOLD_DIR/runs/foundation/results.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 160, len(rows)
assert int(float(rows[-1]["epoch"])) == 160, rows[-1]["epoch"]
PY

cd "$MAIN"
PYTHONPATH=src python scripts/materialize_cv3_oof_config.py \
  --template configs/experiments/m1_yolo26s_1024_cv3_oof_infer.template.yaml \
  --output "$FOLD_DIR/resolved_infer.yaml" \
  --fold 2 \
  --data-root /workspace/data \
  --split-view "$FOLD_DIR/split_view.json" \
  --fold-output-dir "$FOLD_DIR" \
  --checkpoint "$FOLD_DIR/runs/foundation/weights/last.pt"

cd "$MODEL"
PYTHONPATH=src python scripts/infer.py \
  --config "$FOLD_DIR/resolved_infer.yaml" \
  2>&1 | tee "$FOLD_DIR/infer.log"

record_environment

cd "$MAIN"
PYTHONPATH=src python scripts/finalize_cv3_oof_fold.py \
  --plan "$RUN_ROOT/oof_run_plan.json" \
  --fold 2 \
  --train-config "$FOLD_DIR/resolved_config.yaml" \
  --train-summary "$FOLD_DIR/train_summary.json" \
  --infer-config "$FOLD_DIR/resolved_infer.yaml" \
  --environment "$FOLD_DIR/environment.txt" \
  --checkpoint "$FOLD_DIR/runs/foundation/weights/last.pt" \
  --predictions "$FOLD_DIR/predictions_low.json" \
  --runtime "$FOLD_DIR/predictions_low.runtime.json" \
  --data-lock-verification \
    "$FOLD_DIR/input-gates/detection_data_lock_verification.json" \
  --output "$FOLD_DIR/fold_metadata.json"

python - "$FOLD_DIR/fold_metadata.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metadata = json.loads(path.read_text(encoding="utf-8"))
metadata["recovery_amendment"] = {
    "type": "external_power_interruption_resume",
    "resume_used": True,
    "resumed_from_completed_epoch": 134,
    "evidence": (
        "/workspace/results/M1-CV3-OOF/recovery_r2_protocol.json"
    ),
}
path.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

sha256sum -c "$FROZEN_BEFORE"

PYTHONPATH=src python scripts/audit_cv3_oof.py \
  --manifest data/splits/cv3_airport_proxy_k60_v2.json \
  --manifest-sha256 27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331 \
  --plan "$RUN_ROOT/oof_run_plan.json" \
  --run-root "$RUN_ROOT" \
  --output-dir "$AGG_ROOT" \
  --formal-crop-manifest "$FORMAL_CROP"

python - "$RUN_ROOT" "$AGG_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
aggregate_root = Path(sys.argv[2])
for fold in range(3):
    fold_dir = run_root / f"fold_{fold}"
    with (fold_dir / "runs/foundation/results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 160, (fold, len(rows))
    metadata = json.loads(
        (fold_dir / "fold_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "fold_delivery_complete", (fold, metadata)

oof_path = aggregate_root / "oof_metadata.json"
oof = json.loads(oof_path.read_text(encoding="utf-8"))
assert oof["status"] == "complete_downstream_ready", oof
assert oof["image_count"] == 4481, oof
oof["recovery_amendment"] = {
    "scientific_status": "formal_with_power_interruption_resume_amendment",
    "resume_used": True,
    "resumed_fold": 2,
    "resumed_from_completed_epoch": 134,
    "evidence": str(run_root / "recovery_r2_protocol.json"),
}
oof["audit"]["resume_used"] = True
oof_path.write_text(
    json.dumps(oof, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

protocol_path = run_root / "recovery_r2_protocol.json"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
protocol["status"] = "complete"
protocol["aggregate_status"] = oof["status"]
protocol["aggregate_image_count"] = oof["image_count"]
protocol_path.write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

tar --exclude='*.pt' --exclude='prepared_data' -czf "$RETURN_ARCHIVE" \
  -C "$RESULTS" M1-CV3-OOF M1-CV3-OOF-aggregate
sha256sum "$RETURN_ARCHIVE" > "$RETURN_ARCHIVE.sha256"

{
  printf 'status=complete\n'
  printf 'completed_at=%s\n' "$(date -Is)"
  printf 'resume_used=true\n'
  printf 'resumed_fold=2\n'
  printf 'resumed_from_completed_epoch=134\n'
  printf 'interrupted_archive=%s\n' "$INTERRUPTED_ROOT"
  printf 'aggregate=%s\n' "$AGG_ROOT/oof_metadata.json"
  printf 'return_archive=%s\n' "$RETURN_ARCHIVE"
  printf 'recovery_log=%s\n' "$RECOVERY_LOG"
} > "$OPS_ROOT/recovery_r2_status.txt"

printf 'M1 CV3 recovery R2 complete at %s\n' "$(date -Is)"
