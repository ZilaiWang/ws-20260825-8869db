#!/usr/bin/env bash
# User-operated push only. Never stores credentials and never clicks submit.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE="xh-detector:p40-hier-d4-rescue-final"
DELIVERY="$ROOT/dist/p40-hier-d4-rescue-final"
REGISTRY="competition-registry.cn-beijing.cr.aliyuncs.com/competition/team612528"

if [[ ! -f "$DELIVERY/IMAGE_MANIFEST.json" ]]; then
  echo "ERROR: final image manifest is missing; build the frozen delivery first." >&2
  exit 1
fi
EXPECTED_ID=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' "$DELIVERY/IMAGE_MANIFEST.json")
EXPECTED_CANDIDATE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate_id"])' "$DELIVERY/IMAGE_MANIFEST.json")
ACTUAL_ID=$(docker image inspect "$SOURCE" --format '{{.Id}}')
ARCH=$(docker image inspect "$SOURCE" --format '{{.Os}}/{{.Architecture}}')
CANDIDATE=$(docker image inspect "$SOURCE" --format '{{index .Config.Labels "rsdet.candidate.id"}}')
if [[ "$ACTUAL_ID" != "$EXPECTED_ID" || "$ARCH" != "linux/amd64" || "$CANDIDATE" != "$EXPECTED_CANDIDATE" ]]; then
  echo "ERROR: final image identity differs from the built and verified manifest." >&2
  exit 1
fi

echo "Final candidate verified: $CANDIDATE"
echo "Image: $ACTUAL_ID ($ARCH)"
if [[ "${1:-}" == "--check" ]]; then
  echo "Check only; no registry push and no platform submission."
  exit 0
fi

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  read -r -p "请输入比赛网页当前显示的正式 tag（v1.0—v5.0，必须以网页为准）：" TAG
fi
if [[ ! "$TAG" =~ ^v[1-5]\.0$ ]]; then
  echo "ERROR: expected a formal tag v1.0 through v5.0." >&2
  exit 1
fi
TARGET="$REGISTRY:$TAG"
LOCAL_TARGET_ID=$(docker image inspect "$TARGET" --format '{{.Id}}' 2>/dev/null || true)
if [[ -n "$LOCAL_TARGET_ID" && "$LOCAL_TARGET_ID" != "$EXPECTED_ID" ]]; then
  echo "ERROR: local $TARGET points at a different image; verify the website tag." >&2
  exit 1
fi

echo "Will push exactly: $TARGET"
read -r -p "已在网页生成新临时凭证并 docker login，且 tag/地址完全一致？输入 PUSH 继续：" CONFIRM
if [[ "$CONFIRM" != "PUSH" ]]; then
  echo "Cancelled; nothing was pushed."
  exit 0
fi
docker tag "$EXPECTED_ID" "$TARGET"
LOGDIR="$ROOT/outputs/HERA-GUARD-APEX-20260904/submission-push-logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/push-${TAG}-$(date +%Y%m%d-%H%M%S).log"
docker push "$TARGET" 2>&1 | tee "$LOG"
echo "Push completed. Return to the website, verify $TARGET, then click 提交评测."
echo "Push log: $LOG"
