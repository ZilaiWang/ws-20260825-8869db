#!/usr/bin/env bash
# User-operated push only. Never logs credentials or submits a platform job.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE="xh-detector:p40-full-s1280-frozen0536-final"
EXPECTED="sha256:db2a0eaacc0608eecd80193f2cefb83995214288da0250d405b8f016e8ae1303"
WEIGHT="b0df7981f6ad58fe8eca65fb0deef54feed55caf300c2c219ac1ccb3500c8012"
REGISTRY="competition-registry.cn-beijing.cr.aliyuncs.com/competition/team612528"

ACTUAL=$(docker image inspect "$SOURCE" --format '{{.Id}}')
ARCH=$(docker image inspect "$SOURCE" --format '{{.Os}}/{{.Architecture}}')
MODEL=$(docker image inspect "$SOURCE" --format '{{index .Config.Labels "rsdet.weight.sha256"}}')
if [[ "$ACTUAL" != "$EXPECTED" || "$ARCH" != "linux/amd64" || "$MODEL" != "$WEIGHT" ]]; then
  echo "ERROR: 本地镜像身份与已验收的 P40 不一致，停止推送。" >&2
  exit 1
fi
echo "P40 镜像身份校验通过：$EXPECTED"
echo "部署权重：${WEIGHT}；1280推理；融合后固定阈值0.536。"
if [[ "${1:-}" == "--check" ]]; then
  echo "仅检查完成，没有推送，也没有提交官方评测。"
  exit 0
fi
TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  read -r -p "请输入比赛网页当前显示的正式 tag（例如 v2.0，不要猜）：" TAG
fi
if [[ ! "$TAG" =~ ^v[1-5]\.0$ ]]; then
  echo "ERROR: 这里只接受网页分配的正式 v1.0—v5.0，不接受 trial 或自定义 tag。" >&2
  exit 1
fi
TARGET="$REGISTRY:$TAG"
OLD=$(docker image inspect "$TARGET" --format '{{.Id}}' 2>/dev/null || true)
if [[ -n "$OLD" && "$OLD" != "$EXPECTED" ]]; then
  echo "ERROR: 本地 $TARGET 已指向另一镜像；请先核对网页 tag，未自动覆盖。" >&2
  exit 1
fi
echo "将推送：$TARGET"
read -r -p "已确认网页地址完全一致，并且已用网页临时凭证登录？输入 PUSH 继续：" CONFIRM
if [[ "$CONFIRM" != "PUSH" ]]; then
  echo "已取消，没有推送。"
  exit 0
fi
docker tag "$EXPECTED" "$TARGET"
LOGDIR="$ROOT/outputs/P40-DEPLOYMENT-PREFLIGHT-20260903"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/push-${TAG}-$(date +%Y%m%d-%H%M%S).log"
docker push "$TARGET" 2>&1 | tee "$LOG"
echo "推送命令成功。核对上方 digest；回网站确认 $TARGET 后点击提交评测。"
echo "若网站显示的 tag 已变化，停止操作并重新核对；不要覆盖已有 tag。"
echo "本脚本不会点击提交评测；推送日志：$LOG"
