#!/usr/bin/env bash
# =============================================================================
# 跳过 SD+SAM，只用已有 RGBA 贴纸测试粘贴 + 蒙版 +（可选）inpaint。
#
# 用法：
#   SCENE=/path/scene.png STICKER=/path/sticker.png bash scripts/paste_only_with_sticker.sh
#
# 插入几何：改 SLOT 变量，或改成 --slot-at / 位置参数 X,Y,W,H
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SCENE="${SCENE:?设置 SCENE=场景图路径}"
STICKER="${STICKER:?设置 STICKER=RGBA 贴纸路径}"
PROMPT="${PROMPT:-paste_only}"

python3 run_workflow.py "$SCENE" "$PROMPT" \
  --skip-sd-sam \
  --object-sticker "$STICKER" \
  --slot-place bottom-left \
  --slot-size "${SLOT_W:-128},${SLOT_H:-256}" \
  --margin "${MARGIN:-20}" \
  --run-name "${RUN_NAME:-paste_only_test}"
