#!/usr/bin/env bash
# =============================================================================
# 物体插入主入口（等价于 python3 run_workflow.py …）
#
# 默认使用本目录内副本：
#   workflow_backend/     — SD1.5 + SAM + 粘贴（sd_sam_pipeline、paste_sticker_bbox_roi 等）
#   sdxl_inpaint/         — SDXL sticker-fuse（stable_diffusion.py）
# 覆盖方式：export WORKFLOW_BACKEND=… 或 FYP_ROOT=… 指向你自己的目录。
#
# 示例：
#   ./run_workflow.sh scene.png "a cat" --slot-place bottom-left --slot-size 256,256 --margin 24
#   ./run_workflow.sh scene.png "a dog" 256,0,128,256
# 更多示例见 scripts/
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"
exec python3 run_workflow.py "$@"
