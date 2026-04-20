#!/usr/bin/env bash
# =============================================================================
# 把 collage-diffusion-ui/backend 与 FYP_JAYHF1 里的脚本重新拷入本包（上游修 bug 后运行一次）。
# 在 object_insert_workflow 目录执行：  bash scripts/sync_bundled_from_fyp.sh
# 若你的 fyp 不在默认相对路径，先 export：
#   FYP_BASE=/path/to/fyp
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${FYP_BASE:-$(cd "$ROOT/.." && pwd)}"
BE="$BASE/collage-diffusion-ui/backend"
FYP="$BASE/FYP_JAYHF1"

if [[ ! -f "$BE/sd_sam_pipeline.py" ]]; then
  echo "找不到 backend: $BE" >&2
  exit 1
fi
if [[ ! -f "$FYP/stable_diffusion.py" ]]; then
  echo "找不到 FYP_JAYHF1: $FYP/stable_diffusion.py" >&2
  exit 1
fi

mkdir -p "$ROOT/workflow_backend/utils" "$ROOT/sdxl_inpaint"
cp "$BE/sd_sam_pipeline.py" "$BE/paste_sticker_bbox_roi.py" "$BE/paste_sticker_roi.py" \
  "$BE/test_sd_generation.py" "$ROOT/workflow_backend/"
cp "$BE/utils/load_layer_image.py" "$ROOT/workflow_backend/utils/"
cp "$FYP/stable_diffusion.py" "$ROOT/sdxl_inpaint/"
echo "已更新 workflow_backend/ 与 sdxl_inpaint/stable_diffusion.py"
