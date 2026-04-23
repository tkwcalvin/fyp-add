#!/usr/bin/env bash
# =============================================================================
# Re-copy scripts from collage-diffusion-ui/backend and FYP_JAYHF1 into this package
# (run once after upstream bug fixes).
# Run from object_insert_workflow root:  bash scripts/sync_bundled_from_fyp.sh
# If your fyp path differs from default, export first:
#   FYP_BASE=/path/to/fyp
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${FYP_BASE:-$(cd "$ROOT/.." && pwd)}"
BE="$BASE/collage-diffusion-ui/backend"
FYP="$BASE/FYP_JAYHF1"

if [[ ! -f "$BE/sd_sam_pipeline.py" ]]; then
  echo "backend not found: $BE" >&2
  exit 1
fi
if [[ ! -f "$FYP/stable_diffusion.py" ]]; then
  echo "FYP_JAYHF1 not found: $FYP/stable_diffusion.py" >&2
  exit 1
fi

mkdir -p "$ROOT/workflow_backend/utils" "$ROOT/sdxl_inpaint"
cp "$BE/sd_sam_pipeline.py" "$BE/paste_sticker_bbox_roi.py" "$BE/paste_sticker_roi.py" \
  "$BE/test_sd_generation.py" "$ROOT/workflow_backend/"
cp "$BE/utils/load_layer_image.py" "$ROOT/workflow_backend/utils/"
cp "$FYP/stable_diffusion.py" "$ROOT/sdxl_inpaint/"
echo "Updated workflow_backend/ and sdxl_inpaint/stable_diffusion.py"
