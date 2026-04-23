#!/usr/bin/env bash
# =============================================================================
# Example: insert a cat at bottom-left, slot size ~= 1/4 image area (512x512 scene -> 256x256 slot).
#
# Prerequisites (same as README dependencies):
#   - CUDA, PyTorch, diffusers, segment_anything, SAM weights (SAM_CHECKPOINT)
#   - SD1.5 (SD_MODEL_ID or SD_HUB_REPO_DIR)
#   - Local SDXL snapshot (directory used by stable_diffusion.py, commonly autodl-tmp/diffusion)
#
# Usage: run from object_insert_workflow root
#   bash scripts/run_cat_bottom_left_quarter.sh
# Or chmod +x first, then ./scripts/run_cat_bottom_left_quarter.sh
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---- Customize as needed ----
SCENE="${SCENE:-/root/fyp/collage-diffusion-ui/backend/outputs/my_scene.png}"
PROMPT="${PROMPT:-a cat}"
# Slot size: sqrt(512*512/4)=256; if your scene is not square, set W,H to match your resolution
SLOT_W="${SLOT_W:-256}"
SLOT_H="${SLOT_H:-256}"
MARGIN="${MARGIN:-24}"
RUN_NAME="${RUN_NAME:-example_cat_bl_quarter}"
# ----------------

python3 run_workflow.py "$SCENE" "$PROMPT" \
  --slot-place bottom-left \
  --slot-size "${SLOT_W},${SLOT_H}" \
  --margin "$MARGIN" \
  --run-name "$RUN_NAME"
