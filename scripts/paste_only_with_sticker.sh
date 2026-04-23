#!/usr/bin/env bash
# =============================================================================
# Skip SD+SAM and use an existing RGBA sticker to test paste + mask + (optional) inpaint.
#
# Usage:
#   SCENE=/path/scene.png STICKER=/path/sticker.png bash scripts/paste_only_with_sticker.sh
#
# Insertion geometry: edit SLOT vars, or switch to --slot-at / positional X,Y,W,H
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SCENE="${SCENE:?set SCENE=<scene image path>}"
STICKER="${STICKER:?set STICKER=<RGBA sticker path>}"
PROMPT="${PROMPT:-paste_only}"

python3 run_workflow.py "$SCENE" "$PROMPT" \
  --skip-sd-sam \
  --object-sticker "$STICKER" \
  --slot-place bottom-left \
  --slot-size "${SLOT_W:-128},${SLOT_H:-256}" \
  --margin "${MARGIN:-20}" \
  --run-name "${RUN_NAME:-paste_only_test}"
