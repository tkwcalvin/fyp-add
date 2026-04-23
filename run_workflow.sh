#!/usr/bin/env bash
# =============================================================================
# Main object insertion entrypoint (equivalent to python3 run_workflow.py ...)
#
# By default, use bundled copies in this directory:
#   workflow_backend/     - SD1.5 + SAM + paste (sd_sam_pipeline, paste_sticker_bbox_roi, etc.)
#   sdxl_inpaint/         - SDXL sticker-fuse (stable_diffusion.py)
# Override with: export WORKFLOW_BACKEND=... or FYP_ROOT=... to point to your own directories.
#
# Examples:
#   ./run_workflow.sh scene.png "a cat" --slot-place bottom-left --slot-size 256,256 --margin 24
#   ./run_workflow.sh scene.png "a dog" 256,0,128,256
# See scripts/ for more examples
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"
exec python3 run_workflow.py "$@"
