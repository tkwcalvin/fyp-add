#!/usr/bin/env bash
# =============================================================================
# Run SDXL inpaint (sticker-fuse) separately when you already have composite + mask.
#
#   ./fuse_inpaint_example.sh composite.png mask.png [out.png]
#
# Default bundled path: sdxl_inpaint/stable_diffusion.py
# Override: export FYP_ROOT=/path/to/dir   (dir must contain stable_diffusion.py)
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
if [[ -f "$HERE/sdxl_inpaint/stable_diffusion.py" ]]; then
  FYP="${FYP_ROOT:-$HERE/sdxl_inpaint}"
else
  FYP="${FYP_ROOT:-$HERE/../FYP_JAYHF1}"
fi
COMP="${1:?first argument: composite image}"
MASK="${2:?second argument: mask image}"
OUT="${3:-$HERE/runs/last_inpaint_fused.png}"
mkdir -p "$(dirname "$OUT")"
exec python3 "$FYP/stable_diffusion.py" sticker-fuse \
  --composite "$COMP" \
  --mask "$MASK" \
  --out "$OUT"
