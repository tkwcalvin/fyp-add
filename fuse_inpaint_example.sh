#!/usr/bin/env bash
# =============================================================================
# 单独跑 SDXL inpaint（sticker-fuse）：已有 composite + mask 时再融合一版。
#
#   ./fuse_inpaint_example.sh composite.png mask.png [out.png]
#
# 默认同包内：sdxl_inpaint/stable_diffusion.py
# 覆盖：export FYP_ROOT=/path/to/dir   （dir 下须有 stable_diffusion.py）
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
if [[ -f "$HERE/sdxl_inpaint/stable_diffusion.py" ]]; then
  FYP="${FYP_ROOT:-$HERE/sdxl_inpaint}"
else
  FYP="${FYP_ROOT:-$HERE/../FYP_JAYHF1}"
fi
COMP="${1:?第一个参数: 合成图}"
MASK="${2:?第二个参数: 蒙版}"
OUT="${3:-$HERE/runs/last_inpaint_fused.png}"
mkdir -p "$(dirname "$OUT")"
exec python3 "$FYP/stable_diffusion.py" sticker-fuse \
  --composite "$COMP" \
  --mask "$MASK" \
  --out "$OUT"
