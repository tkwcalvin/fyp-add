#!/usr/bin/env bash
# =============================================================================
# 示例：左下角插入猫，插入槽约 = 整图面积的 1/4（512×512 场景 → 槽 256×256）。
#
# 前置（与 README「依赖」一致）：
#   - CUDA、PyTorch、diffusers、segment_anything、SAM 权重（SAM_CHECKPOINT）
#   - SD1.5（SD_MODEL_ID 或 SD_HUB_REPO_DIR）
#   - SDXL 本地快照（stable_diffusion.py 使用的目录，常见 autodl-tmp/diffusion）
#
# 用法：在 object_insert_workflow 目录执行
#   bash scripts/run_cat_bottom_left_quarter.sh
# 或先 chmod +x 再 ./scripts/run_cat_bottom_left_quarter.sh
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---- 按需修改 ----
SCENE="${SCENE:-/root/fyp/collage-diffusion-ui/backend/outputs/my_scene.png}"
PROMPT="${PROMPT:-a cat}"
# 槽宽高：sqrt(512*512/4)=256；若场景不是正方形，可改成与你分辨率匹配的 W,H
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
