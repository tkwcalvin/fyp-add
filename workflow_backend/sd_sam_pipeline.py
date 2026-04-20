#!/usr/bin/env python3
"""
Pipeline: text prompt → Stable Diffusion 1.5 image → mask → RGBA / sticker outputs.

**Why collage demos (e.g. demo_completion.py) look “perfectly aligned”:** they never
infer a mask from pixels. ``demo_completion`` → ``StableDiffusionControlNetPipeline.preprocess_layers``
builds each layer’s alpha from **polygon metadata**, composites ``composite_image`` and
``mask_layers`` on the **same 512×512 canvas**, then ``generate_mask`` / ``generate_control_mask``
only blend those **authoritative** masks (see pipeline_controlnet.py). The mask and image
match because they share one geometric source.

**This script** usually estimates a mask with **SAM** after generation, so alignment is
best-effort. For **pixel-identical** cutouts like the collage path, pass ``--mask-image``
(a grayscale image, white = keep): it is resized to the generated resolution and SAM is
skipped—same “single source of truth” pattern if you drew the mask for that frame.

Geared toward **sticker / texture cutouts**: whole subject in frame (e.g. full-body dog),
not a tight face-only portrait. Defaults bias SD + mask selection that way.

Outputs (default under /root/fyp/outputs/<run_name>/):
  - generated.png       — full SD RGB image
  - mask.png            — binary mask (0/255), always same H×W as generated.png
  - object_rgba.png     — full canvas, RGBA (alpha = mask); transparent background
  - object_on_white.png — full canvas, subject composited on white (no alpha)
  - object_crop_rgba.png — tight bbox around mask + padding; RGBA sticker crop
  - object_sticker_<N>.png — optional: letterboxed RGBA square (see --sticker-size)

Run from backend directory:
  python sd_sam_pipeline.py --prompt "a golden retriever dog"
  python sd_sam_pipeline.py --prompt "a wooden desk" --sam-preset object
  python sd_sam_pipeline.py --prompt "a mug" --sam-clip-rerank
  # Optional: pip install rembg &&  python sd_sam_pipeline.py --prompt "a lamp" --matting rembg

Env:
  SD_MODEL_ID, SD_HUB_REPO_DIR — same as test_sd_generation.py
  SAM_CHECKPOINT — path to SAM .pth (default: /root/autodl-tmp/sam_vit_h_4b8939.pth)
  SD_SAM_CLIP_MODEL_ID — optional; local dir or hub id for CLIP rerank (default: openai/clip-vit-base-patch32).
  CLIP load temporarily disables HF_HUB_ENABLE_HF_TRANSFER (hf_transfer often breaks on xet downloads).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from test_sd_generation import pick_model_id

# Default prompt suffix: wide framing, full subject, isolated — reduces “head-only” crops.
# Keep short: SD1.5 CLIP truncates at 77 tokens; a long suffix was cutting off "high quality".
STICKER_PROMPT_SUFFIX = (
    ", full body in frame, single subject, centered, pure white background, no floor, "
    "studio photo, sharp focus"
)

# Push SD away from portrait crops and partial subjects.
STICKER_DEFAULT_NEGATIVE = (
    "close-up, extreme close-up, cropped, tight crop, head only, face only, "
    "portrait photo, bust shot, shoulders up, partial body, cut off legs, "
    "missing limbs, out of frame body, macro, selfie angle, "
    "blurry, low quality, watermark, text, multiple animals, duplicate subject, "
    "wooden floor, floorboards, ground, pavement, grass ground"
)

DEFAULT_SAM_CKPT = "/root/autodl-tmp/sam_vit_h_4b8939.pth"
DEFAULT_OUTPUT_ROOT = Path("/root/fyp/outputs")

LOG = logging.getLogger("sd_sam")


def _mask_bbox_bool(seg: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(seg)
    if ys.size == 0:
        return 0, 0, 0, 0
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _log_mask_summary(tag: str, seg: np.ndarray, h: int, w: int) -> None:
    total = float(h * w)
    area = int(np.sum(seg))
    bx, by, bw, bh = _mask_bbox_bool(seg)
    cy = float(np.mean(np.where(seg)[0])) if area else 0.0
    cx = float(np.mean(np.where(seg)[1])) if area else 0.0
    LOG.info(
        "%s: fg_pixels=%d (%.1f%% canvas) bbox=(%d,%d,%d,%d) centroid≈(%.0f,%.0f)",
        tag,
        area,
        100.0 * area / total,
        bx,
        by,
        bw,
        bh,
        cx,
        cy,
    )


def pick_sam_multimask(
    masks_sam: np.ndarray,
    iou_scores: np.ndarray,
    pick: str,
    tie_margin: float,
    ih: int,
    iw: int,
) -> tuple[int, np.ndarray]:
    """
    SAM ``multimask_output=True`` returns 3 options; argmax(iou) often prefers a tight
    fragment. ``largest_in_tie`` keeps masks whose model iou is within ``tie_margin`` of
    the best and picks the largest area (better for full-object cutouts).
    """
    scores = np.asarray(iou_scores, dtype=np.float64).ravel()
    n = min(int(masks_sam.shape[0]), scores.size)
    areas: list[int] = []
    bboxes: list[tuple[int, int, int, int]] = []
    for i in range(n):
        m = masks_sam[i]
        seg = (m > 0.5) if m.dtype != np.bool_ else m.astype(bool)
        areas.append(int(np.sum(seg)))
        bboxes.append(_mask_bbox_bool(seg))

    for i in range(n):
        LOG.info(
            "  multimask[%d]: model_iou=%.4f area=%d bbox=%s",
            i,
            float(scores[i]),
            areas[i],
            bboxes[i],
        )

    if pick == "iou":
        chosen = int(np.argmax(scores[:n]))
        LOG.info("  pick=iou → using multimask[%d]", chosen)
        return chosen, masks_sam[chosen]

    best_s = float(scores[:n].max())
    thr = best_s - tie_margin
    eligible = [i for i in range(n) if scores[i] >= thr]

    def near_full_frame(i: int) -> bool:
        _x, _y, bw, bh = bboxes[i]
        return bh >= ih * 0.88 and bw >= iw * 0.68

    # Prefer not to pick a mask that is almost the whole canvas (often floor+subject or bg).
    filtered = [i for i in eligible if not near_full_frame(i)]
    pool = filtered if filtered else eligible
    chosen = max(pool, key=lambda i: areas[i])
    if pool != eligible:
        LOG.info(
            "  largest_in_tie: dropped near-fullframe from %s → using pool %s",
            eligible,
            pool,
        )
    LOG.info(
        "  pick=largest_in_tie (margin=%.3f, eligible=%s) → multimask[%d] area=%d",
        tie_margin,
        eligible,
        chosen,
        areas[chosen],
    )
    return chosen, masks_sam[chosen]


_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def build_sam_prompt_points(
    preset: str,
    w: int,
    h: int,
    sam_fg_yfrac: float,
    sam_bg_yfrac: float,
    no_floor: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Build (point_coords, point_labels, multimask_output).

    ``object`` preset: center foreground + four corner background points (label 0).
    SAM works better with ``multimask_output=False`` when several points are given.
    """
    wf, hf = float(w), float(h)
    inset = max(4.0, min(wf, hf) * 0.04)

    if preset == "center":
        pts = np.array([[wf * 0.5, hf * 0.5]], dtype=np.float32)
        return pts, np.array([1], dtype=np.int32), True

    if preset == "object":
        fg = np.array([[wf * 0.5, hf * 0.48]], dtype=np.float32)
        corners = np.array(
            [
                [inset, inset],
                [wf - 1.0 - inset, inset],
                [inset, hf - 1.0 - inset],
                [wf - 1.0 - inset, hf - 1.0 - inset],
            ],
            dtype=np.float32,
        )
        pts = np.vstack([fg, corners])
        lbs = np.array([1, 0, 0, 0, 0], dtype=np.int32)
        return pts, lbs, False

    # animal (default): upper-mid torso + optional bottom “floor” negative
    fg = np.array([[wf * 0.5, hf * sam_fg_yfrac]], dtype=np.float32)
    if no_floor:
        return fg, np.array([1], dtype=np.int32), True
    bg = np.array([[wf * 0.5, hf * sam_bg_yfrac]], dtype=np.float32)
    return np.vstack([fg, bg]), np.array([1, 0], dtype=np.int32), True


def _ensure_clip_for_rerank(device: torch.device) -> None:
    """Load CLIP once; disable hf_transfer during download (avoids broken xet/hf_transfer paths)."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is not None:
        return
    from transformers import CLIPModel, CLIPProcessor

    raw = os.environ.get("SD_SAM_CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
    expanded = str(Path(raw).expanduser().resolve())
    model_id = expanded if Path(expanded).is_dir() else raw

    prev_transfer = os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    try:
        LOG.info(
            "Loading CLIP for rerank: %s (HF_HUB_ENABLE_HF_TRANSFER cleared for this load)",
            model_id,
        )
        m = CLIPModel.from_pretrained(model_id).to(device)
        p = CLIPProcessor.from_pretrained(model_id)
        m.eval()
        _CLIP_MODEL = m
        _CLIP_PROCESSOR = p
    except (OSError, RuntimeError) as e:
        raise RuntimeError(
            f"CLIP load failed ({model_id}). Try: export HF_HUB_ENABLE_HF_TRANSFER=0 globally, "
            f"or set SD_SAM_CLIP_MODEL_ID to a local snapshot (see HF cache under "
            f"hub/models--openai--clip-vit-base-patch32/snapshots/<hash>). Original: {e}"
        ) from e
    finally:
        if prev_transfer is not None:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = prev_transfer


def clip_rerank_multimasks(
    rgb: np.ndarray,
    masks_sam: np.ndarray,
    text_prompt: str,
    device: torch.device,
) -> int:
    """Pick multimask index that maximizes CLIP image–text similarity on a cutout composite."""
    _ensure_clip_for_rerank(device)

    model = _CLIP_MODEL
    processor = _CLIP_PROCESSOR
    n = int(masks_sam.shape[0])
    gray = np.full_like(rgb, 122)
    scores: list[float] = []
    for i in range(n):
        m = masks_sam[i]
        seg = (m > 0.5) if m.dtype != np.bool_ else m.astype(bool)
        comp = np.where(seg[..., None], rgb, gray).astype(np.uint8)
        pil = Image.fromarray(comp)
        inp = processor(text=[text_prompt], images=pil, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp)
            scores.append(float(out.logits_per_image[0, 0].detach().cpu()))
    bi = int(np.argmax(scores))
    LOG.info("CLIP rerank (prompt=%r): scores=%s → multimask[%d]", text_prompt[:80], scores, bi)
    return bi


def _configure_logging(verbose: bool) -> None:
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOG.propagate = False
    if LOG.handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(levelname)s [sd_sam] %(message)s"))
    LOG.addHandler(h)


def slugify(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return (s[:max_len] or "run").lower()


def _mask_merges_floor(m: dict, height: int, width: int, total: float) -> bool:
    """Heuristic: wide mask glued to image bottom is often floor + subject (automatic SAM)."""
    x, y, bw, bh = (float(t) for t in m["bbox"])
    area = float(m.get("area", int(m["segmentation"].sum())))
    bottom = y + bh
    if bottom >= height - 4 and bw >= width * 0.70 and area / total >= 0.22:
        return True
    return False


def pick_best_mask(
    masks: list[dict],
    height: int,
    width: int,
    max_cover: float = 0.92,
    min_area_ratio: float = 0.002,
    area_preference: float = 0.35,
    exclude_floor_merge: bool = True,
) -> dict | None:
    """
    Choose one SAM mask: skip near-full-image and tiny blobs; prefer high IoU * stability.
    When ``area_preference > 0``, boost larger plausible foreground masks so we keep
    whole-object regions instead of a small face patch when SD still produces clutter.
    """
    if not masks:
        return None
    total = float(height * width)
    min_area = max(64.0, min_area_ratio * total)

    scored: list[tuple[float, dict]] = []
    for m in masks:
        seg = m["segmentation"]
        area = float(m.get("area", int(seg.sum())))
        if area >= max_cover * total or area < min_area:
            continue
        if exclude_floor_merge and _mask_merges_floor(m, height, width, total):
            continue
        iou = float(m.get("predicted_iou", 0.0))
        stab = float(m.get("stability_score", 0.0))
        score = iou * stab * np.log1p(area)
        if area_preference > 0:
            # Favor masks covering more of the canvas (full-body cutouts vs. tiny head blob).
            score *= (area / total) ** area_preference
        scored.append((score, m))

    if not scored:
        # Fallback: largest mask that is not full frame
        for m in sorted(masks, key=lambda x: -x.get("area", int(x["segmentation"].sum()))):
            area = float(m.get("area", int(m["segmentation"].sum())))
            if area < max_cover * total and area >= min_area:
                if exclude_floor_merge and _mask_merges_floor(m, height, width, total):
                    continue
                return m
        return masks[0] if masks else None

    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def align_mask_to_pil(mask: np.ndarray, pil: Image.Image) -> np.ndarray:
    """Force mask to (height, width) == pil.size[::-1], nearest-neighbor if SAM size drifts."""
    m = np.asarray(mask)
    if m.dtype == bool:
        m = m.astype(np.uint8) * 255
    elif m.max() <= 1:
        m = (m > 0).astype(np.uint8) * 255
    else:
        m = (m > 127).astype(np.uint8) * 255
    th, tw = pil.height, pil.width
    if m.shape[:2] != (th, tw):
        print(
            f"Warning: mask shape {m.shape[:2]} != image ({th}, {tw}); resizing mask to match.",
            file=sys.stderr,
        )
        m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
    return m


def load_user_mask(mask_path: Path, pil: Image.Image) -> np.ndarray:
    """Load L mask from disk; resize to ``pil`` size; return uint8 0/255 aligned to pil."""
    m = Image.open(mask_path).convert("L")
    if m.size != (pil.width, pil.height):
        m = m.resize((pil.width, pil.height), Image.Resampling.NEAREST)
    return align_mask_to_pil(np.array(m), pil)


def rgba_full_frame(rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) + mask (H,W) → RGBA (H,W,4)."""
    h, w = mask_u8.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = rgb[:, :, :3]
    out[:, :, 3] = mask_u8
    return out


def composite_on_white(rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Alpha-blend subject onto white; output RGB uint8, same size."""
    a = (mask_u8.astype(np.float32) / 255.0)[..., None]
    fg = rgb.astype(np.float32)
    bg = np.full_like(fg, 255.0)
    out = fg * a + bg * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_rgba_by_mask(
    rgb: np.ndarray, mask_u8: np.ndarray, padding: int
) -> np.ndarray | None:
    """Tight crop around mask foreground; returns RGBA or None if mask empty."""
    fg = mask_u8 > 127
    if not np.any(fg):
        return None
    ys, xs = np.where(fg)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h, w = mask_u8.shape[:2]
    y0 = max(0, y0 - padding)
    y1 = min(h - 1, y1 + padding)
    x0 = max(0, x0 - padding)
    x1 = min(w - 1, x1 + padding)
    rgb_c = rgb[y0 : y1 + 1, x0 : x1 + 1].copy()
    m_c = mask_u8[y0 : y1 + 1, x0 : x1 + 1].copy()
    return rgba_full_frame(rgb_c, m_c)


def build_sam(model_type: str, checkpoint: Path, device: torch.device):
    ckpt = str(checkpoint.expanduser().resolve())
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    sam.to(device)
    sam.eval()
    return sam


def letterbox_rgba_square(rgba: np.ndarray, side: int) -> np.ndarray:
    """Fit RGBA crop inside ``side``×``side`` with transparency, aspect preserved."""
    h, w = rgba.shape[:2]
    if h < 1 or w < 1:
        return np.zeros((side, side, 4), dtype=np.uint8)
    scale = min(side / w, side / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    y0 = (side - nh) // 2
    x0 = (side - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SD generate + SAM segment → mask + RGBA object.")
    p.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Object / scene description, e.g. "a ceramic mug, studio lighting, plain background".',
    )
    p.add_argument(
        "--prompt-suffix",
        type=str,
        default=STICKER_PROMPT_SUFFIX,
        help="Appended to prompt (sticker / full-subject bias). Set empty to disable.",
    )
    p.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Overrides built-in sticker negative prompt. Use with --no-negative to disable negatives.",
    )
    p.add_argument(
        "--no-negative",
        action="store_true",
        help="Do not pass a negative prompt to SD.",
    )
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width (slightly portrait default height helps standing full-body subjects).",
    )
    p.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height; default 512 (square sticker texture). Use 640 for tall full-body.",
    )
    p.add_argument("--seed", type=int, default=49, help="Use -1 for random.")
    p.add_argument("--model", type=str, default=None, help="SD model id or path (see test_sd_generation).")
    p.add_argument("--cpu", action="store_true")

    p.add_argument(
        "--sam-checkpoint",
        type=str,
        default=os.environ.get("SAM_CHECKPOINT", DEFAULT_SAM_CKPT),
        help="Path to SAM .pth weights.",
    )
    p.add_argument(
        "--sam-model",
        type=str,
        default="vit_h",
        choices=list(sam_model_registry.keys()),
        help="SAM backbone registry key.",
    )

    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_ROOT}/<run-name>).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subfolder under the default output root when --out-dir is not set.",
    )
    p.add_argument(
        "--min-mask-area-ratio",
        type=float,
        default=0.012,
        help="Ignore SAM masks smaller than this fraction of the image (default 0.012).",
    )
    p.add_argument(
        "--mask-area-preference",
        type=float,
        default=0.35,
        help="0 disables; >0 prefers larger masks (default 0.35) for whole-subject cutouts.",
    )
    p.add_argument(
        "--cutout-padding",
        type=int,
        default=12,
        help="Pixels of padding around mask bbox for object_crop_rgba.png.",
    )
    p.add_argument(
        "--no-crop-output",
        action="store_true",
        help="Do not write object_crop_rgba.png.",
    )
    p.add_argument(
        "--sticker-size",
        type=int,
        default=512,
        help="If >0, write object_sticker_<N>.png (letterboxed square RGBA). 0 to skip.",
    )
    p.add_argument(
        "--sam-mode",
        type=str,
        choices=["point", "auto"],
        default="point",
        help="point: SAM with preset points. auto: dense grid (slower; try with object preset).",
    )
    p.add_argument(
        "--sam-preset",
        type=str,
        choices=["animal", "object", "center"],
        default="animal",
        help="animal: center-x + upper-mid fg + bottom bg (quadrupeds). "
        "object: center fg + four corner bg labels (furniture/products on seamless bg). "
        "center: single center click.",
    )
    p.add_argument(
        "--matting",
        type=str,
        choices=["sam", "rembg"],
        default="sam",
        help="sam: Segment Anything (default). rembg: U2Net matting (pip install rembg); strong on products.",
    )
    p.add_argument(
        "--sam-clip-rerank",
        action="store_true",
        help="After SAM multimask (single-fg mode only), pick mask with best CLIP match to --prompt.",
    )
    p.add_argument(
        "--no-sam-floor-hint",
        action="store_true",
        help="animal preset only: single fg point, no bottom background point.",
    )
    p.add_argument(
        "--sam-fg-yfrac",
        type=float,
        default=0.42,
        help="Point mode: foreground y as fraction of height (0=top). ~0.42 targets torso.",
    )
    p.add_argument(
        "--sam-bg-yfrac",
        type=float,
        default=0.96,
        help="Point mode: background point y when floor hint is on (label 0 = not object).",
    )
    p.add_argument(
        "--mask-image",
        type=str,
        default=None,
        help="Grayscale mask file (white=foreground). Resized to generated W×H; SAM is skipped. "
        "Use for collage-style alignment when the mask is authored for this image.",
    )
    p.add_argument(
        "--sam-multimask-pick",
        type=str,
        choices=["iou", "largest_in_tie"],
        default="iou",
        help="Among 3 SAM point masks: iou=model quality head (default). largest_in_tie=biggest area "
        "among similar scores, skipping near-full-image boxes.",
    )
    p.add_argument(
        "--sam-tie-margin",
        type=float,
        default=0.12,
        help="largest_in_tie: include masks with model_iou >= best - margin.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging (multimask stats, mask summary).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.verbose)

    full_prompt = args.prompt + (args.prompt_suffix or "")

    if args.no_negative:
        negative_prompt = None
    elif args.negative_prompt is not None:
        negative_prompt = args.negative_prompt or None
    else:
        negative_prompt = STICKER_DEFAULT_NEGATIVE
    run_name = args.run_name or slugify(args.prompt)
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (DEFAULT_OUTPUT_ROOT / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda" and not args.cpu:
        print("Warning: CUDA not available; using CPU.", file=sys.stderr)

    sd_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_id = pick_model_id(args.model)

    LOG.info("SD model: %s", model_id)
    if len(full_prompt) > 220:
        LOG.warning(
            "Prompt is long (%d chars); SD1.5 CLIP may truncate at ~77 tokens — "
            "shorten --prompt-suffix if conditioning looks wrong.",
            len(full_prompt),
        )
    LOG.info("Generating (%dx%d): %s", args.width, args.height, full_prompt[:160])
    print(f"SD model: {model_id}")
    print(f"Generating: {full_prompt[:120]!r}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=sd_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    generator = None
    if args.seed >= 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    result = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        generator=generator,
    )
    pil_rgb = result.images[0]
    gen_path = out_dir / "generated.png"
    pil_rgb.save(gen_path)
    print(f"Saved: {gen_path}")
    LOG.info("Saved generated: %s (PIL size %s)", gen_path, pil_rgb.size)

    rgb = np.array(pil_rgb.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    LOG.debug("RGB array shape=%s (expect H,W = %d,%d)", rgb.shape, h, w)

    user_mask_path = (
        Path(args.mask_image).expanduser().resolve() if args.mask_image else None
    )
    if user_mask_path is not None:
        if not user_mask_path.is_file():
            print(f"Mask file not found: {user_mask_path}", file=sys.stderr)
            return 1
        mask_u8 = load_user_mask(user_mask_path, pil_rgb)
        print(f"Using user mask (SAM skipped), same alignment as collage-style pipelines: {user_mask_path}")
    elif args.matting == "rembg":
        try:
            from rembg import remove
        except ImportError:
            print(
                "rembg is not installed. Run: pip install rembg",
                file=sys.stderr,
            )
            return 1
        LOG.info("Matting backend: rembg (U2Net)")
        print("Matting: rembg (U2Net)")
        out_rgba = remove(Image.fromarray(rgb))
        mask_u8 = align_mask_to_pil(np.asarray(out_rgba.convert("RGBA"))[:, :, 3], pil_rgb)
    else:
        sam_device = device if device.type == "cuda" else torch.device("cpu")
        sam_ckpt = Path(args.sam_checkpoint)
        print(f"Loading SAM ({args.sam_model}): {sam_ckpt}")
        sam = build_sam(args.sam_model, sam_ckpt, sam_device)

        if args.sam_mode == "point":
            predictor = SamPredictor(sam)
            predictor.set_image(rgb)
            pts, lbs, use_multimask = build_sam_prompt_points(
                args.sam_preset,
                w,
                h,
                args.sam_fg_yfrac,
                args.sam_bg_yfrac,
                args.no_sam_floor_hint,
            )
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            LOG.info(
                "SAM point preset=%s multimask_output=%s points=%d first_fg=(%.1f,%.1f)",
                args.sam_preset,
                use_multimask,
                len(pts),
                float(pts[0, 0]),
                float(pts[0, 1]),
            )
            if args.sam_clip_rerank and not use_multimask:
                LOG.warning(
                    "--sam-clip-rerank needs 3 multimasks; ignored for preset=%r "
                    "(try --sam-preset center --sam-clip-rerank, or animal).",
                    args.sam_preset,
                )
            masks_sam, iou_scores, _ = predictor.predict(
                point_coords=pts,
                point_labels=lbs,
                multimask_output=use_multimask,
            )
            iou_flat = np.asarray(iou_scores, dtype=np.float64).ravel()
            if use_multimask:
                if args.sam_clip_rerank:
                    bi = clip_rerank_multimasks(rgb, masks_sam, args.prompt, sam_device)
                    m = masks_sam[bi]
                    iou_pick = float(iou_flat[bi]) if bi < iou_flat.size else 0.0
                    print(
                        f"SAM point ({args.sam_preset}): CLIP rerank → multimask[{bi}] model_iou={iou_pick:.3f}"
                    )
                else:
                    bi, m = pick_sam_multimask(
                        masks_sam,
                        iou_scores,
                        args.sam_multimask_pick,
                        args.sam_tie_margin,
                        h,
                        w,
                    )
                    iou_pick = float(iou_flat[bi]) if bi < iou_flat.size else 0.0
                    print(
                        f"SAM point ({args.sam_preset}): multimask[{bi}] model_iou={iou_pick:.3f}"
                    )
            else:
                m = masks_sam[0]
                bi = 0
                iou_pick = float(iou_flat[0]) if iou_flat.size else 0.0
                print(f"SAM point ({args.sam_preset}): single mask model_iou={iou_pick:.3f}")
            seg = (m > 0.5) if m.dtype != np.bool_ else m.astype(bool)
        else:
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if args.sam_preset == "object":
                mask_generator = SamAutomaticMaskGenerator(
                    sam,
                    pred_iou_thresh=0.92,
                    stability_score_thresh=0.96,
                    min_mask_region_area=100,
                )
                LOG.info("SAM auto: stricter thresholds for object preset")
            else:
                mask_generator = SamAutomaticMaskGenerator(sam)
            masks = mask_generator.generate(img_bgr)
            LOG.info("SAM auto: raw mask count=%d", len(masks))
            if args.verbose and masks:
                top = sorted(
                    masks,
                    key=lambda x: float(x.get("predicted_iou", 0))
                    * float(x.get("stability_score", 0)),
                    reverse=True,
                )[:8]
                for j, mk in enumerate(top):
                    ar = int(mk.get("area", int(mk["segmentation"].sum())))
                    LOG.debug(
                        "  auto candidate[%d]: area=%d iou=%s stab=%s bbox=%s",
                        j,
                        ar,
                        mk.get("predicted_iou"),
                        mk.get("stability_score"),
                        mk.get("bbox"),
                    )
            best = pick_best_mask(
                masks,
                h,
                w,
                min_area_ratio=args.min_mask_area_ratio,
                area_preference=args.mask_area_preference,
                exclude_floor_merge=True,
            )
            if best is None:
                print("SAM auto mode returned no usable mask.", file=sys.stderr)
                return 1
            seg = np.asarray(best["segmentation"], dtype=bool)

        if not np.any(seg):
            print("SAM produced an empty mask.", file=sys.stderr)
            return 1

        seg = np.asarray(seg, dtype=bool)
        mask_u8 = align_mask_to_pil(seg.astype(np.uint8) * 255, pil_rgb)

    if not np.any(mask_u8 > 127):
        print("Mask has no foreground pixels.", file=sys.stderr)
        return 1

    assert rgb.shape[:2] == mask_u8.shape[:2], (rgb.shape, mask_u8.shape)
    _log_mask_summary("Cutout mask (aligned to generated image)", mask_u8 > 127, h, w)

    mask_path = out_dir / "mask.png"
    cv2.imwrite(str(mask_path), mask_u8)
    print(
        f"Saved: {mask_path} ({mask_u8.shape[1]}×{mask_u8.shape[0]}, "
        f"fg_pixels={int((mask_u8 > 127).sum())})"
    )

    object_path = out_dir / "object_rgba.png"
    Image.fromarray(rgba_full_frame(rgb, mask_u8), mode="RGBA").save(object_path)
    print(f"Saved: {object_path}")

    white_path = out_dir / "object_on_white.png"
    Image.fromarray(composite_on_white(rgb, mask_u8), mode="RGB").save(white_path)
    print(f"Saved: {white_path}")

    crop: np.ndarray | None = None
    if not args.no_crop_output or args.sticker_size > 0:
        crop = crop_rgba_by_mask(rgb, mask_u8, args.cutout_padding)

    if not args.no_crop_output:
        if crop is not None:
            crop_path = out_dir / "object_crop_rgba.png"
            Image.fromarray(crop, mode="RGBA").save(crop_path)
            print(f"Saved: {crop_path} ({crop.shape[1]}×{crop.shape[0]})")
        else:
            print("Skipped object_crop_rgba.png (empty mask).", file=sys.stderr)

    if args.sticker_size > 0:
        if crop is not None:
            sq = letterbox_rgba_square(crop, args.sticker_size)
            sticker_path = out_dir / f"object_sticker_{args.sticker_size}.png"
            Image.fromarray(sq, mode="RGBA").save(sticker_path)
            print(f"Saved: {sticker_path} ({args.sticker_size}×{args.sticker_size} letterbox)")
        else:
            print("Skipped sticker square (no crop).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
