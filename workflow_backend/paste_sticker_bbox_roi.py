#!/usr/bin/env python3
"""
Scale the sticker by the object's bounding box, then paste into a
user-specified axis-aligned rectangle in the scene (position and size are explicit).
Inpaint mask behavior: **contour** (default) expands sticker alpha outward on the scene,
so repaint covers the subject plus a surrounding environmental ring (instead of a hard
rectangle), which usually blends more naturally. **rect** / **alpha** / **hybrid** are
still available.

Approach:
  1. Determine the object AABB in sticker coordinates. By default this is a tight bbox
     over RGBA alpha, or pass a known box via ``--object-bbox`` from detection/SAM.
  2. Crop RGBA inside that bbox, scale with ``--fit`` to ``--insert-rect`` size, and paste
     at the scene rectangle (same contain/cover/stretch semantics as paste_sticker_roi).

Difference vs paste_sticker_roi: that script scales the full sticker canvas to ROI; this
script uses only pixels inside the object bbox, avoiding tiny subjects caused by large blank
areas in the full sticker.

Run from backend/:
  python paste_sticker_bbox_roi.py \\
    --scene outputs/my_scene.png \\
    --sticker /root/fyp/outputs/a_dog/object_sticker_512.png \\
    --insert-rect 256,0,128,256 \\
    --out outputs/my_scene_with_sticker.png \\
    --out-mask outputs/my_scene_roi_mask.png \\
    --write-meta outputs/my_scene_insert_box.json

If you already have an object bbox (sticker pixel xywh):
  --object-bbox 64,32,384,400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from paste_sticker_roi import layout_sticker_on_roi, paste_sticker
from utils.load_layer_image import load_layer_rgba


def _parse_xywh(s: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x,y,w,h")
    return tuple(int(float(p)) for p in parts)


def _parse_xyxy(s: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x0,y0,x1,y1")
    x0, y0, x1, y1 = (int(float(p)) for p in parts)
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("degenerate xyxy")
    return x0, y0, x1, y1


def tight_alpha_bbox(
    rgba: Image.Image,
    alpha_threshold: int = 8,
) -> tuple[int, int, int, int]:
    """Tight axis-aligned bbox of pixels with alpha > threshold. xyxy in sticker pixels."""
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    a = np.asarray(rgba.split()[3], dtype=np.uint8)
    ys, xs = np.where(a > alpha_threshold)
    if ys.size == 0:
        raise ValueError(
            "no opaque pixels in sticker (alpha all <= threshold); "
            "cannot auto object-bbox — set --object-bbox or --object-bbox-xyxy"
        )
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def clip_xyxy_to_image(
    x0: int, y0: int, x1: int, y1: int, w: int, h: int
) -> tuple[int, int, int, int]:
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("object bbox clips to empty region")
    return x0, y0, x1, y1


def make_rect_mask(scene_w: int, scene_h: int, x0: int, y0: int, x1: int, y1: int) -> Image.Image:
    m = Image.new("L", (scene_w, scene_h), 0)
    dr = ImageDraw.Draw(m)
    dr.rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
    return m


def make_alpha_inpaint_mask(
    scene_w: int,
    scene_h: int,
    st_rgba: Image.Image,
    ox: int,
    oy: int,
    *,
    feather: float,
    dilate: int,
) -> Image.Image:
    """
    Full-scene L mask: pasted sticker alpha (optional dilate + Gaussian feather).
    Softer edges than a rectangle; better for SDXL inpaint blending.
    """
    st_rgba = st_rgba.convert("RGBA")
    alpha = st_rgba.split()[3]
    if dilate > 0:
        a = np.asarray(alpha, dtype=np.uint8)
        k = max(1, int(dilate))
        kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
        a = cv2.dilate(a, kernel, iterations=1)
        alpha = Image.fromarray(a, mode="L")
    canvas = Image.new("L", (scene_w, scene_h), 0)
    canvas.paste(alpha, (ox, oy))
    if feather > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    return canvas


def make_hybrid_inpaint_mask(
    scene_w: int,
    scene_h: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    st_rgba: Image.Image,
    ox: int,
    oy: int,
    *,
    alpha_feather: float,
    alpha_dilate: int,
    slot_strength: float,
    slot_feather: float,
) -> Image.Image:
    """
    max(soft_slot, alpha): slot lets letterbox / floor inside insert-rect harmonize;
    alpha keeps strong edit on the subject silhouette.
    """
    alpha_m = make_alpha_inpaint_mask(
        scene_w,
        scene_h,
        st_rgba,
        ox,
        oy,
        feather=alpha_feather,
        dilate=alpha_dilate,
    )
    rect = make_rect_mask(scene_w, scene_h, x0, y0, x1, y1)
    ss = float(np.clip(slot_strength, 0.0, 1.0))
    r = np.asarray(rect, dtype=np.float32) * (255.0 * ss)
    r_img = Image.fromarray(np.clip(r, 0, 255).astype(np.uint8), mode="L")
    if slot_feather > 0:
        r_img = r_img.filter(ImageFilter.GaussianBlur(radius=float(slot_feather)))
    a = np.asarray(alpha_m, dtype=np.float32)
    rr = np.asarray(r_img, dtype=np.float32)
    out = np.maximum(a, rr)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="L")


def raw_alpha_on_canvas(
    scene_w: int,
    scene_h: int,
    st_rgba: Image.Image,
    ox: int,
    oy: int,
) -> Image.Image:
    """Sticker alpha placed on full canvas (no blur); used to locate subject vs slot."""
    st_rgba = st_rgba.convert("RGBA")
    alpha = st_rgba.split()[3]
    canvas = Image.new("L", (scene_w, scene_h), 0)
    canvas.paste(alpha, (ox, oy))
    return canvas


def make_contour_expand_mask(
    scene_w: int,
    scene_h: int,
    st_rgba: Image.Image,
    ox: int,
    oy: int,
    *,
    expand_px: int,
    feather: float,
    alpha_thresh: int = 18,
) -> Image.Image:
    """
    Binary fg from pasted alpha, dilated outward by ``expand_px`` (square kernel),
    then optional Gaussian feather. Inpaint region = subject + a band of surrounding
    scene pixels (not the full insert rectangle).
    """
    raw = raw_alpha_on_canvas(scene_w, scene_h, st_rgba, ox, oy)
    a = np.asarray(raw, dtype=np.uint8)
    fg = np.where(a > alpha_thresh, 255, 0).astype(np.uint8)
    if not np.any(fg):
        return Image.new("L", (scene_w, scene_h), 0)
    ex = max(0, int(expand_px))
    if ex > 0:
        k = 2 * ex + 1
        kernel = np.ones((k, k), np.uint8)
        fg = cv2.dilate(fg, kernel, iterations=1)
    img = Image.fromarray(fg, mode="L")
    if feather > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    return img


def attenuate_mask_on_subject(
    mask_l: Image.Image,
    raw_alpha: Image.Image,
    *,
    edge_px: int,
    atten_interior: float,
    atten_edge: float,
    floor_min: float,
) -> Image.Image:
    """
    Lower inpaint weights on the pasted subject so SDXL does not eat fur/edges;
    keep higher weights on letterbox / floor inside the slot (raw alpha ~ 0).

    - Interior (dist to silhouette > edge_px): multiply by atten_interior (~0.35).
    - Thin edge band: multiply by atten_edge (~0.55), slightly stronger for seam only.
    - floor_min: minimum L on subject pixels so a tiny color harmonization can remain.
    """
    if atten_interior >= 0.999 and atten_edge >= 0.999:
        return mask_l
    m = np.asarray(mask_l, dtype=np.float32)
    ra = np.asarray(raw_alpha, dtype=np.uint8)
    fg = (ra > 18).astype(np.uint8)
    if not np.any(fg):
        return mask_l
    dist_in = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    ed = float(max(1, edge_px))
    interior = (fg > 0) & (dist_in > ed)
    edge_ring = (fg > 0) & (dist_in <= ed)
    fl = float(np.clip(floor_min, 0, 255))
    m = m.copy()
    mi, me = float(atten_interior), float(atten_edge)
    if np.any(interior):
        m[interior] = np.maximum(m[interior] * mi, fl)
    if np.any(edge_ring):
        m[edge_ring] = np.maximum(m[edge_ring] * me, fl * 0.72)
    return Image.fromarray(np.clip(m, 0, 255).astype(np.uint8), mode="L")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scale sticker by object AABB, paste into user-set scene rectangle; emit matching box mask."
    )
    p.add_argument("--scene", type=str, required=True)
    p.add_argument("--sticker", type=str, required=True)
    p.add_argument(
        "--insert-rect",
        type=_parse_xywh,
        metavar="X,Y,W,H",
        required=True,
        help="Insertion slot in scene pixels: top-left (x,y) and size (w,h). Mask matches this rectangle.",
    )
    og = p.add_mutually_exclusive_group()
    og.add_argument(
        "--object-bbox",
        type=_parse_xywh,
        metavar="X,Y,W,H",
        default=None,
        help="Object AABB on the sticker image (xywh). Default: tight bbox from alpha.",
    )
    og.add_argument(
        "--object-bbox-xyxy",
        type=_parse_xyxy,
        metavar="X0,Y0,X1,Y1",
        default=None,
        help="Object AABB on sticker as xyxy corners.",
    )
    p.add_argument(
        "--alpha-threshold",
        type=int,
        default=8,
        help="For auto bbox: min alpha 0–255 treated as foreground.",
    )
    p.add_argument(
        "--fit",
        choices=("contain", "cover", "stretch"),
        default="contain",
        help="How cropped object scales to insert-rect (same semantics as paste_sticker_roi).",
    )
    p.add_argument(
        "--anchor",
        choices=("center", "topleft"),
        default="center",
        help="For fit=contain, placement inside insert-rect.",
    )
    p.add_argument("--out", type=str, required=True, help="Output composite RGB PNG.")
    p.add_argument(
        "--out-mask",
        type=str,
        default=None,
        help="Grayscale inpaint mask, full scene size (see --mask-mode).",
    )
    p.add_argument(
        "--mask-mode",
        choices=("contour", "rect", "alpha", "hybrid"),
        default="contour",
        help="contour=dilated subject + env ring (default); rect|alpha|hybrid=legacy modes.",
    )
    p.add_argument(
        "--mask-contour-expand",
        type=int,
        default=55,
        help="contour mode: dilate subject mask by this many pixels (ring of scene for inpaint).",
    )
    p.add_argument(
        "--mask-feather",
        type=float,
        default=16.0,
        help="Gaussian blur: alpha/hybrid on alpha; contour on final dilated mask.",
    )
    p.add_argument(
        "--mask-dilate",
        type=int,
        default=0,
        help="Morph dilate on alpha before feather (alpha/hybrid).",
    )
    p.add_argument(
        "--mask-slot-strength",
        type=float,
        default=0.42,
        help="hybrid only: min weight inside insert-rect from soft slot (0–1); higher = more floor/context repaint.",
    )
    p.add_argument(
        "--mask-slot-feather",
        type=float,
        default=18.0,
        help="hybrid only: Gaussian blur on slot rectangle before max; softens outer transition.",
    )
    p.add_argument(
        "--preserve-subject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="alpha/hybrid: lower mask on pasted subject to reduce blur/chipping (default on).",
    )
    p.add_argument(
        "--mask-edge-pixels",
        type=int,
        default=6,
        help="preserve-subject: silhouette band width with milder attenuation than interior.",
    )
    p.add_argument(
        "--mask-atten-interior",
        type=float,
        default=0.36,
        help="preserve-subject: multiply mask inside subject core.",
    )
    p.add_argument(
        "--mask-atten-edge",
        type=float,
        default=0.55,
        help="preserve-subject: multiply mask in edge band (still <1).",
    )
    p.add_argument(
        "--mask-subject-floor",
        type=float,
        default=22.0,
        help="preserve-subject: minimum mask L on subject.",
    )
    p.add_argument(
        "--write-meta",
        type=str,
        default=None,
        help="JSON: object bbox, insert rect, paths (for downstream inpaint).",
    )
    p.add_argument(
        "--preview-bbox",
        type=str,
        default=None,
        help="Optional PNG: sticker with object bbox drawn (debug).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    scene_path = Path(args.scene).expanduser().resolve()
    sticker_path = Path(args.sticker).expanduser().resolve()
    if not scene_path.is_file():
        print(f"scene not found: {scene_path}", file=sys.stderr)
        return 1
    if not sticker_path.is_file():
        print(f"sticker not found: {sticker_path}", file=sys.stderr)
        return 1

    scene = Image.open(scene_path).convert("RGB")
    sw, sh = scene.size
    sticker = load_layer_rgba(str(sticker_path))
    iw, ih = sticker.size

    if args.object_bbox_xyxy is not None:
        ox0, oy0, ox1, oy1 = args.object_bbox_xyxy
    elif args.object_bbox is not None:
        ox, oy, ow, oh = args.object_bbox
        ox0, oy0, ox1, oy1 = ox, oy, ox + ow, oy + oh
    else:
        ox0, oy0, ox1, oy1 = tight_alpha_bbox(sticker, args.alpha_threshold)

    ox0, oy0, ox1, oy1 = clip_xyxy_to_image(ox0, oy0, ox1, oy1, iw, ih)

    ix, iy, ins_w, ins_h = args.insert_rect
    x0, y0, x1, y1 = ix, iy, ix + ins_w, iy + ins_h
    x0 = max(0, min(sw, x0))
    y0 = max(0, min(sh, y0))
    x1 = max(0, min(sw, x1))
    y1 = max(0, min(sh, y1))
    if x1 <= x0 or y1 <= y0:
        print("insert-rect is empty or outside scene", file=sys.stderr)
        return 1

    crop = sticker.crop((ox0, oy0, ox1, oy1))
    out_img = paste_sticker(scene, crop, x0, y0, x1, y1, args.fit, args.anchor)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)

    if args.out_mask:
        mp = Path(args.out_mask).expanduser().resolve()
        mp.parent.mkdir(parents=True, exist_ok=True)
        st_placed: Image.Image | None = None
        px = py = 0
        if args.mask_mode == "rect":
            mask = make_rect_mask(sw, sh, x0, y0, x1, y1)
        else:
            st_placed, px, py = layout_sticker_on_roi(
                (sw, sh), crop, x0, y0, x1, y1, args.fit, args.anchor
            )
            if args.mask_mode == "contour":
                mask = make_contour_expand_mask(
                    sw,
                    sh,
                    st_placed,
                    px,
                    py,
                    expand_px=args.mask_contour_expand,
                    feather=args.mask_feather,
                )
            elif args.mask_mode == "alpha":
                mask = make_alpha_inpaint_mask(
                    sw,
                    sh,
                    st_placed,
                    px,
                    py,
                    feather=args.mask_feather,
                    dilate=args.mask_dilate,
                )
            else:
                mask = make_hybrid_inpaint_mask(
                    sw,
                    sh,
                    x0,
                    y0,
                    x1,
                    y1,
                    st_placed,
                    px,
                    py,
                    alpha_feather=args.mask_feather,
                    alpha_dilate=args.mask_dilate,
                    slot_strength=args.mask_slot_strength,
                    slot_feather=args.mask_slot_feather,
                )
        if (
            args.preserve_subject
            and args.mask_mode in ("alpha", "hybrid")
            and st_placed is not None
        ):
            raw_a = raw_alpha_on_canvas(sw, sh, st_placed, px, py)
            mask = attenuate_mask_on_subject(
                mask,
                raw_a,
                edge_px=args.mask_edge_pixels,
                atten_interior=args.mask_atten_interior,
                atten_edge=args.mask_atten_edge,
                floor_min=args.mask_subject_floor,
            )
        mask.save(mp)
        if args.mask_mode == "hybrid":
            print(
                f"Saved mask (hybrid feather={args.mask_feather} dilate={args.mask_dilate} "
                f"slot={args.mask_slot_strength}/{args.mask_slot_feather} "
                f"preserve={args.preserve_subject}): {mp}",
                file=sys.stderr,
            )
        elif args.mask_mode == "contour":
            print(
                f"Saved mask (contour expand={args.mask_contour_expand}px feather={args.mask_feather}): {mp}",
                file=sys.stderr,
            )
        else:
            print(
                f"Saved mask ({args.mask_mode}, feather={args.mask_feather}, dilate={args.mask_dilate}, "
                f"preserve={args.preserve_subject}): {mp}",
                file=sys.stderr,
            )

    meta = {
        "object_bbox_sticker_xyxy": [ox0, oy0, ox1, oy1],
        "object_bbox_sticker_xywh": [ox0, oy0, ox1 - ox0, oy1 - oy0],
        "insert_rect_scene_xyxy": [x0, y0, x1, y1],
        "insert_rect_scene_xywh": [x0, y0, x1 - x0, y1 - y0],
        # Kept consistent with make_box_mask --write-meta / paste_sticker_roi --box-json
        "rect_pixels_xyxy": [x0, y0, x1, y1],
        "rect_pixels_xywh": [x0, y0, x1 - x0, y1 - y0],
        "scene_size_wh": [sw, sh],
        "sticker_path": str(sticker_path),
        "scene_path": str(scene_path),
        "fit": args.fit,
        "anchor": args.anchor,
        "object_bbox_source": (
            "xyxy_arg"
            if args.object_bbox_xyxy is not None
            else ("xywh_arg" if args.object_bbox is not None else "alpha_tight")
        ),
        "inpaint_mask_mode": args.mask_mode,
        "inpaint_mask_contour_expand_px": args.mask_contour_expand,
        "inpaint_mask_feather": args.mask_feather,
        "inpaint_mask_dilate": args.mask_dilate,
        "inpaint_mask_slot_strength": args.mask_slot_strength,
        "inpaint_mask_slot_feather": args.mask_slot_feather,
        "inpaint_preserve_subject": args.preserve_subject,
        "inpaint_mask_edge_pixels": args.mask_edge_pixels,
        "inpaint_mask_atten_interior": args.mask_atten_interior,
        "inpaint_mask_atten_edge": args.mask_atten_edge,
        "inpaint_mask_subject_floor": args.mask_subject_floor,
    }
    if args.write_meta:
        wp = Path(args.write_meta).expanduser().resolve()
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved meta: {wp}", file=sys.stderr)

    if args.preview_bbox:
        pv = sticker.convert("RGBA").copy()
        dr = ImageDraw.Draw(pv)
        dr.rectangle((ox0, oy0, ox1 - 1, oy1 - 1), outline=(255, 0, 0, 255), width=3)
        pp = Path(args.preview_bbox).expanduser().resolve()
        pp.parent.mkdir(parents=True, exist_ok=True)
        pv.save(pp)
        print(f"Saved bbox preview: {pp}", file=sys.stderr)

    print(
        f"Saved: {out_path} | object bbox on sticker [{ox0},{oy0})–[{ox1},{oy1}) | "
        f"insert [{x0},{y0})–[{x1},{y1}) | fit={args.fit}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
