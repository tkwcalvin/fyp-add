#!/usr/bin/env python3
"""
Paste an RGBA sticker (e.g. sd_sam_pipeline object_sticker / object_crop) into a
rectangular ROI on a scene image, alpha-blended — same idea as preview_collage_stages
(ImageLayer composite) but minimal: no diffusion, fixed axis-aligned box.

ROI can come from make_box_mask --write-meta JSON (--box-json) or --rect-pixels.

Run from backend/:
  python paste_sticker_roi.py \\
    --scene outputs/my_scene.png \\
    --sticker /root/fyp/outputs/a_dog/object_sticker_512.png \\
    --box-json outputs/my_scene_roi_mask_box.json \\
    --out outputs/my_scene_with_dog.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from utils.load_layer_image import load_layer_rgba


def _parse_xywh(s: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x,y,w,h")
    return tuple(int(float(p)) for p in parts)


def rect_from_box_json(path: Path) -> tuple[int, int, int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "rect_pixels_xyxy" in data:
        x0, y0, x1, y1 = (int(v) for v in data["rect_pixels_xyxy"])
        return x0, y0, x1, y1
    if "rect_pixels_xywh" in data:
        x, y, w, h = (int(v) for v in data["rect_pixels_xywh"])
        return x, y, x + w, y + h
    raise KeyError("JSON needs rect_pixels_xyxy or rect_pixels_xywh")


def resize_sticker_for_roi(
    sticker: Image.Image,
    roi_w: int,
    roi_h: int,
    fit: str,
) -> Image.Image:
    """Return RGBA image sized to paste into ROI (exact size for stretch/cover)."""
    sw, sh = sticker.size
    if sw < 1 or sh < 1:
        raise ValueError("sticker has empty size")
    if fit == "stretch":
        return sticker.resize((roi_w, roi_h), Image.Resampling.LANCZOS)

    if fit == "contain":
        scale = min(roi_w / sw, roi_h / sh)
    elif fit == "cover":
        scale = max(roi_w / sw, roi_h / sh)
    else:
        raise ValueError(fit)

    nw = max(1, int(round(sw * scale)))
    nh = max(1, int(round(sh * scale)))
    resized = sticker.resize((nw, nh), Image.Resampling.LANCZOS)

    if fit == "contain":
        return resized

    # cover: center-crop to roi_w x roi_h
    left = max(0, (nw - roi_w) // 2)
    top = max(0, (nh - roi_h) // 2)
    return resized.crop((left, top, left + roi_w, top + roi_h))


def layout_sticker_on_roi(
    scene_size: tuple[int, int],
    sticker_rgba: Image.Image,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    fit: str,
    anchor: str,
) -> tuple[Image.Image, int, int]:
    """
    Resize/placement only: returns (sticker_rgba_ready, ox, oy) in scene pixel coords.
    Same geometry as paste_sticker; use for alpha-shaped inpaint masks.
    """
    w, h = scene_size
    x0 = max(0, min(w, x0))
    y0 = max(0, min(h, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("degenerate ROI")

    roi_w, roi_h = x1 - x0, y1 - y0
    st = resize_sticker_for_roi(sticker_rgba, roi_w, roi_h, fit)
    sw, sh = st.size

    if fit == "contain":
        if anchor == "center":
            ox = x0 + (roi_w - sw) // 2
            oy = y0 + (roi_h - sh) // 2
        else:
            ox, oy = x0, y0
    else:
        ox, oy = x0, y0

    return st, ox, oy


def paste_sticker(
    scene: Image.Image,
    sticker_rgba: Image.Image,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    fit: str,
    anchor: str,
) -> Image.Image:
    w, h = scene.size
    st, ox, oy = layout_sticker_on_roi(
        (w, h), sticker_rgba, x0, y0, x1, y1, fit, anchor
    )

    scene_rgba = scene.convert("RGBA")
    layer = Image.new("RGBA", scene_rgba.size, (0, 0, 0, 0))
    layer.paste(st, (ox, oy), st)

    out = Image.alpha_composite(scene_rgba, layer)
    return out.convert("RGB")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alpha-blend RGBA sticker into scene ROI.")
    p.add_argument("--scene", type=str, required=True, help="Background RGB/RGBA image.")
    p.add_argument("--sticker", type=str, required=True, help="Sticker path (RGBA preferred).")
    p.add_argument(
        "--box-json",
        type=str,
        default=None,
        help="make_box_mask --write-meta JSON (rect_pixels_xyxy or xywh).",
    )
    p.add_argument(
        "--rect-pixels",
        type=_parse_xywh,
        metavar="X,Y,W,H",
        default=None,
        help="ROI as top-left + size; overrides --box-json if both set.",
    )
    p.add_argument(
        "--fit",
        choices=("contain", "cover", "stretch"),
        default="contain",
        help="How sticker is scaled into ROI (default: contain, letterboxed with transparency).",
    )
    p.add_argument(
        "--anchor",
        choices=("center", "topleft"),
        default="center",
        help="For fit=contain, where to place the sticker inside ROI.",
    )
    p.add_argument("--out", type=str, required=True, help="Output RGB PNG path.")
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

    if args.rect_pixels is not None:
        x, y, rw, rh = args.rect_pixels
        x0, y0, x1, y1 = x, y, x + rw, y + rh
    elif args.box_json:
        box_path = Path(args.box_json).expanduser().resolve()
        if not box_path.is_file():
            print(f"box-json not found: {box_path}", file=sys.stderr)
            return 1
        x0, y0, x1, y1 = rect_from_box_json(box_path)
    else:
        print("Provide --rect-pixels or --box-json.", file=sys.stderr)
        return 1

    scene = Image.open(scene_path).convert("RGB")
    sticker = load_layer_rgba(str(sticker_path))

    out_img = paste_sticker(scene, sticker, x0, y0, x1, y1, args.fit, args.anchor)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    print(
        f"Saved: {out_path} (ROI [{x0},{y0})–[{x1},{y1}), fit={args.fit}, anchor={args.anchor})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
