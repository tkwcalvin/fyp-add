#!/usr/bin/env python3
"""
End-to-end workflow: scene image + text object prompt + insertion position/size
  -> SD1.5 generates object image -> SAM segments foreground
  -> object-bbox-aware scaling and pasting into scene
  -> (default) SDXL inpaint blends lighting and seams inside the mask.

Dependencies (bundled by default, no need to point to collage/FYP paths):
  - workflow_backend/: sd_sam_pipeline.py, paste_sticker_bbox_roi.py, paste_sticker_roi.py,
    test_sd_generation.py, utils/
  - sdxl_inpaint/stable_diffusion.py: sticker-fuse (still requires local SDXL snapshot and CUDA)
  If WORKFLOW_BACKEND / FYP_ROOT are set, external paths override bundled copies.

Usage:
  # Explicit pixel rectangle (top-left + width/height)
  python run_workflow.py scene.png \"a dog\" 256,0,128,256
  # Slot size + corner placement + margin
  python run_workflow.py scene.png \"a cat\" --slot-place bottom-left --slot-size 128,256 --margin 24
  # Arbitrary pixel location: slot top-left + width/height
  python run_workflow.py scene.png \"a mug\" --slot-at 120,80 --slot-size 160,200
  python run_workflow.py scene.png \"a mug\" 100,100,200,200 --no-inpaint   # One-shot X,Y,W,H form
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image


def _default_backend() -> Path:
    env = os.environ.get("WORKFLOW_BACKEND")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent
    bundled = (here / "workflow_backend").resolve()
    if (bundled / "sd_sam_pipeline.py").is_file():
        return bundled
    return (here.parent / "collage-diffusion-ui" / "backend").resolve()


def _default_fyp_root() -> Path:
    env = os.environ.get("FYP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent
    bundled = (here / "sdxl_inpaint").resolve()
    if (bundled / "stable_diffusion.py").is_file():
        return bundled
    return (here.parent / "FYP_JAYHF1").resolve()


def _slug(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", s.strip(), flags=re.UNICODE)
    s = s.strip("_") or "run"
    return s[:max_len]


def _parse_xywh(s: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("insert_rect requires four integers: x,y,w,h")
    return tuple(int(float(p)) for p in parts)


def _parse_wh(s: str) -> tuple[int, int]:
    parts = [p.strip() for p in s.replace(" ", "").split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--slot-size requires two integers: W,H")
    return int(float(parts[0])), int(float(parts[1]))


def _parse_xy(s: str) -> tuple[int, int]:
    parts = [p.strip() for p in s.replace(" ", "").split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--slot-at requires two integers: X,Y (slot top-left)")
    return int(float(parts[0])), int(float(parts[1]))


def slot_at_to_xywh(
    scene_w: int,
    scene_h: int,
    x0: int,
    y0: int,
    slot_w: int,
    slot_h: int,
) -> tuple[int, int, int, int]:
    """Compute insertion rectangle from slot top-left + desired size, clamped to scene bounds."""
    if slot_w < 1 or slot_h < 1:
        raise ValueError("slot size must be positive")
    x0 = max(0, min(scene_w - 1, int(x0)))
    y0 = max(0, min(scene_h - 1, int(y0)))
    w = min(slot_w, scene_w - x0)
    h = min(slot_h, scene_h - y0)
    if w < 1 or h < 1:
        raise ValueError("slot does not fit in scene from this origin")
    return x0, y0, w, h


def slot_place_to_xywh(
    scene_w: int,
    scene_h: int,
    place: str,
    margin: int,
    slot_w: int,
    slot_h: int,
) -> tuple[int, int, int, int]:
    """Compute slot top-left + size from placement; margin is spacing from corresponding scene edges."""
    m = max(0, int(margin))
    if slot_w < 1 or slot_h < 1:
        raise ValueError("slot size must be positive")
    if place == "top-left":
        x0, y0 = m, m
    elif place == "top-right":
        x0, y0 = scene_w - slot_w - m, m
    elif place == "bottom-left":
        x0, y0 = m, scene_h - slot_h - m
    elif place == "bottom-right":
        x0, y0 = scene_w - slot_w - m, scene_h - slot_h - m
    elif place == "center":
        inner_w = scene_w - 2 * m
        inner_h = scene_h - 2 * m
        x0 = m + max(0, (inner_w - slot_w) // 2)
        y0 = m + max(0, (inner_h - slot_h) // 2)
    else:
        raise ValueError(place)
    x0 = max(0, min(scene_w - 1, x0))
    y0 = max(0, min(scene_h - 1, y0))
    w = min(slot_w, scene_w - x0)
    h = min(slot_h, scene_h - y0)
    if w < 1 or h < 1:
        raise ValueError("slot does not fit in scene with this margin/size")
    return x0, y0, w, h


def main() -> int:
    p = argparse.ArgumentParser(description="Scene + text object → SD+SAM → paste at rect.")
    p.add_argument("scene", type=Path, help="Scene image path (RGB).")
    p.add_argument("prompt", type=str, help="Object prompt to generate (sd_sam default sticker suffix is appended).")
    p.add_argument(
        "insert_rect",
        nargs="?",
        default=None,
        metavar="X,Y,W,H",
        help="Optional: slot top-left + width/height. If omitted, use --slot-place/--slot-at with --slot-size.",
    )
    p.add_argument("--run-name", type=str, default=None, help="Run subdirectory name (default: timestamp + prompt slug).")
    p.add_argument(
        "--backend",
        type=Path,
        default=None,
        help="Directory of SD+SAM+paste scripts (default bundled workflow_backend/ or WORKFLOW_BACKEND).",
    )
    p.add_argument("--runs-root", type=Path, default=None, help="Root directory for all runs (default runs/ under this repo).")
    p.add_argument("--sam-preset", type=str, default="animal", choices=("animal", "object", "center"))
    p.add_argument("--sam-mode", type=str, default="point", choices=("point", "auto"))
    p.add_argument("--sticker-size", type=int, default=512, help="Square sticker side length from sd_sam; 0 means crop only.")
    p.add_argument("--sd-steps", type=int, default=28)
    p.add_argument("--sd-seed", type=int, default=-1, help="-1 for random.")
    p.add_argument("--fit", type=str, default="contain", choices=("contain", "cover", "stretch"))
    p.add_argument(
        "--sticker-anchor",
        "--anchor",
        type=str,
        default="center",
        choices=("center", "topleft"),
        dest="sticker_anchor",
        help="Sticker alignment inside slot when fit=contain (not scene corner). --anchor is a compatibility alias.",
    )
    p.add_argument(
        "--slot-place",
        type=str,
        default=None,
        choices=("top-left", "top-right", "bottom-left", "bottom-right", "center"),
        help="Slot placement in scene; with --slot-size you can omit insert_rect.",
    )
    p.add_argument(
        "--slot-at",
        type=_parse_xy,
        default=None,
        metavar="X,Y",
        help="Slot top-left pixel coordinates; use with --slot-size for free placement (mutually exclusive with --slot-place).",
    )
    p.add_argument(
        "--slot-size",
        type=_parse_wh,
        default=None,
        metavar="W,H",
        help="Slot width/height in scene pixels (axis-aligned paste box); with contain fit, smaller slots make smaller visible objects.",
    )
    p.add_argument(
        "--margin",
        type=int,
        default=20,
        help="Margin from scene edges (effective for corner slot-place; for center it acts as safe padding).",
    )
    p.add_argument("--skip-sd-sam", action="store_true", help="Skip generation; requires existing --object-sticker.")
    p.add_argument(
        "--object-sticker",
        type=Path,
        default=None,
        help="RGBA sticker path used when SD+SAM is skipped.",
    )
    p.add_argument(
        "--inpaint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run SDXL inpaint after paste to blend scene_with_object + inpaint_mask (enabled by default; disable with --no-inpaint).",
    )
    p.add_argument(
        "--fyp-root",
        type=Path,
        default=None,
        help="Directory containing stable_diffusion.py (default bundled sdxl_inpaint/, sibling FYP_JAYHF1, or FYP_ROOT).",
    )
    p.add_argument(
        "--inpaint-steps",
        type=int,
        default=None,
        help="Diffusion steps passed to sticker-fuse (if unset, stable_diffusion.py default is used).",
    )
    p.add_argument(
        "--inpaint-strength",
        type=float,
        default=None,
        help="Strength passed to sticker-fuse (if unset, script default is used).",
    )
    p.add_argument(
        "--mask-mode",
        type=str,
        default="contour",
        choices=("contour", "hybrid", "alpha", "rect"),
        help="contour=expand sticker silhouette then inpaint (default); hybrid/alpha/rect are alternatives.",
    )
    p.add_argument(
        "--mask-contour-expand",
        type=int,
        default=55,
        help="contour: expansion pixels; larger values include more surrounding scene for seam/light blending.",
    )
    p.add_argument(
        "--mask-feather",
        type=float,
        default=16.0,
        help="Feather radius for contour/hybrid/alpha.",
    )
    p.add_argument(
        "--mask-dilate",
        type=int,
        default=0,
        help="Alpha dilation pixels (hybrid/alpha).",
    )
    p.add_argument(
        "--mask-slot-strength",
        type=float,
        default=0.42,
        help="hybrid: base strength inside insertion slot (0-1); larger values blend more of the full slot.",
    )
    p.add_argument(
        "--mask-slot-feather",
        type=float,
        default=18.0,
        help="hybrid: feather radius for insertion rectangle slot.",
    )
    p.add_argument(
        "--preserve-subject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lower inpaint weight on the pasted subject to reduce edge blur/chipping (enabled by default).",
    )
    p.add_argument("--mask-edge-pixels", type=int, default=6)
    p.add_argument("--mask-atten-interior", type=float, default=0.36)
    p.add_argument("--mask-atten-edge", type=float, default=0.55)
    p.add_argument("--mask-subject-floor", type=float, default=22.0)
    args = p.parse_args()

    backend = (args.backend or _default_backend()).resolve()
    if not (backend / "sd_sam_pipeline.py").is_file():
        print(f"Invalid backend (missing sd_sam_pipeline.py): {backend}", file=sys.stderr)
        return 1

    wf_root = Path(__file__).resolve().parent
    runs_root = (args.runs_root or (wf_root / "runs")).expanduser().resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = args.run_name or _slug(args.prompt)
    run_dir = runs_root / f"{ts}_{slug}"
    gen_dir = run_dir / "object_sd_sam"
    comp_dir = run_dir / "composite"
    gen_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        print(f"Scene image not found: {scene}", file=sys.stderr)
        return 1

    has_rect = bool(args.insert_rect)
    has_place = args.slot_place is not None
    has_at = args.slot_at is not None
    n_modes = sum((has_rect, has_place, has_at))
    if n_modes > 1:
        print(
            "Use exactly one insertion mode: insert_rect, or --slot-place + --slot-size, or --slot-at + --slot-size",
            file=sys.stderr,
        )
        return 1
    if n_modes == 0:
        print(
            "Please provide an insertion mode: insert_rect X,Y,W,H, or --slot-place + --slot-size, or --slot-at + --slot-size",
            file=sys.stderr,
        )
        return 1
    if has_place and args.slot_size is None:
        print("--slot-place requires --slot-size W,H", file=sys.stderr)
        return 1
    if has_at and args.slot_size is None:
        print("--slot-at requires --slot-size W,H", file=sys.stderr)
        return 1

    scene_w, scene_h = Image.open(scene).convert("RGB").size
    if args.insert_rect:
        ix, iy, iw, ih = _parse_xywh(args.insert_rect)
        slot_meta: dict = {"mode": "rect", "insert_rect_xywh": [ix, iy, iw, ih]}
    elif has_at:
        assert args.slot_at is not None and args.slot_size is not None
        sx, sy = args.slot_at
        sw, sh = args.slot_size
        try:
            ix, iy, iw, ih = slot_at_to_xywh(scene_w, scene_h, sx, sy, sw, sh)
        except ValueError as e:
            print(f"Invalid insertion slot: {e}", file=sys.stderr)
            return 1
        slot_meta = {
            "mode": "slot_at",
            "slot_xy": [sx, sy],
            "slot_size_wh": [sw, sh],
            "insert_rect_xywh": [ix, iy, iw, ih],
        }
        if (ix, iy, iw, ih) != (sx, sy, sw, sh):
            print(
                f"Computed insert_rect from --slot-at {sx},{sy} --slot-size {sw},{sh} "
                f"(clamped to scene) → {ix},{iy},{iw},{ih}",
                file=sys.stderr,
            )
        else:
            print(
                f"Computed insert_rect from --slot-at {sx},{sy} --slot-size {sw},{sh} → {ix},{iy},{iw},{ih}",
                file=sys.stderr,
            )
    else:
        assert args.slot_place is not None and args.slot_size is not None
        sw, sh = args.slot_size
        try:
            ix, iy, iw, ih = slot_place_to_xywh(scene_w, scene_h, args.slot_place, args.margin, sw, sh)
        except ValueError as e:
            print(f"Invalid insertion slot: {e}", file=sys.stderr)
            return 1
        slot_meta = {
            "mode": "slot_place",
            "slot_place": args.slot_place,
            "slot_size_wh": [sw, sh],
            "margin": args.margin,
            "insert_rect_xywh": [ix, iy, iw, ih],
        }
        print(
            f"Computed insert_rect from --slot-place {args.slot_place} --slot-size {sw},{sh} "
            f"--margin {args.margin} → {ix},{iy},{iw},{ih}",
            file=sys.stderr,
        )

    # Keep an input copy for reproducibility.
    shutil.copy2(scene, run_dir / "scene_input.png")

    sticker_path: Path | None = None
    if args.skip_sd_sam:
        if not args.object_sticker or not args.object_sticker.is_file():
            print("--skip-sd-sam requires a valid --object-sticker", file=sys.stderr)
            return 1
        sticker_path = args.object_sticker.expanduser().resolve()
        shutil.copy2(sticker_path, gen_dir / "object_sticker_manual.png")
        sticker_path = gen_dir / "object_sticker_manual.png"
    else:
        cmd = [
            sys.executable,
            str(backend / "sd_sam_pipeline.py"),
            "--prompt",
            args.prompt,
            "--out-dir",
            str(gen_dir),
            "--sam-preset",
            args.sam_preset,
            "--sam-mode",
            args.sam_mode,
            "--steps",
            str(args.sd_steps),
            "--width",
            "512",
            "--height",
            "512",
            "--sticker-size",
            str(max(0, int(args.sticker_size))),
        ]
        if args.sd_seed >= 0:
            cmd.extend(["--seed", str(args.sd_seed)])

        print("Running SD + SAM …", file=sys.stderr)
        r = subprocess.run(cmd, cwd=str(backend))
        if r.returncode != 0:
            return r.returncode

        if args.sticker_size > 0:
            cand = gen_dir / f"object_sticker_{args.sticker_size}.png"
            if cand.is_file():
                sticker_path = cand
        if sticker_path is None:
            crop = gen_dir / "object_crop_rgba.png"
            if crop.is_file():
                sticker_path = crop
            else:
                print(f"Could not find object_sticker_*.png or object_crop_rgba.png in: {gen_dir}", file=sys.stderr)
                return 1

    assert sticker_path is not None
    paste_cmd = [
        sys.executable,
        str(backend / "paste_sticker_bbox_roi.py"),
        "--scene",
        str(scene),
        "--sticker",
        str(sticker_path),
        "--insert-rect",
        f"{ix},{iy},{iw},{ih}",
        "--out",
        str(comp_dir / "scene_with_object.png"),
        "--out-mask",
        str(comp_dir / "inpaint_mask.png"),
        "--write-meta",
        str(comp_dir / "insert_meta.json"),
        "--preview-bbox",
        str(comp_dir / "sticker_object_bbox_preview.png"),
        "--fit",
        args.fit,
        "--anchor",
        args.sticker_anchor,
        "--mask-mode",
        args.mask_mode,
        "--mask-contour-expand",
        str(args.mask_contour_expand),
        "--mask-feather",
        str(args.mask_feather),
        "--mask-dilate",
        str(args.mask_dilate),
        "--mask-slot-strength",
        str(args.mask_slot_strength),
        "--mask-slot-feather",
        str(args.mask_slot_feather),
    ]
    paste_cmd.append("--preserve-subject" if args.preserve_subject else "--no-preserve-subject")
    paste_cmd.extend(
        [
            "--mask-edge-pixels",
            str(args.mask_edge_pixels),
            "--mask-atten-interior",
            str(args.mask_atten_interior),
            "--mask-atten-edge",
            str(args.mask_atten_edge),
            "--mask-subject-floor",
            str(args.mask_subject_floor),
        ]
    )
    print("Pasting sticker into scene …", file=sys.stderr)
    r = subprocess.run(paste_cmd, cwd=str(backend))
    if r.returncode != 0:
        return r.returncode

    manifest: dict = {
        "scene_original": str(scene),
        "scene_copy": str(run_dir / "scene_input.png"),
        "object_prompt": args.prompt,
        "insert_rect_xywh": [ix, iy, iw, ih],
        "slot_placement": slot_meta,
        "sticker_anchor_in_slot": args.sticker_anchor,
        "backend": str(backend),
        "sam_preset": args.sam_preset,
        "sam_mode": args.sam_mode,
        "sticker_source": str(sticker_path),
        "composite": str(comp_dir / "scene_with_object.png"),
        "inpaint_mask": str(comp_dir / "inpaint_mask.png"),
        "inpaint_mask_mode": args.mask_mode,
        "inpaint_mask_contour_expand": args.mask_contour_expand,
        "inpaint_mask_feather": args.mask_feather,
        "inpaint_mask_dilate": args.mask_dilate,
        "inpaint_mask_slot_strength": args.mask_slot_strength,
        "inpaint_mask_slot_feather": args.mask_slot_feather,
        "inpaint_preserve_subject": args.preserve_subject,
        "inpaint_mask_atten_interior": args.mask_atten_interior,
        "inpaint_mask_atten_edge": args.mask_atten_edge,
        "insert_meta": str(comp_dir / "insert_meta.json"),
        "inpaint_fused": None,
        "inpaint_intermediate_dir": None,
    }

    if args.inpaint:
        fyp_root = (args.fyp_root or _default_fyp_root()).resolve()
        sd_script = fyp_root / "stable_diffusion.py"
        if not sd_script.is_file():
            print(
                f"{sd_script} not found, cannot run SDXL inpaint. Set FYP_ROOT/--fyp-root or use --no-inpaint.",
                file=sys.stderr,
            )
            return 1
        fuse_out = comp_dir / "scene_inpaint_fused.png"
        inp_cmd = [
            sys.executable,
            str(sd_script),
            "sticker-fuse",
            "--composite",
            str(comp_dir / "scene_with_object.png"),
            "--mask",
            str(comp_dir / "inpaint_mask.png"),
            "--out",
            str(fuse_out),
        ]
        if args.inpaint_steps is not None:
            inp_cmd.extend(["--steps", str(args.inpaint_steps)])
        if args.inpaint_strength is not None:
            inp_cmd.extend(["--strength", str(args.inpaint_strength)])

        print("Running SDXL inpaint (sticker-fuse) …", file=sys.stderr)
        r = subprocess.run(inp_cmd, cwd=str(fyp_root))
        if r.returncode != 0:
            return r.returncode
        manifest["inpaint_fused"] = str(fuse_out)
        mid = comp_dir / f"{fuse_out.stem}_intermediate"
        if mid.is_dir():
            manifest["inpaint_intermediate_dir"] = str(mid)
    else:
        manifest["inpaint_skipped"] = True

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nRun directory: {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
