"""
SDXL Inpaint (using local ``/root/autodl-tmp/diffusion`` snapshots with
StableDiffusionXLInpaintPipeline).

- ``demo``: original full-image inpaint generation / example inpaint.
- ``sticker-fuse``: inpaint only inside the ROI mask. Pixels inside the mask are
  re-denoised by diffusion (not just edge feathering). If strength is too high or
  the resolution mismatches the pasted sticker, output can look blurrier than the
  original RGBA sticker. Default prompt is tuned for a dog sticker and tries to
  preserve dog appearance.
  Default ``--size 512`` matches common collage pipelines and reduces upscale blur
  (set ``--size 1024`` if needed).

Environment variables: ``SDXL_MODEL_DIR`` or ``MODEL_DIR`` should point to a
directory containing ``model_index.json`` (same default logic as upstream).

Example:
  python stable_diffusion.py sticker-fuse \\
    --composite /root/fyp/collage-diffusion-ui/backend/outputs/my_scene_with_sticker.png \\
    --mask /root/fyp/collage-diffusion-ui/backend/outputs/my_scene_roi_mask.png \\
    --out outputs/my_scene_inpaint_fused.png

Mask semantics follow diffusers: **brighter pixels (255) are repainted more**,
darker pixels are preserved more.

By default, ``{output_name}_intermediate/`` is written next to the final image
(disable with ``--no-intermediates``). It includes model input image/mask, red overlay
preview, ROI diagnostics, side-by-side input/output, etc., useful when diagnosing
whether blending issues come from mask shape or denoise strength.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from PIL import Image

_REPO = Path(__file__).resolve().parent


def _mask_stats(mask_l: Image.Image) -> tuple[int, int, float, float]:
    """min, max, mean, fraction of pixels >= 128 (inpaint region)."""
    a = np.asarray(mask_l, dtype=np.int32)
    flat = a.ravel()
    n = flat.size
    ge = float(np.sum(flat >= 128)) / max(n, 1)
    return int(flat.min()), int(flat.max()), float(flat.mean()), ge


def _overlay_inpaint_region(rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    base = rgb.convert("RGBA")
    h, w = rgb.size[1], rgb.size[0]
    tint = Image.new("RGBA", (w, h), (255, 40, 40, 90))
    alpha = mask_l.point(lambda p: min(255, int(p * 0.35)))
    tint.putalpha(alpha)
    return Image.alpha_composite(base, tint).convert("RGB")


def save_sticker_fuse_intermediates(
    dest: Path,
    composite_path: Path,
    mask_path: Path,
    comp_original: Image.Image,
    mask_aligned: Image.Image,
    comp_model: Image.Image,
    mask_model: Image.Image,
    result: Image.Image,
    *,
    size: int,
    strength: float,
    steps: int,
    guidance: float,
    prompt: str,
    negative_prompt: str,
    seed: int,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    mn, mx, mean, frac = _mask_stats(mask_model)

    meta = dest / "00_meta.txt"
    meta.write_text(
        "\n".join(
            [
                "SDXL sticker-fuse intermediates (exact tensors sent to inpaint are 03/04).",
                "",
                f"composite_file: {composite_path}",
                f"mask_file: {mask_path}",
                f"original_composite_size: {comp_original.size}",
                f"aligned_mask_size: {mask_aligned.size}",
                f"model_input_size: {comp_model.size} (square --size {size})",
                "",
                f"mask_model_L: min={mn} max={mx} mean={mean:.2f} frac_pixels_ge_128={frac:.4f}",
                "  (diffusers: bright = repaint; dark = keep)",
                "",
                f"strength={strength} steps={steps} guidance={guidance} seed={seed}",
                "",
                "Why it can look blurrier than the sticker: inpaint re-denoises the masked latent;",
                "the model is not doing edge-aware sharpening. Lower --strength and match --size",
                "to the pasted image (512) to keep more of the original dog pixels.",
                "",
                "prompt:",
                prompt,
                "",
                "negative_prompt:",
                negative_prompt,
            ]
        ),
        encoding="utf-8",
    )

    comp_original.save(dest / "01_composite_original.png")
    mask_aligned.save(dest / "02_mask_aligned_to_composite.png")
    comp_model.save(dest / "03_composite_model_input.png")
    mask_model.save(dest / "04_mask_model_input.png")
    _overlay_inpaint_region(comp_model, mask_model).save(dest / "05_overlay_inpaint_region_red.png")

    gray = Image.new("RGB", comp_model.size, (128, 128, 128))
    Image.composite(comp_model, gray, mask_model).save(dest / "06_roi_only_on_gray.png")
    inv = mask_model.point(lambda p: 255 - p)
    Image.composite(comp_model, gray, inv).save(dest / "07_outside_only_on_gray.png")

    inv_mask = mask_model.point(lambda p: 255 - p)
    inv_mask.save(dest / "08_mask_inverted_preview.png")

    w, h = comp_model.size
    dual = Image.new("RGB", (w * 2 + 16, h), (32, 32, 32))
    dual.paste(comp_model, (0, 0))
    dual.paste(result, (w + 16, 0))
    dual.save(dest / "09_side_by_side_input_vs_output.png")


def resolve_model_dir() -> str:
    for key in ("SDXL_MODEL_DIR", "MODEL_DIR"):
        raw = os.environ.get(key)
        if raw:
            p = Path(raw).expanduser().resolve()
            if (p / "model_index.json").is_file():
                return str(p)
    root = Path("/root/autodl-tmp")
    diffusion = root / "diffusion"
    if (diffusion / "model_index.json").is_file():
        return str(diffusion)
    return str(root)


def load_pipe(model_dir: str):
    return AutoPipelineForInpainting.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")


def run_demo(pipe, mode: str, w: int, h: int, steps: int, strength: float, seed: int) -> Path:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    prompt = "a dogs"

    if mode == "generate":
        image = Image.new("RGB", (w, h), (128, 128, 128))
        mask_image = Image.new("L", (w, h), 255)
    else:
        img_url = str(_REPO / "paint_by_example_assets/image_example_1.png")
        mask_url = str(_REPO / "paint_by_example_assets/mask_example_1.png")
        image = load_image(img_url).resize((w, h))
        mask_image = load_image(mask_url).resize((w, h))

    out_path = _REPO / "outputs" / "output1.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=8.0,
        num_inference_steps=steps,
        strength=strength,
        generator=generator,
    ).images[0]
    result.save(str(out_path))
    return out_path


def run_sticker_fuse(
    pipe,
    composite_path: Path,
    mask_path: Path,
    out_path: Path,
    prompt: str,
    negative_prompt: str,
    size: int,
    steps: int,
    strength: float,
    guidance: float,
    seed: int,
    intermediates_dir: Path | None,
) -> Path:
    comp_original = Image.open(composite_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != comp_original.size:
        mask = mask.resize(comp_original.size, Image.Resampling.NEAREST)
    mask_aligned = mask.copy()

    w = h = int(size)
    comp_model = comp_original
    mask_model = mask_aligned
    if comp_model.size != (w, h):
        comp_model = comp_original.resize((w, h), Image.Resampling.LANCZOS)
        mask_model = mask_aligned.resize((w, h), Image.Resampling.NEAREST)

    gen = None
    if seed >= 0:
        gen = torch.Generator(device="cuda").manual_seed(seed)

    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=comp_model,
        mask_image=mask_model,
        guidance_scale=guidance,
        num_inference_steps=steps,
        strength=strength,
        generator=gen,
    ).images[0]
    result.save(str(out_path))

    if intermediates_dir is not None:
        save_sticker_fuse_intermediates(
            intermediates_dir.resolve(),
            composite_path,
            mask_path,
            comp_original,
            mask_aligned,
            comp_model,
            mask_model,
            result,
            size=size,
            strength=strength,
            steps=steps,
            guidance=guidance,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
        )
        print(f"Intermediates: {intermediates_dir.resolve()}", file=sys.stderr)

    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SDXL inpaint: demo or sticker ROI fusion.")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="Original full-frame or asset inpaint smoke test.")
    d.add_argument(
        "--mode",
        choices=("generate", "inpaint"),
        default="generate",
        help='generate: gray canvas + full white mask; inpaint: paint_by_example assets.',
    )
    d.add_argument("--width", type=int, default=1024)
    d.add_argument("--height", type=int, default=1024)
    d.add_argument("--steps", type=int, default=20)
    d.add_argument("--strength", type=float, default=0.99)
    d.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("sticker-fuse", help="Inpaint only masked region on pasted composite.")
    s.add_argument(
        "--composite",
        type=str,
        default="/root/fyp/collage-diffusion-ui/backend/outputs/my_scene_with_sticker.png",
        help="RGB image with sticker already pasted.",
    )
    s.add_argument(
        "--mask",
        type=str,
        default="/root/fyp/collage-diffusion-ui/backend/outputs/my_scene_roi_mask.png",
        help="L mask: 255 = repaint (blend sticker), 0 = keep.",
    )
    s.add_argument(
        "--out",
        type=str,
        default=str(_REPO / "outputs" / "my_scene_inpaint_fused.png"),
    )
    s.add_argument(
        "--prompt",
        type=str,
        default=(
            "the same dog from the pasted cutout, preserve its appearance pose and markings, "
            "sharp detailed fur and eyes, crisp focus on the dog. "
            "photorealistic indoor scene, the dog naturally on the floor, "
            "matching room lighting and soft contact shadow, "
            "seamless blend with floor and walls, no visible sticker edge, high quality"
        ),
    )
    s.add_argument(
        "--negative-prompt",
        type=str,
        default=(
            "different dog, wrong breed, deformed dog, extra legs, duplicate animals, "
            "worst quality, low quality, blurry, soft focus, out of focus, painting, illustration, "
            "cutout, sticker edge, collage seam, cartoon, oversaturated"
        ),
    )
    s.add_argument(
        "--size",
        type=int,
        default=512,
        help="Square side sent to SDXL; default 512 matches typical collage to avoid upscale blur. Use 1024 if you want full SDXL resolution.",
    )
    s.add_argument("--steps", type=int, default=35)
    s.add_argument(
        "--strength",
        type=float,
        default=0.62,
        help="Masked-region denoise: lower preserves more of the pasted dog; higher changes lighting/blend more but risks blur.",
    )
    s.add_argument("--guidance", type=float, default=7.5)
    s.add_argument("--seed", type=int, default=42, help="-1 for random.")
    s.add_argument(
        "--no-intermediates",
        action="store_true",
        help="Do not write debug PNGs / meta next to the output.",
    )
    s.add_argument(
        "--intermediates-dir",
        type=str,
        default=None,
        help="Override folder for intermediates (default: {out_stem}_intermediate next to --out).",
    )
    return p


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        return 1

    # Keep old behavior: `python stable_diffusion.py` == `demo --mode generate`
    if len(sys.argv) <= 1:
        sys.argv.extend(["demo", "--mode", "generate"])

    parser = build_arg_parser()
    args = parser.parse_args()
    model_dir = resolve_model_dir()
    print(f"Model dir: {model_dir}", file=sys.stderr)
    pipe = load_pipe(model_dir)
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    if args.command == "demo":
        path = run_demo(
            pipe,
            args.mode,
            args.width,
            args.height,
            args.steps,
            args.strength,
            args.seed,
        )
        print(f"Saved: {path}", file=sys.stderr)
        return 0

    if args.command == "sticker-fuse":
        cp = Path(args.composite).expanduser().resolve()
        mp = Path(args.mask).expanduser().resolve()
        if not cp.is_file():
            print(f"composite not found: {cp}", file=sys.stderr)
            return 1
        if not mp.is_file():
            print(f"mask not found: {mp}", file=sys.stderr)
            return 1
        seed = args.seed
        out_p = Path(args.out).expanduser().resolve()
        if args.no_intermediates:
            inter: Path | None = None
        elif args.intermediates_dir:
            inter = Path(args.intermediates_dir).expanduser().resolve()
        else:
            inter = out_p.parent / f"{out_p.stem}_intermediate"

        path = run_sticker_fuse(
            pipe,
            cp,
            mp,
            out_p,
            args.prompt,
            args.negative_prompt,
            args.size,
            args.steps,
            args.strength,
            args.guidance,
            seed,
            inter,
        )
        print(f"Saved: {path}", file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
