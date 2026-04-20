#!/usr/bin/env python3
"""
Quick text-to-image test using the same base model as serve_diffusion.py:
  runwayml/stable-diffusion-v1-5

Run from this directory (so local imports work if you extend this later):
  cd /path/to/collage-diffusion-ui/backend
  python test_sd_generation.py

Examples:
  python test_sd_generation.py --prompt "a red sports car, studio lighting"
  python test_sd_generation.py --steps 35 --seed 42 --out test_outputs/my_test.png

If downloads are slow or blocked, set a mirror before running:
  export HF_ENDPOINT=https://hf-mirror.com

Use a local diffusers snapshot (full path to the hash folder under snapshots/):
  export SD_MODEL_ID=/path/to/snapshots/<hash>
  python test_sd_generation.py

Or point at the Hugging Face *hub repo cache dir* (the folder named models--runwayml--stable-diffusion-v1-5);
the script picks the snapshot under snapshots/ that contains model_index.json:
  export SD_HUB_REPO_DIR=/root/autodl-tmp/cache/hub/models--runwayml--stable-diffusion-v1-5
  python test_sd_generation.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


# Matches DiffusionDeployment in serve_diffusion.py (StableDiffusionControlNetPipeline base).
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Tried when --model and SD_MODEL_ID are unset (first existing path wins).
_DEFAULT_HUB_REPO_CANDIDATES = (
    Path("/root/autodl-tmp/cache/hub/models--runwayml--stable-diffusion-v1-5"),
    Path.home() / ".cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5",
)


def resolve_diffusers_snapshot(hub_repo_dir: Path) -> Path | None:
    """hub_repo_dir = .../models--runwayml--stable-diffusion-v1-5 → .../snapshots/<hash>."""
    snapshots = hub_repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(p for p in snapshots.iterdir() if p.is_dir())
    for snap in candidates:
        if (snap / "model_index.json").is_file():
            return snap
    return None


def pick_model_id(args_model: str | None) -> str:
    if args_model:
        return args_model
    env_id = os.environ.get("SD_MODEL_ID")
    if env_id:
        return env_id
    hub_env = os.environ.get("SD_HUB_REPO_DIR")
    if hub_env:
        snap = resolve_diffusers_snapshot(Path(hub_env).expanduser().resolve())
        if snap is not None:
            return str(snap)
        print(f"Warning: SD_HUB_REPO_DIR has no valid snapshot: {hub_env}", file=sys.stderr)
    for cand in _DEFAULT_HUB_REPO_CANDIDATES:
        snap = resolve_diffusers_snapshot(cand)
        if snap is not None:
            return str(snap)
    return DEFAULT_MODEL_ID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SD 1.5 txt2img smoke test (same base as collage backend).")
    p.add_argument(
        "--prompt",
        type=str,
        default="a photo of a golden retriever on grass, natural light, sharp focus",
        help="Text prompt.",
    )
    p.add_argument("--negative-prompt", type=str, default=None, help="Optional negative prompt.")
    p.add_argument("--steps", type=int, default=28, help="Inference steps.")
    p.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale.")
    p.add_argument("--width", type=int, default=512, help="Image width (SD1.5 default-friendly).")
    p.add_argument("--height", type=int, default=512, help="Image height.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (set -1 for random).")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG file (overrides --out-dir / --out-name).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Default output directory relative to this script's folder (default: outputs).",
    )
    p.add_argument(
        "--out-name",
        type=str,
        default=None,
        help="Filename under --out-dir when --out is not set (default: sd15_test_<seed>.png).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model id or local path (else SD_MODEL_ID env, else runway SD1.5).",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (very slow; for sanity checks only).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    backend_dir = Path(__file__).resolve().parent
    outputs_root = (backend_dir / args.out_dir).resolve()
    outputs_root.mkdir(parents=True, exist_ok=True)

    model_id = pick_model_id(args.model)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda" and not args.cpu:
        print("Warning: CUDA not available; using CPU.", file=sys.stderr)

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading model: {model_id}")
    print(f"Device: {device}, dtype: {dtype}")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    generator = None
    if args.seed >= 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        name = args.out_name or f"sd15_test_seed{args.seed}.png"
        out_path = outputs_root / name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating: {args.prompt[:80]!r}...")
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        generator=generator,
    )
    image = result.images[0]
    image.save(out_path)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
