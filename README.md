# Object Insertion Workflow — An object insertion pipeline based on SD1.5 + SAM + SDXL Inpaint

> **Input**: a scene image, a text prompt describing the object to insert, and the insertion slot position/size  
> **Output**: a natural composite result with two variants (direct alpha paste + SDXL inpaint lighting/seam fusion)

This repository is a minimal **end-to-end workflow** that chains the following 4 steps into **one command**:

1. **Stable Diffusion 1.5** generates an object image from text (with `workflow_backend/sd_sam_pipeline.py`).
2. **SAM** (Segment Anything) segments the foreground object and exports an RGBA sticker.
3. **paste_sticker_bbox_roi.py** scales the object using the alpha tight bounding box into a user-defined insertion rectangle, then outputs both an **inpaint mask** and metadata JSON.
4. **SDXL Inpaint (sticker-fuse)** repaints only inside the mask to blend lighting and seams for the final image.

---

## Table of Contents

- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Model Weights Download (Required)](#model-weights-download-required)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Output Files](#output-files)
- [How to choose insertion slot size?](#how-to-choose-insertion-slot-size)
- [Mask Modes and Parameter Tuning](#mask-modes-and-parameter-tuning)
- [Run SDXL Inpaint Separately](#run-sdxl-inpaint-separately)
- [Troubleshooting](#troubleshooting)

---

## Requirements

| Item | Requirement |
|------|-------------|
| OS | Linux (other platforms may work in theory, but are untested) |
| GPU | NVIDIA, **VRAM >= 12 GB** (SDXL inpaint uses about 9 GB with `--size 512`, and ~14 GB with `--size 1024`) |
| CUDA | 11.8 / 12.1 with matching PyTorch builds |
| Python | 3.10 or 3.11 |
| Disk | ~20 GB (SD1.5 ~4 GB + SDXL inpaint ~7 GB + SAM ViT-H ~2.6 GB + CLIP, etc.) |

---

## Quick Start

```bash
# 1) Clone repository
git clone https://github.com/<your-org>/object-insert-workflow.git
cd object-insert-workflow

# 2) Create an isolated environment (recommended)
python3 -m venv .venv && source .venv/bin/activate
# or: conda create -n obj-insert python=3.10 -y && conda activate obj-insert

# 3) Install PyTorch (choose wheel by your CUDA version, official guide: https://pytorch.org/)
# example (CUDA 12.1):
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# 4) Install remaining dependencies
pip install -r requirements.txt

# 5) Download model weights (see next section), then set env vars
cp config.example.env config.env
# edit SAM_CHECKPOINT / SD_HUB_REPO_DIR / SDXL_MODEL_DIR in config.env
source config.env

# 6) Run a minimal example (prepare a scene image at inputs/scene.png)
python3 run_workflow.py inputs/scene.png "a golden retriever dog" \
  --slot-place bottom-left --slot-size 256,256 --margin 24

# output: runs/<timestamp>_<slug>/composite/scene_inpaint_fused.png
```

---

## Model Weights Download (Required)

**This repository does not include model weights.** You must download them yourself and point to them via environment variables. The three groups below are **required**. CLIP is only needed when enabling `--sam-clip-rerank`.

Recommended: store all weights under `weights/` (already ignored by `.gitignore`), or place them anywhere else and configure env vars accordingly.

### 1. SAM (Segment Anything, ViT-H checkpoint)

```bash
mkdir -p weights
wget -O weights/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
export SAM_CHECKPOINT="$PWD/weights/sam_vit_h_4b8939.pth"
```

### 2. Stable Diffusion 1.5 (text-to-image)

Recommended: use Hugging Face `snapshot_download` or local snapshot caching:

```bash
# option A: download directly with huggingface-cli
huggingface-cli download runwayml/stable-diffusion-v1-5 \
  --local-dir weights/sd15 --local-dir-use-symlinks False

export SD_MODEL_ID="$PWD/weights/sd15"
```

Or use the diffusers hub cache directory (the `models--runwayml--stable-diffusion-v1-5` level). The script will auto-select `snapshots/<hash>`:

```bash
export SD_HUB_REPO_DIR="$HOME/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5"
```

> If Hugging Face is slow in your region, try `export HF_ENDPOINT=https://hf-mirror.com`.

### 3. SDXL Inpaint (`diffusers`-compatible snapshot)

By default this workflow uses `diffusers.AutoPipelineForInpainting`. Download a local SDXL inpaint snapshot whose directory contains `model_index.json`. Example: official `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`:

```bash
huggingface-cli download diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
  --local-dir weights/sdxl_inpaint --local-dir-use-symlinks False

export SDXL_MODEL_DIR="$PWD/weights/sdxl_inpaint"
# equivalent alias: MODEL_DIR
```

### 4. (Optional) CLIP ViT-Base / Patch32

Only required with `--sam-clip-rerank`, used to pick among SAM candidates by image-text similarity. Default model is `openai/clip-vit-base-patch32`:

```bash
huggingface-cli download openai/clip-vit-base-patch32 \
  --local-dir weights/clip --local-dir-use-symlinks False
export SD_SAM_CLIP_MODEL_ID="$PWD/weights/clip"
```

---

## Repository Structure

```
object-insert-workflow/
├── run_workflow.py              # main entrypoint (Python)
├── run_workflow.sh              # bash wrapper for the same pipeline
├── fuse_inpaint_example.sh      # rerun SDXL inpaint on existing composite+mask
│
├── workflow_backend/            # SD1.5 + SAM + sticker paste scripts
│   ├── sd_sam_pipeline.py       # text-to-image -> SAM segmentation -> RGBA sticker
│   ├── paste_sticker_bbox_roi.py# scale/paste by object bounding box into scene
│   ├── paste_sticker_roi.py     # basic paste/ROI layout helpers
│   ├── test_sd_generation.py    # SD1.5 smoke test (imported by pipeline)
│   └── utils/
│       └── load_layer_image.py
│
├── sdxl_inpaint/                # SDXL inpaint fuser
│   └── stable_diffusion.py      # sticker-fuse subcommand
│
├── scripts/                     # runnable examples / utility scripts
│   ├── run_cat_bottom_left_quarter.sh
│   ├── paste_only_with_sticker.sh
│   └── sync_bundled_from_fyp.sh
│
├── inputs/                      # example scene images / your own inputs
├── runs/                        # one subfolder per run (ignored by .gitignore)
├── weights/                     # local SAM/SD/SDXL weights (ignored by .gitignore)
│
├── requirements.txt
├── config.example.env           # copy to config.env, edit, then source
├── .gitignore
└── README.md
```

---

## Usage

### Three insertion geometry modes (pick one)

| Mode | Syntax | Best for |
|------|--------|----------|
| ① Pixel rectangle (positional arg) | `X,Y,W,H` | You already know top-left and width/height |
| ② Scene corner + slot size | `--slot-place {top-left,top-right,bottom-left,bottom-right,center} --slot-size W,H --margin M` | "Put it bottom-left, 24px from edges" without manual coordinate math |
| ③ Arbitrary point + slot size | `--slot-at X,Y --slot-size W,H` | You want top-left coordinate + size with explicit flags |

### Basic examples

```bash
# explicit pixel rectangle (quote prompt if it contains spaces)
python3 run_workflow.py inputs/scene.png "a golden retriever dog" 256,0,128,256

# slot size + scene corner + margin
python3 run_workflow.py inputs/scene.png "a cat" \
  --slot-place bottom-left --slot-size 128,256 --margin 24

# arbitrary top-left point + width/height
python3 run_workflow.py inputs/scene.png "a mug" \
  --slot-at 120,80 --slot-size 160,200

# custom run folder name; for furniture-like objects use a different SAM preset
python3 run_workflow.py inputs/scene.png "a wooden chair" 120,80,200,300 \
  --run-name chair01 --sam-preset object

# alpha composite only, skip SDXL inpaint (faster / no SDXL setup needed)
python3 run_workflow.py inputs/scene.png "a dog" 256,0,128,256 --no-inpaint

# custom inpaint strength / steps
python3 run_workflow.py inputs/scene.png "a dog" 256,0,128,256 \
  --inpaint-strength 0.55 --inpaint-steps 30

# skip SD+SAM, use an existing RGBA sticker directly
python3 run_workflow.py inputs/scene.png "unused" 256,0,128,256 \
  --skip-sd-sam --object-sticker /path/to/sticker.png
```

### Common options quick reference

| Option | Default | Meaning |
|--------|---------|---------|
| `--sam-preset` | `animal` | SAM prompt point strategy: `animal` / `object` / `center` |
| `--sam-mode` | `point` | `point` prompt mode or `auto` segmentation |
| `--sticker-size` | `512` | Side length of square sticker output from SAM; `0` keeps crop only |
| `--sd-steps` | `28` | SD1.5 inference steps |
| `--sd-seed` | `-1` | `-1` random; set fixed seed for reproducibility |
| `--fit` | `contain` | Sticker resize behavior into slot: `contain` / `cover` / `stretch` |
| `--sticker-anchor` | `center` | Sticker alignment inside slot in `contain` mode: `center` / `topleft` |
| `--mask-mode` | `contour` | `contour` (default, more natural) / `hybrid` / `alpha` / `rect` |
| `--inpaint` / `--no-inpaint` | on | Whether to run SDXL fusion |

For full arguments, run `python3 run_workflow.py --help`.

---

## Output Files

Each run creates a folder under `runs/<YYYYMMDD_HHMMSS>_<slug>/`:

| Path | Description |
|------|-------------|
| `scene_input.png` | Copy of input scene image (for reproducibility) |
| `object_sd_sam/generated.png` | SD1.5 generated object image |
| `object_sd_sam/mask.png` | Binary mask from SAM |
| `object_sd_sam/object_rgba.png` | Full-frame RGBA image (alpha = SAM mask) |
| `object_sd_sam/object_crop_rgba.png` | Tight-bbox RGBA sticker |
| `object_sd_sam/object_sticker_512.png` | Square letterboxed sticker (default 512) |
| `composite/scene_with_object.png` | **alpha composite result** (after paste, before inpaint) |
| `composite/inpaint_mask.png` | Grayscale mask for SDXL (bright = repaint, dark = keep) |
| `composite/sticker_object_bbox_preview.png` | Bounding box visualization preview (for debugging) |
| `composite/insert_meta.json` | Insertion rectangle metadata (includes `rect_pixels_xyxy`) |
| `composite/scene_inpaint_fused.png` | **final SDXL-fused image** (missing when `--no-inpaint`) |
| `composite/scene_inpaint_fused_intermediate/` | Inpaint intermediates / mask stats / IO comparisons |
| `manifest.json` | Run manifest (all paths + hyperparameters) |

---

## How to choose insertion slot size?

- `--slot-size W,H` (or positional `w,h`) is the **axis-aligned paste box size in scene pixels**, **not** sticker resolution.
- The sticker is typically 512x512; with default `--fit contain`, it is fully scaled into that box. A smaller box means a visually smaller inserted object.
- Rule of thumb: foreground props often look natural at **15%-35% of scene width**. Check `composite/scene_with_object.png` and `sticker_object_bbox_preview.png`, then tune `W,H` or switch to `--fit cover` (fills slot, may crop).
- To keep margins from scene edges: use `--margin` with corner placement, or manually choose `--slot-at`.

---

## Mask Modes and Parameter Tuning

Default `--mask-mode contour` expands the sticker alpha by a few pixels in scene space (`--mask-contour-expand`, default 55; `--mask-feather`, default 16). Repaint region is approximately "object + a small surrounding background", which helps avoid hard rectangular seams and subject deformation.

| Mode | Behavior | When to use |
|------|----------|-------------|
| `contour` (default) | Expand + feather around sticker contour | For **natural edge blending** while preserving the object body |
| `hybrid` | Contour + slot interior base strength | When you want **whole-slot blending** (stronger lighting/shadow reconstruction) |
| `alpha` | Use sticker alpha directly as mask | Strictly repaint the object only |
| `rect` | Full white insertion rectangle | Most aggressive; can produce rectangular seams |

You can combine this with `--preserve-subject` (enabled by default) to reduce inpaint strength on the object itself and limit edge blur.

---

## Run SDXL Inpaint Separately

If you already have a composite and a mask and want to rerun fusion manually, you can skip SD + SAM:

```bash
./fuse_inpaint_example.sh runs/<some_run>/composite/scene_with_object.png \
                         runs/<some_run>/composite/inpaint_mask.png \
                         runs/<some_run>/composite/manual_fused.png
```

Equivalent command:

```bash
python3 sdxl_inpaint/stable_diffusion.py sticker-fuse \
  --composite path/to/composite.png \
  --mask      path/to/mask.png \
  --out       path/to/out.png \
  --size 512 --steps 35 --strength 0.62
```

---

## Troubleshooting

- **`CUDA is required.`** -> This workflow requires GPU. Confirm `nvidia-smi` works and that you installed GPU-enabled PyTorch.
- **`sd_sam_pipeline.py` / `stable_diffusion.py` not found** -> `--backend` or `--fyp-root` is likely misconfigured. Defaults should be `workflow_backend/` and `sdxl_inpaint/`. You can also set `WORKFLOW_BACKEND` / `FYP_ROOT` manually.
- **SAM includes floor/background with the object** -> Try `--sam-preset object` (corners as negative points), or enable `--sam-clip-rerank`.
- **Fused output looks blurrier than sticker** -> Lower `--inpaint-strength` (e.g. from 0.62 to 0.45), or keep `--size 512` instead of scaling to 1024.
- **HF download fails** -> Try `export HF_ENDPOINT=https://hf-mirror.com`; set `HF_HUB_ENABLE_HF_TRANSFER=0` to avoid some xet-related issues.
- **Out of VRAM** -> First validate SD+SAM+paste with `--no-inpaint`, then run `fuse_inpaint_example.sh` separately and keep `--size 512`.

---

## License / Acknowledgements

- Segment Anything (Meta AI) — Apache 2.0
- Stable Diffusion 1.5 / SDXL Inpaint (StabilityAI / RunwayML) — follow each model's license terms
