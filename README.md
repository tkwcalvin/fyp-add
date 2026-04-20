# Object Insertion Workflow — 基于 SD1.5 + SAM + SDXL Inpaint 的物体插入流水线

> **输入**：一张场景图、一段要插入的物品文字描述、以及插入槽的位置与大小
> **输出**：把新物品自然地合成到场景中（含 alpha 直接贴 + SDXL inpaint 光影融合两版结果）

本仓库是一个**端到端工作流**的最小实现，把下面 4 步串成 **一条命令**：

1. **Stable Diffusion 1.5** 按文本描述生成白底物品图（`workflow_backend/sd_sam_pipeline.py`）。
2. **SAM**（Segment Anything）分割物品前景，导出 RGBA 贴纸。
3. **paste_sticker_bbox_roi.py**：按物体 alpha 的紧包围盒，将物品缩放到用户指定的插入矩形并合成，同时输出 **inpaint 蒙版** 与元信息 JSON。
4. **SDXL Inpaint（sticker-fuse）**：仅在蒙版内重绘，融合光影与接缝，得到最终成图。

---

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [模型权重下载（必读）](#模型权重下载必读)
- [仓库结构](#仓库结构)
- [用法](#用法)
- [输出文件说明](#输出文件说明)
- [插入槽大小怎么定？](#插入槽大小怎么定)
- [蒙版模式与参数调节](#蒙版模式与参数调节)
- [单独运行 SDXL Inpaint](#单独运行-sdxl-inpaint)
- [常见问题排查](#常见问题排查)

---

## 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（其他平台理论可用，未测试） |
| GPU | NVIDIA，**显存 ≥ 12 GB**（SDXL inpaint 默认 `--size 512` 实测约 9 GB；`--size 1024` 约 14 GB） |
| CUDA | 11.8 / 12.1 + 对应 PyTorch |
| Python | 3.10 或 3.11 |
| 磁盘 | ≈ 20 GB（SD1.5 ≈ 4 GB + SDXL inpaint ≈ 7 GB + SAM ViT‑H ≈ 2.6 GB + CLIP 等） |

---

## 快速开始

```bash
# 1) 克隆仓库
git clone https://github.com/<your-org>/object-insert-workflow.git
cd object-insert-workflow

# 2) 建议新建独立环境
python3 -m venv .venv && source .venv/bin/activate
# 或：conda create -n obj-insert python=3.10 -y && conda activate obj-insert

# 3) 安装 PyTorch（按你的 CUDA 版本选对应 wheel，官方命令：https://pytorch.org/）
# 示例（CUDA 12.1）：
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# 4) 安装其他依赖
pip install -r requirements.txt

# 5) 下载模型权重（见下一节），并设置环境变量
cp config.example.env config.env
# 按注释编辑 config.env 中的 SAM_CHECKPOINT / SD_HUB_REPO_DIR / SDXL_MODEL_DIR
source config.env

# 6) 跑一个最小示例（准备一张场景图 inputs/scene.png）
python3 run_workflow.py inputs/scene.png "a golden retriever dog" \
  --slot-place bottom-left --slot-size 256,256 --margin 24

# 结果在 runs/<时间戳>_<描述缩写>/composite/scene_inpaint_fused.png
```

---

## 模型权重下载（必读）

**本仓库不包含任何模型权重**，需要自行下载并通过环境变量指向。下面三组权重是**必须**的，CLIP 仅在开启 `--sam-clip-rerank` 时需要。

建议都放到仓库下的 `weights/` 目录（已被 `.gitignore` 忽略），或任意其他位置再用环境变量指向。

### 1. SAM（Segment Anything，ViT‑H 检查点）

```bash
mkdir -p weights
wget -O weights/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
export SAM_CHECKPOINT="$PWD/weights/sam_vit_h_4b8939.pth"
```

### 2. Stable Diffusion 1.5（文生图）

推荐使用 Hugging Face `snapshot_download` 缓存一份本地副本：

```bash
# 方式 A：直接用 huggingface-cli 下载到本机缓存
huggingface-cli download runwayml/stable-diffusion-v1-5 \
  --local-dir weights/sd15 --local-dir-use-symlinks False

export SD_MODEL_ID="$PWD/weights/sd15"
```

或使用 diffusers 的 hub 缓存目录（即 `models--runwayml--stable-diffusion-v1-5` 这一级），脚本会自动挑选 `snapshots/<hash>`：

```bash
export SD_HUB_REPO_DIR="$HOME/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5"
```

> 国内网络慢可先 `export HF_ENDPOINT=https://hf-mirror.com`。

### 3. SDXL Inpaint（`diffusers` 兼容快照）

默认使用 `diffusers.AutoPipelineForInpainting`，请下载一个本地 SDXL inpaint 快照，目录里需含 `model_index.json`。例如官方的 `stabilityai/stable-diffusion-xl-base-1.0` 加 inpaint 插件，或使用任一 SDXL‑Inpaint 合集模型：

```bash
huggingface-cli download diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
  --local-dir weights/sdxl_inpaint --local-dir-use-symlinks False

export SDXL_MODEL_DIR="$PWD/weights/sdxl_inpaint"
# 或等价的别名：MODEL_DIR
```

### 4.（可选）CLIP ViT-Base / Patch32

仅在 `--sam-clip-rerank` 时需要，用于在 SAM 多候选里按「图文相似度」挑选。默认直接从 HF 拉 `openai/clip-vit-base-patch32`：

```bash
huggingface-cli download openai/clip-vit-base-patch32 \
  --local-dir weights/clip --local-dir-use-symlinks False
export SD_SAM_CLIP_MODEL_ID="$PWD/weights/clip"
```

---

## 仓库结构

```
object-insert-workflow/
├── run_workflow.py              # 主入口（Python）
├── run_workflow.sh              # 同上，bash 包装
├── fuse_inpaint_example.sh      # 仅对已有 composite+mask 再跑一次 SDXL inpaint
│
├── workflow_backend/            # SD1.5 + SAM + 粘贴相关脚本
│   ├── sd_sam_pipeline.py       # 文生图 → SAM 分割 → RGBA sticker
│   ├── paste_sticker_bbox_roi.py# 按物体外接矩形缩放并粘贴到场景
│   ├── paste_sticker_roi.py     # 基础粘贴/ROI 布局
│   ├── test_sd_generation.py    # SD1.5 smoke test（会被上面脚本 import）
│   └── utils/
│       └── load_layer_image.py
│
├── sdxl_inpaint/                # SDXL inpaint 融合器
│   └── stable_diffusion.py      # sticker-fuse 子命令
│
├── scripts/                     # 可直接跑的示例 / 维护脚本
│   ├── run_cat_bottom_left_quarter.sh
│   ├── paste_only_with_sticker.sh
│   └── sync_bundled_from_fyp.sh
│
├── inputs/                      # 示例场景图 / 你自己的输入
├── runs/                        # 每次运行生成一个子目录（被 .gitignore 忽略）
├── weights/                     # 存放 SAM / SD / SDXL 本地权重（被 .gitignore 忽略）
│
├── requirements.txt
├── config.example.env           # 复制为 config.env 后编辑 → source
├── .gitignore
└── README.md
```

---

## 用法

### 三种插入几何（三选一）

| 方式 | 语法 | 适用场景 |
|------|------|----------|
| ① 像素矩形（位置参数） | `X,Y,W,H` | 已经手算好左上角和宽高 |
| ② 场景角位 + 槽大小 | `--slot-place {top-left,top-right,bottom-left,bottom-right,center} --slot-size W,H --margin M` | 想「放左下角，离边 24 像素」不想自己算坐标 |
| ③ 任意点 + 槽大小 | `--slot-at X,Y --slot-size W,H` | 指定左上角坐标但懒得写宽高拼一起 |

### 基础示例

```bash
# 显式像素矩形（prompt 含空格请加引号）
python3 run_workflow.py inputs/scene.png "a golden retriever dog" 256,0,128,256

# 只给槽大小 + 场景角位 + 边距
python3 run_workflow.py inputs/scene.png "a cat" \
  --slot-place bottom-left --slot-size 128,256 --margin 24

# 任意点 + 宽高
python3 run_workflow.py inputs/scene.png "a mug" \
  --slot-at 120,80 --slot-size 160,200

# 指定 run 子目录名、家具类可换 SAM preset
python3 run_workflow.py inputs/scene.png "a wooden chair" 120,80,200,300 \
  --run-name chair01 --sam-preset object

# 只想要 alpha 合成、不跑 SDXL inpaint（更快 / 无 SDXL 环境）
python3 run_workflow.py inputs/scene.png "a dog" 256,0,128,256 --no-inpaint

# 自定义 inpaint 强度 / 步数
python3 run_workflow.py inputs/scene.png "a dog" 256,0,128,256 \
  --inpaint-strength 0.55 --inpaint-steps 30

# 跳过 SD+SAM，直接用一张已有的 RGBA 贴图
python3 run_workflow.py inputs/scene.png "unused" 256,0,128,256 \
  --skip-sd-sam --object-sticker /path/to/sticker.png
```

### 常用参数速查

| 参数 | 默认 | 含义 |
|------|------|------|
| `--sam-preset` | `animal` | SAM 提示点策略：`animal` / `object` / `center` |
| `--sam-mode` | `point` | `point` 点提示 / `auto` 自动分割 |
| `--sticker-size` | `512` | SAM 输出方形贴纸的边长；`0` = 只保留 crop 版本 |
| `--sd-steps` | `28` | SD1.5 推理步数 |
| `--sd-seed` | `-1` | `-1` 随机；固定种子以复现 |
| `--fit` | `contain` | 贴纸进入插入槽的缩放方式：`contain` / `cover` / `stretch` |
| `--sticker-anchor` | `center` | `contain` 时贴纸在槽内的对齐：`center` / `topleft` |
| `--mask-mode` | `contour` | `contour`（默认，更自然）/ `hybrid` / `alpha` / `rect` |
| `--inpaint` / `--no-inpaint` | 开 | 是否运行 SDXL 融合 |

完整参数请运行 `python3 run_workflow.py --help`。

---

## 输出文件说明

每次运行会在 `runs/<YYYYMMDD_HHMMSS>_<slug>/` 下生成：

| 路径 | 说明 |
|------|------|
| `scene_input.png` | 场景图副本（便于日后复现） |
| `object_sd_sam/generated.png` | SD1.5 文生图原图 |
| `object_sd_sam/mask.png` | SAM 输出的二值蒙版 |
| `object_sd_sam/object_rgba.png` | 全幅 RGBA（alpha = 蒙版） |
| `object_sd_sam/object_crop_rgba.png` | 紧包围 RGBA 贴纸 |
| `object_sd_sam/object_sticker_512.png` | 方形 letterbox 贴纸（默认 512） |
| `composite/scene_with_object.png` | **alpha 合成结果**（粘贴后、inpaint 前） |
| `composite/inpaint_mask.png` | 送给 SDXL 的灰度蒙版（亮=重绘，暗=保留） |
| `composite/sticker_object_bbox_preview.png` | 框可视化预览（调试用） |
| `composite/insert_meta.json` | 插入矩形坐标（含 `rect_pixels_xyxy`） |
| `composite/scene_inpaint_fused.png` | **SDXL 融合后的最终图**（`--no-inpaint` 时无此项） |
| `composite/scene_inpaint_fused_intermediate/` | inpaint 中间图 / 蒙版统计 / 输入输出对比 |
| `manifest.json` | 本次运行总览（全部路径 + 超参） |

---

## 插入槽大小怎么定？

- `--slot-size W,H`（或位置参数里的 `w,h`）是**场景里的轴对齐粘贴框像素尺寸**，**不是**贴纸分辨率。
- 贴纸本身一般是 512×512，默认 `--fit contain` 会把它**整体缩放进这个框**；框越小，物体在场景里显得越小。
- 经验：前景道具取**场景宽的 15%–35%** 比较自然；先看 `composite/scene_with_object.png` 和 `sticker_object_bbox_preview.png` 预览，再调 W,H 或切 `--fit cover`（铺满槽，可能裁切）。
- 想让物体离边缘留白：角位用 `--margin`；任意点用 `--slot-at` 自己留空。

---

## 蒙版模式与参数调节

默认 `--mask-mode contour` 会把贴纸 alpha 在场景上**外扩若干像素**（`--mask-contour-expand`，默认 55；`--mask-feather` 默认 16），重绘区域 ≈「物体 + 外圈一点真实背景」，避免硬矩形和主体变形。

| 模式 | 行为 | 何时选 |
|------|------|--------|
| `contour`（默认） | 贴纸轮廓外扩 + 羽化 | 想要**自然边缘融合**，保留贴纸本体 |
| `hybrid` | 轮廓 + 槽内底强度 | 想让**整个槽内**一起融（光影/阴影重建更强） |
| `alpha` | 直接用贴纸 alpha 当蒙版 | 严格只重绘物体本身 |
| `rect` | 插入矩形整个白 | 最粗暴，容易出现矩形接缝 |

可配合 `--preserve-subject`（默认开）降低物体本体上的 inpaint 权重，减轻边缘模糊。

---

## 单独运行 SDXL Inpaint

如果你已经有一张 composite 和一张 mask，想再手动融一版，不用重跑 SD + SAM：

```bash
./fuse_inpaint_example.sh runs/某次/composite/scene_with_object.png \
                         runs/某次/composite/inpaint_mask.png \
                         runs/某次/composite/manual_fused.png
```

等价命令：

```bash
python3 sdxl_inpaint/stable_diffusion.py sticker-fuse \
  --composite path/to/composite.png \
  --mask      path/to/mask.png \
  --out       path/to/out.png \
  --size 512 --steps 35 --strength 0.62
```

---

## 常见问题排查

- **`CUDA is required.`** → 本流水线强依赖 GPU，请确认 `nvidia-smi` 可见、且装的是 GPU 版 PyTorch。
- **`未找到 sd_sam_pipeline.py` / `stable_diffusion.py`** → 说明 `--backend` 或 `--fyp-root` 指歪了，默认应为 `workflow_backend/` 与 `sdxl_inpaint/`，可手动 `export WORKFLOW_BACKEND=…` / `FYP_ROOT=…`。
- **SAM 输出的物体包进了地板 / 背景** → 换 `--sam-preset object`（四角作为负点）或加 `--sam-clip-rerank`。
- **融合后比贴纸更糊** → 调低 `--inpaint-strength`（默认 0.62 → 0.45）；或把 `--size` 保持在 512（不要放大到 1024）。
- **HF 下载失败** → `export HF_ENDPOINT=https://hf-mirror.com`；`HF_HUB_ENABLE_HF_TRANSFER=0` 可规避部分 xet 报错。
- **显存不足** → 先 `--no-inpaint` 验证 SD+SAM+粘贴链路，再单独跑 `fuse_inpaint_example.sh` 并考虑 `--size 512`。

---

## License / 鸣谢

- Segment Anything © Meta AI — Apache 2.0
- Stable Diffusion 1.5 / SDXL Inpaint © StabilityAI / RunwayML — 请遵守各自许可
- 本仓库代码以 MIT 发布（或按你项目实际 License 填写）
