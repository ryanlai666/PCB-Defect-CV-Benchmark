# PCB Wafer Defect Detection

A research pipeline for PCB (Printed Circuit Board) defect detection,
comparing multiple deep-learning model architectures on the
[DeepPCB](https://github.com/tangsanli5201/DeepPCB) dataset.

---

## Supported Models

| ID | Model | Backbone | Training engine |
|----|-------|----------|-----------------|
| `faster_rcnn` | Faster R-CNN | ResNet-50-FPN | PyTorch |
| `faster_rcnn_ft` | Faster R-CNN (fine-tuned) | ResNet-50-FPN | PyTorch |
| `vit_det` | ViT-Det | ViT-Base (timm) | PyTorch |
| `vit_mamba` | ViT-Mamba | MambaVision | PyTorch |
| `yolo26` | YOLO26 | — | Ultralytics |
| `sme_yolo` | SME-YOLO | — | Ultralytics |
| `rt_detr` | RT-DETR-L | ResNet | Ultralytics |
| `deimv2_l` | **DEIMv2-L** | **DINOv3-S** | DEIMv2 / torchrun |
| `deimv2_x` | **DEIMv2-X** | **DINOv3-S+** | DEIMv2 / torchrun |

DEIMv2-L and DEIMv2-X are state-of-the-art real-time object detectors that
combine a DEIM-style DETR decoder with DINOv3 vision-foundation backbones
(see [`DEIMv2/README.md`](DEIMv2/README.md)).

---

## Dataset — DeepPCB

Located at `DeepPCB/PCBData/`. Six defect classes:

| ID | Name |
|----|------|
| 1  | open |
| 2  | short |
| 3  | mousebite |
| 4  | spur |
| 5  | copper |
| 6  | pin-hole |

Images are 640 × 640 px. Annotations are per-image `.txt` files in
`x1,y1,x2,y2,type` format. See [`DeepPCB/README.md`](DeepPCB/README.md).

---

## External Dependencies (not tracked by git)

> [!IMPORTANT]
> The following directories must be **cloned / placed manually** —  
> they are excluded from this repository via `.gitignore`.

### `DEIMv2/`
Real-time object detection framework using DINOv3 features.

```bash
git clone https://github.com/Intellindust-AI-Lab/DEIMv2.git DEIMv2
cd DEIMv2
pip install -r requirements.txt
```

See [`DEIMv2/README.md`](DEIMv2/README.md) for full documentation,
model zoo, and deployment guides.

### `dinov3/`
Facebook Research's DINOv3 vision foundation model.

```bash
git clone https://github.com/facebookresearch/dinov3.git dinov3
```

Download the pretrained backbone weights and place them at:
```
dinov3/pretrained_weight/
├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth    # DINOv3-S  → used by DEIMv2-L
└── dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth # DINOv3-S+ → used by DEIMv2-X
```

The training scripts automatically symlink these into `DEIMv2/ckpts/`  
when you first run a DEIMv2 model.

---

## Quickstart

### 1. Install dependencies

```bash
conda activate pcb
bash install_deps.sh
```

For DEIMv2, additionally:

```bash
cd DEIMv2 && pip install -r requirements.txt && cd ..
```

### 2. (Optional) Pre-convert DeepPCB to COCO format

DEIMv2 models need COCO-format JSON annotations. This is done **automatically**
on the first training run, but you can also run it manually:

```bash
python scripts/convert_deeppcb_to_coco.py \
    --deeppcb_dir DeepPCB/PCBData \
    --output_dir  data/deeppcb_coco \
    --val_ratio   0.15 \
    --seed        42
```

### 3. Train a single model

```bash
# Standard PyTorch / Ultralytics models
python train_model.py --model vit_det --epochs 20

# DEIMv2-L (DINOv3-S backbone, ~32M params)
python train_model.py --model deimv2_l

# DEIMv2-X (DINOv3-S+ backbone, ~50M params)
python train_model.py --model deimv2_x

# Quick smoke-test (1 epoch)
python train_model.py --model deimv2_l --test_mode
```

### 4. Train all models sequentially

```bash
bash run_all_models.sh           # all models, full run
bash run_all_models.sh --test    # test mode (1 epoch each)
bash run_all_models.sh deimv2_l  # single model
```

### 5. Evaluate and compare

```bash
python eval_compare.py      # generate comparison table
python parse_logs.py        # parse training logs & plot curves
python inference_demo.py    # speed benchmark
```

---

## Directory Structure

```
PCB_wafer_defect_detection/
├── DEIMv2/                  ← external repo (not in git)
├── dinov3/                  ← external repo (not in git)
├── DeepPCB/                 ← dataset (not in git)
├── data/deeppcb_coco/       ← converted COCO JSONs (auto-generated, not in git)
├── models/
│   ├── __init__.py          ← model registry
│   ├── faster_rcnn.py
│   ├── vit_det.py
│   ├── deimv2_l.py          ← DEIMv2-L wrapper
│   └── deimv2_x.py          ← DEIMv2-X wrapper
├── scripts/
│   └── convert_deeppcb_to_coco.py
├── train_model.py           ← unified training entry-point
├── run_all_models.sh        ← sequential training orchestrator
├── eval_compare.py
├── inference_demo.py
├── parse_logs.py
└── config.py
```

---

## Citation

If you use DEIMv2 or DINOv3 in your work, please cite:

```bibtex
@article{huang2025deimv2,
  title={Real-Time Object Detection Meets DINOv3},
  author={Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
  journal={arXiv},
  year={2025}
}
```
