# PCB Wafer Defect Detection

A research pipeline for PCB (Printed Circuit Board) defect detection,
comparing multiple deep-learning model architectures on the
[DeepPCB](https://github.com/tangsanli5201/DeepPCB) dataset.

---

## Supported Models

| ID | Model | Backbone | Training engine |
|----|-------|----------|-----------------|
| `faster_rcnn` | Faster R-CNN | ResNet-50-FPN v2 | PyTorch |
| `vit_det` | ViT-Det | ViT-Base/16 + FPN | PyTorch |
| `sme_yolo` | SME-YOLO | YOLOv11n (CSPDarknet) | Ultralytics |
| `yolo26` | YOLO26 | YOLO26n | Ultralytics |
| `rt_detr` | RT-DETR-L | ResNet-based | Ultralytics |
| `deimv2_l` | **DEIMv2-L** | **DINOv3-S** | DEIMv2 / torchrun |

DEIMv2-L is a state-of-the-art real-time object detector that combines a
DEIM-style DETR decoder with a DINOv3 vision-foundation backbone
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
bash scripts/install_deps.sh
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

# Quick smoke-test (1 epoch)
python train_model.py --model deimv2_l --test_mode
```

### 4. Train all models sequentially

```bash
bash scripts/run_all_models.sh           # all models, full run
bash scripts/run_all_models.sh --test    # test mode (1 epoch each)
bash scripts/run_all_models.sh deimv2_l  # single model
```

### 5. Evaluate and compare

```bash
python eval_compare.py      # generate comparison table
python parse_logs.py        # parse training logs & plot curves
python inference_demo.py    # speed benchmark
```

---

## Code Flowcharts

### Overall Pipeline

```mermaid
flowchart LR
    A["DeepPCB Dataset<br/>(640×640 images)"] --> B["Dataset Conversion"]
    B --> C["Training<br/>train_model.py"]
    C --> D["Best Checkpoints<br/>outputs/&lt;model&gt;/"]
    D --> E["Evaluation<br/>eval_compare.py"]
    D --> F["Speed Benchmark<br/>inference_demo.py"]
    D --> G["Log Parsing<br/>parse_logs.py"]
    E --> H["Comparison Tables<br/>(CSV + JSON)"]
    F --> I["Latency / FPS<br/>Summary"]
    G --> J["Training Curves<br/>(PNG plots)"]

    style A fill:#4a9eff,color:#fff
    style C fill:#ff6b6b,color:#fff
    style E fill:#51cf66,color:#fff
    style F fill:#51cf66,color:#fff
    style G fill:#51cf66,color:#fff
```

### Training Pipeline  (`train_model.py`)

```mermaid
flowchart TD
    Start(["python train_model.py --model &lt;name&gt;"]) --> Parse["Parse CLI args<br/>(model, epochs, batch_size,<br/>test_mode, resume)"]
    Parse --> SetOutput["Set output dir<br/>outputs/&lt;model&gt;/"]
    SetOutput --> Config["Load config.py<br/>(DATA_DIR, hyperparams, DEVICE)"]
    Config --> Registry["create_model(name)<br/>models/__init__.py registry"]

    Registry --> Branch{Model type?}

    %% ── PyTorch branch ──
    Branch -->|"PyTorch<br/>(faster_rcnn, vit_det)"| DS_PT["dataset.py<br/>create_dataloaders()<br/>train / val / test splits"]
    DS_PT --> ResumePT{"--resume?"}
    ResumePT -->|Yes| LoadCkpt["Load last_ckpt / best .pth<br/>+ infer start_epoch"]
    ResumePT -->|No| TrainPT
    LoadCkpt --> TrainPT["training.py → train_model()<br/>• train_one_epoch() per epoch<br/>• validate() each epoch<br/>• cosine-annealing LR<br/>• best-model checkpointing"]
    TrainPT --> TestPT["Test-set evaluation<br/>• validate() → P, R, F1, mIoU<br/>• compute_map() → mAP50, mAP50-95"]
    TestPT --> SavePT["Save history JSON<br/>+ test_metrics JSON<br/>+ best_&lt;model&gt;.pth"]

    %% ── Ultralytics branch ──
    Branch -->|"Ultralytics<br/>(sme_yolo, yolo26, rt_detr)"| DS_UL["utils.py<br/>convert_deeppcb_to_yolo()<br/>→ YOLO-format dirs + data.yaml"]
    DS_UL --> ResumeUL{"--resume?"}
    ResumeUL -->|Yes| PatchPT["Patch last.pt<br/>(epochs + epoch marker)"]
    PatchPT --> TrainUL
    ResumeUL -->|No| TrainUL["model.train()<br/>(Ultralytics engine)<br/>→ runs/&lt;model&gt;/weights/"]
    TrainUL --> CopyBest["Copy best.pt → best_&lt;model&gt;.pth"]
    CopyBest --> ValUL["model.val()"]

    %% ── DEIMv2 branch ──
    Branch -->|"DEIMv2<br/>(deimv2_l)"| DS_COCO["scripts/convert_deeppcb_to_coco.py<br/>→ COCO JSON + symlinked images"]
    DS_COCO --> ResumeDEIM{"--resume?"}
    ResumeDEIM -->|Yes| ResDEIM["--resume=last.pth"]
    ResumeDEIM -->|No| TrainDEIM
    ResDEIM --> TrainDEIM["torchrun DEIMv2/train.py<br/>-c config.yml --use-amp"]
    TrainDEIM --> SaveDEIM["Checkpoints in outputs/deimv2_l/<br/>(last.pth, best_stg2.pth)"]

    SavePT --> Done(["Training Complete"])
    ValUL --> Done
    SaveDEIM --> Done

    style Start fill:#4a9eff,color:#fff
    style Branch fill:#ffd43b,color:#333
    style Done fill:#51cf66,color:#fff
```

### Evaluation & Reporting Pipeline

```mermaid
flowchart TD
    subgraph eval_compare ["eval_compare.py"]
        EC_Start["Load MODEL_META<br/>(6 models)"] --> EC_Loop["For each model:"]
        EC_Loop --> EC_Metrics{"Model type?"}
        EC_Metrics -->|PyTorch| EC_PT["get_pytorch_metrics()<br/>→ load test_metrics JSON<br/>run_pytorch_test_eval() if missing"]
        EC_Metrics -->|Ultralytics| EC_UL["get_ultralytics_metrics()<br/>→ parse results.csv<br/>run_ultralytics_test_eval() if needed"]
        EC_Metrics -->|DEIMv2| EC_DM["get_deimv2_metrics()<br/>→ parse COCO eval from log.txt<br/>run_deimv2_test_eval() if needed"]
        EC_PT --> EC_Complexity["get_model_complexity()<br/>→ params, GFLOPs"]
        EC_UL --> EC_Complexity
        EC_DM --> EC_ComplexDEIM["get_deimv2_complexity()<br/>→ params, GFLOPs via thop"]
        EC_Complexity --> EC_Speed["load_inference_summary()<br/>→ FPS, latency"]
        EC_ComplexDEIM --> EC_Speed
        EC_Speed --> EC_Tables["Print tables:<br/>• Architecture & Complexity<br/>• Detection Accuracy<br/>• Inference Speed"]
        EC_Tables --> EC_Save["Save CSV + JSON<br/>results/comparison_table.*"]
    end

    subgraph parse_logs ["parse_logs.py"]
        PL_Start["For each model:"] --> PL_Load{"Model type?"}
        PL_Load -->|PyTorch| PL_PT["load_pytorch_history()<br/>→ parse history JSON"]
        PL_Load -->|Ultralytics| PL_UL["load_ultralytics_results()<br/>→ parse results.csv"]
        PL_Load -->|DEIMv2| PL_DM["load_deimv2_history()<br/>→ parse log.txt JSON lines"]
        PL_PT --> PL_Plot
        PL_UL --> PL_Plot
        PL_DM --> PL_Plot
        PL_Plot["Plotting:<br/>• plot_individual_model()<br/>  (loss + val metrics per model)<br/>• plot_comparison()<br/>  (overlay all models)"]
        PL_Plot --> PL_Save["Save PNGs to<br/>results/plots/"]
    end

    subgraph infer ["inference_demo.py"]
        INF_Start["Resolve best weights<br/>for each model"] --> INF_Run{"Model type?"}
        INF_Run -->|PyTorch| INF_PT["run_pytorch_inference()<br/>→ GPU-synced timing"]
        INF_Run -->|Ultralytics| INF_UL["run_ultralytics_inference()<br/>→ per-image predict()"]
        INF_Run -->|DEIMv2| INF_DM["run_deimv2_inference()<br/>→ CUDA event timing"]
        INF_PT --> INF_Summary
        INF_UL --> INF_Summary
        INF_DM --> INF_Summary
        INF_Summary["Compute stats:<br/>• mean/median/p95 latency<br/>• FPS<br/>Save inference_summary.json<br/>+ annotated images"]
    end

    %% Invisible links to force vertical stacking of subgraphs
    EC_Save ~~~ PL_Start
    PL_Save ~~~ INF_Start

    style eval_compare fill:#e8f5e9,stroke:#43a047
    style parse_logs fill:#e3f2fd,stroke:#1e88e5
    style infer fill:#fff3e0,stroke:#fb8c00
```

---

## Directory Structure

```
PCB_wafer_defect_detection/
├── models/                          ← model definitions
│   ├── __init__.py                  ← model registry (6 models)
│   ├── faster_rcnn.py / vit_det.py  ← PyTorch-loop models
│   ├── sme_yolo.py / yolo26.py / rt_detr.py  ← Ultralytics
│   └── deimv2_l.py                  ← DEIMv2-L wrapper
├── scripts/                         ← shell scripts & utilities
│   ├── run_all_models.sh            ← sequential training orchestrator
│   ├── run_resume_50ep.sh           ← resume training to 50 epochs
│   ├── install_deps.sh              ← install dependencies
│   └── convert_deeppcb_to_coco.py   ← dataset conversion
├── train_model.py                   ← unified training entry-point
├── eval_compare.py                  ← evaluation & comparison tables
├── inference_demo.py                ← speed benchmarks
├── parse_logs.py                    ← log parsing & plots
├── config.py / dataset.py / training.py / evaluation.py / utils.py
├── REPORT.md                        ← final results report
├── DEIMv2/                          ← external repo (not in git)
├── dinov3/                          ← external repo (not in git)
├── DeepPCB/                         ← dataset (not in git)
├── outputs/                         ← training outputs (not in git)
├── results/                         ← evaluation results (not in git)
└── _archive/                        ← old/unused files (not in git)
```

---

