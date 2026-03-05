"""
Centralized configuration for PCB/Wafer Defect Detection.
Override any values in the Colab notebook before importing other modules.
"""

import torch

import os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Dataset ────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_DIR, 'DeepPCB/PCBData/')
YOLO_DATA_DIR = os.path.join(PROJECT_DIR, 'pcb_yolo_dataset')  # converted YOLO-format output

# ─── Classes ────────────────────────────────────────────────────────────────
NUM_CLASSES = 7  # 6 defect types + 1 background
CLASS_MAP = {
    1: 'open',
    2: 'short',
    3: 'mousebite',
    4: 'spur',
    5: 'copper',
    6: 'pin-hole',
}

# ─── Training Hyperparameters ───────────────────────────────────────────────
BATCH_SIZE = 4
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 2.0
IMG_SIZE = 640          # used by YOLO / RT-DETR / ViT models
TEST_SPLIT = 0.2        # fraction of trainval used for validation
RANDOM_SEED = 42

# ─── Evaluation ─────────────────────────────────────────────────────────────
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5

# ─── Checkpoints ────────────────────────────────────────────────────────────
import os as _os
BEST_MODEL_PATH = _os.environ.get('BEST_MODEL_PATH', 'best_model.pth')

# ─── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Pretrained Weights ────────────────────────────────────────────────────
USE_PRETRAINED = True   # Whether to load pretrained backbones (ImageNet, COCO, etc.)
