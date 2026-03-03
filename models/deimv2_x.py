"""
models/deimv2_x.py
==================
Model wrapper for DEIMv2-X (DINOv3-S+ backbone) on the DeepPCB dataset.

Mirrors deimv2_l.py but uses the larger DINOv3-S+ (vits16plus) backbone
and the corresponding X-variant config.
"""

import os
import sys
from pathlib import Path


# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent.resolve()
DEIMV2_DIR  = PROJECT_DIR / "DEIMv2"
CONFIG_PATH = DEIMV2_DIR / "configs" / "deimv2" / "deimv2_dinov3_x_deeppcb.yml"
CKPTS_DIR   = DEIMV2_DIR / "ckpts"
BACKBONE_SRC = (
    PROJECT_DIR
    / "dinov3"
    / "pretrained_weight"
    / "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
)
BACKBONE_DST = CKPTS_DIR / "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"

MODEL_NAME    = "deimv2_x"
VARIANT_LABEL = "DEIMv2-X (DINOv3-S+ backbone)"


def _ensure_ckpts():
    """Symlink / copy backbone weights into DEIMv2/ckpts/ if not already there."""
    CKPTS_DIR.mkdir(parents=True, exist_ok=True)
    if BACKBONE_DST.exists():
        return
    if BACKBONE_SRC.exists():
        try:
            os.symlink(BACKBONE_SRC.resolve(), BACKBONE_DST)
            print(f"[{MODEL_NAME}] Symlinked backbone: {BACKBONE_DST}")
        except (OSError, NotImplementedError):
            import shutil
            shutil.copy2(BACKBONE_SRC, BACKBONE_DST)
            print(f"[{MODEL_NAME}] Copied backbone: {BACKBONE_DST}")
    else:
        print(
            f"[{MODEL_NAME}] WARNING: DINOv3-S+ backbone not found at:\n"
            f"  {BACKBONE_SRC}\n"
            f"Please download it from the DINOv3 repository and place it at "
            f"that path (or at {BACKBONE_DST}).",
            file=sys.stderr,
        )


class DEIMv2Wrapper:
    """
    Thin wrapper exposing the same interface as other project models.

    train_model.py detects this class (via ._is_deimv2 flag) and calls
    subprocess-based training instead of the standard PyTorch training loop.
    """

    def __init__(self, config_path: Path, model_name: str, label: str):
        self.config_path = config_path
        self.model_name  = model_name
        self.label       = label
        self._is_deimv2  = True   # sentinel used by train_model.py

    def __repr__(self):
        return f"<{self.__class__.__name__} config={self.config_path.name}>"


def build(num_classes: int = 6, pretrained: bool = True, **kwargs):
    """
    Registry entry-point.

    Returns a DEIMv2Wrapper. Training is handled in train_model.py via
    subprocess (torchrun DEIMv2/train.py -c <config>).
    """
    print(f"[{MODEL_NAME}] Building {VARIANT_LABEL}")
    print(f"[{MODEL_NAME}] Config  : {CONFIG_PATH}")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"DEIMv2-X config not found: {CONFIG_PATH}\n"
            "Make sure the DEIMv2/ directory is present."
        )

    _ensure_ckpts()

    return DEIMv2Wrapper(
        config_path=CONFIG_PATH,
        model_name=MODEL_NAME,
        label=VARIANT_LABEL,
    )
