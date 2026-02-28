#!/usr/bin/env bash
# ==============================================================================
# install_deps.sh — Install all dependencies for PCB Defect Detection
#
# Usage:
#   conda activate pcb
#   bash install_deps.sh
# ==============================================================================

set -euo pipefail

echo "============================================================"
echo "  PCB Defect Detection — Dependency Installer"
echo "  Time:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python: $(which python)"
echo "  Conda:  ${CONDA_DEFAULT_ENV:-unknown}"
echo "============================================================"
echo ""

# ─── Helper ──────────────────────────────────────────────────────────────────
check_and_install() {
    local pkg_import="$1"
    local pkg_pip="$2"
    if python -c "import ${pkg_import}" 2>/dev/null; then
        local ver
        ver=$(python -c "import ${pkg_import}; print(getattr(${pkg_import}, '__version__', 'installed'))" 2>/dev/null || echo "installed")
        echo "  ✓ ${pkg_pip} (${ver})"
    else
        echo "  ✗ ${pkg_pip} not found — installing..."
        pip install "${pkg_pip}"
        echo "  ✓ ${pkg_pip} installed"
    fi
}

# ─── Core ML packages ───────────────────────────────────────────────────────
echo ">>> PyTorch ecosystem..."
check_and_install "torch"        "torch"
check_and_install "torchvision"  "torchvision"

echo ""
echo ">>> Ultralytics (YOLO26, SME-YOLO, RT-DETR)..."
check_and_install "ultralytics"  "ultralytics"

echo ""
echo ">>> timm (ViT backbone for vit_det)..."
check_and_install "timm"         "timm"

# NOTE: mambavision skipped — mamba-ssm fails to compile on this server
# echo ""
# echo ">>> MambaVision (vit_mamba backbone)..."
# check_and_install "mambavision"  "mambavision"

echo ""
echo ">>> Data processing & utilities..."
check_and_install "yaml"         "pyyaml"
check_and_install "sklearn"      "scikit-learn"
check_and_install "PIL"          "pillow"
check_and_install "matplotlib"   "matplotlib"

# ─── Verify CUDA ─────────────────────────────────────────────────────────────
echo ""
echo ">>> Verifying CUDA..."
python -c "
import torch
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA avail:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: CUDA not available — training will be slow on CPU!')
"

# ─── Quick import test ───────────────────────────────────────────────────────
echo ""
echo ">>> Verifying model imports..."
python -c "
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath('$0')))
sys.path.insert(0, '$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)')
from models import AVAILABLE_MODELS
print(f'  Available models: {AVAILABLE_MODELS}')
# Test creating non-mamba models
for m in ['faster_rcnn', 'vit_det', 'yolo26', 'sme_yolo', 'rt_detr']:
    try:
        __import__('importlib').import_module(f'models.{m.replace(\"-\",\"_\")}')
        print(f'  ✓ {m} module imports OK')
    except Exception as e:
        print(f'  ✗ {m} import FAILED: {e}')
print('  (vit_mamba skipped — mambavision not installed)')
"

echo ""
echo "============================================================"
echo "  All dependencies installed and verified!"
echo "  You can now run: bash run_all_models.sh"
echo "============================================================"
