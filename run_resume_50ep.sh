#!/usr/bin/env bash
# ==============================================================================
# run_resume_50ep.sh — Resume all PCB defect detection models from ~20 epoch
#                      checkpoints and continue training to 50 total epochs.
#
# IMPORTANT: Before using this script, ensure that outputs_20ep_backup/ exists
#            as a safe copy of the original 20-epoch checkpoints.
#
# Usage:
#   bash run_resume_50ep.sh                     # resume all models to 50 epochs
#   bash run_resume_50ep.sh vit_det             # resume only vit_det
#   bash run_resume_50ep.sh yolo26 sme_yolo     # resume specific models
#
# Environment: conda activate pcb
# ==============================================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="pcb"
EPOCHS=50
BATCH_SIZE=4

# Output directories
OUTPUT_BASE="${PROJECT_DIR}/outputs"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/resume_50ep_${TIMESTAMP}.log"

# Models to resume.
# Notes:
# - faster_rcnn:    trained 1 epoch initially (used Colab pre-trained checkpoint)
# - faster_rcnn_ft: trained 1 epoch initially (used Colab pre-trained checkpoint)
# - vit_det:        trained 20 epochs
# - yolo26:         trained 20 epochs (Ultralytics)
# - sme_yolo:       trained 20 epochs (Ultralytics)
# - rt_detr:        trained 20 epochs (Ultralytics)
# - deimv2_l:       trained ~31 epochs (DEIMv2 subprocess)
ALL_MODELS=( vit_det yolo26 sme_yolo rt_detr faster_rcnn deimv2_l )

# ─── Parse arguments ────────────────────────────────────────────────────────
MODEL_ARGS=()

for arg in "$@"; do
    MODEL_ARGS+=("${arg}")
done

if [ ${#MODEL_ARGS[@]} -gt 0 ]; then
    MODELS=("${MODEL_ARGS[@]}")
else
    MODELS=("${ALL_MODELS[@]}")
fi

# ─── Setup directories ──────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_BASE}"

# ─── Logging helper ─────────────────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}" | tee -a "${MASTER_LOG}"
}

log_separator() {
    echo "======================================================================" | tee -a "${MASTER_LOG}"
}

# ─── Conda activation ───────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo '/home/yuhsuanho/anaconda3')"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

log_separator
log "  PCB Defect Detection — RESUME Training to ${EPOCHS} Epochs"
log "  Timestamp:    ${TIMESTAMP}"
log "  Conda Env:    ${CONDA_ENV}"
log "  Python:       $(which python)"
log "  PyTorch:      $(python -c 'import torch; print(torch.__version__)')" 
log "  CUDA:         $(python -c 'import torch; print(torch.cuda.is_available())')"
log "  GPU:          $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
log "  Target Epochs: ${EPOCHS}"
log "  Batch Size:   ${BATCH_SIZE}"
log "  Models:       ${MODELS[*]}"
log "  Project Dir:  ${PROJECT_DIR}"
log "  Output Base:  ${OUTPUT_BASE}"
log "  Master Log:   ${MASTER_LOG}"
log ""
log "  20-epoch backup: ${PROJECT_DIR}/outputs_20ep_backup/"
if [ -d "${PROJECT_DIR}/outputs_20ep_backup" ]; then
    log "    ✓ Backup directory exists"
else
    log "    ✗ WARNING: Backup directory NOT found!"
    log "    Please run first: cp -a outputs outputs_20ep_backup"
fi
log_separator

# ─── Pre-flight dependency check ────────────────────────────────────────────
log ""
log ">>> Pre-flight dependency check..."
MISSING=0
for pkg in ultralytics timm yaml sklearn PIL matplotlib torch torchvision; do
    if ! python -c "import ${pkg}" 2>/dev/null; then
        log "    ✗ MISSING: ${pkg}"
        MISSING=1
    fi
done

if [ ${MISSING} -eq 1 ]; then
    log ""
    log "ERROR: Some dependencies are missing!"
    log "Please run first:  bash ${PROJECT_DIR}/install_deps.sh"
    exit 1
fi
log "    All dependencies OK."
log ""

# ─── Track results ──────────────────────────────────────────────────────────
declare -A MODEL_STATUS
declare -A MODEL_ELAPSED
TOTAL_START=$(date +%s)

# ─── Resume each model ─────────────────────────────────────────────────────
for MODEL_NAME in "${MODELS[@]}"; do
    log_separator
    log ">>> RESUMING training for: ${MODEL_NAME} (target: ${EPOCHS} epochs)"
    log_separator

    MODEL_OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
    MODEL_LOG="${LOG_DIR}/${MODEL_NAME}_resume50_${TIMESTAMP}.log"
    mkdir -p "${MODEL_OUTPUT_DIR}"

    MODEL_START=$(date +%s)

    # Run training with --resume flag
    set +e
    python "${PROJECT_DIR}/train_model.py" \
        --model "${MODEL_NAME}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_dir "${MODEL_OUTPUT_DIR}" \
        --resume \
        2>&1 | tee "${MODEL_LOG}" | tee -a "${MASTER_LOG}"
    EXIT_CODE=$?
    set -e

    MODEL_END=$(date +%s)
    MODEL_ELAPSED_SEC=$((MODEL_END - MODEL_START))
    MODEL_HOURS=$((MODEL_ELAPSED_SEC / 3600))
    MODEL_MINS=$(((MODEL_ELAPSED_SEC % 3600) / 60))
    MODEL_SECS=$((MODEL_ELAPSED_SEC % 60))
    MODEL_ELAPSED["${MODEL_NAME}"]="${MODEL_HOURS}h ${MODEL_MINS}m ${MODEL_SECS}s"

    if [ ${EXIT_CODE} -eq 0 ]; then
        MODEL_STATUS["${MODEL_NAME}"]="SUCCESS"
        log ""
        log ">>> ${MODEL_NAME}: Training completed SUCCESSFULLY in ${MODEL_ELAPSED[${MODEL_NAME}]}"

        # Check if best model file exists
        BEST_PT="${MODEL_OUTPUT_DIR}/best_${MODEL_NAME}.pth"
        if [ -f "${BEST_PT}" ]; then
            FILESIZE=$(du -h "${BEST_PT}" | cut -f1)
            log "    Best model: ${BEST_PT} (${FILESIZE})"
        else
            log "    WARNING: Best model file not found at ${BEST_PT}"
        fi
    else
        MODEL_STATUS["${MODEL_NAME}"]="FAILED (exit code: ${EXIT_CODE})"
        log ""
        log ">>> ${MODEL_NAME}: Training FAILED with exit code ${EXIT_CODE}"
        log "    See log: ${MODEL_LOG}"
    fi

    log ""
done

# ─── Final summary ──────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINS=$(((TOTAL_ELAPSED % 3600) / 60))
TOTAL_SECS=$((TOTAL_ELAPSED % 60))

log_separator
log "  FINAL SUMMARY — Resume to ${EPOCHS} Epochs"
log_separator
log ""
log "  Total elapsed:  ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
log ""

printf "  %-15s %-25s %-20s\n" "Model" "Status" "Time" | tee -a "${MASTER_LOG}"
printf "  %-15s %-25s %-20s\n" "─────────────" "───────────────────────" "──────────────────" | tee -a "${MASTER_LOG}"

for MODEL_NAME in "${MODELS[@]}"; do
    STATUS="${MODEL_STATUS[${MODEL_NAME}]:-UNKNOWN}"
    ELAPSED="${MODEL_ELAPSED[${MODEL_NAME}]:-N/A}"
    printf "  %-15s %-25s %-20s\n" "${MODEL_NAME}" "${STATUS}" "${ELAPSED}" | tee -a "${MASTER_LOG}"
done

log ""
log "  Best model checkpoints (50 epoch):"
for MODEL_NAME in "${MODELS[@]}"; do
    BEST_PT="${OUTPUT_BASE}/${MODEL_NAME}/best_${MODEL_NAME}.pth"
    if [ -f "${BEST_PT}" ]; then
        FILESIZE=$(du -h "${BEST_PT}" | cut -f1)
        log "    ${MODEL_NAME}:$(printf '%*s' $((15 - ${#MODEL_NAME})) '')${BEST_PT} (${FILESIZE})"
    else
        log "    ${MODEL_NAME}:$(printf '%*s' $((15 - ${#MODEL_NAME})) '')NOT FOUND"
    fi
done

log ""
log "  20-epoch backup preserved at: ${PROJECT_DIR}/outputs_20ep_backup/"
log_separator
log "  All done!"
log_separator
