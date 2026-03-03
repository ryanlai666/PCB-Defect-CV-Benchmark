#!/usr/bin/env bash
# ==============================================================================
# run_all_models.sh — Train all PCB defect detection models sequentially
# (faster_rcnn / faster_rcnn_ft excluded — using pre-trained Colab checkpoint)
# (vit_mamba excluded — mambavision cannot compile on this server)
#
# Usage:
#   bash run_all_models.sh                     # train all models (full)
#   bash run_all_models.sh --test              # test mode: all models, 1 epoch, 80/40/40 images
#   bash run_all_models.sh faster_rcnn_ft      # run faster_rcnn_ft only
#   bash run_all_models.sh --test vit_det      # test mode: only vit_det
#
# Environment: conda activate pcb
# ==============================================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="pcb"
EPOCHS=20
BATCH_SIZE=4

# Output directories
OUTPUT_BASE="${PROJECT_DIR}/outputs"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/run_all_${TIMESTAMP}.log"

# All available models (faster_rcnn / faster_rcnn_ft skipped: using Colab checkpoint)
# DEIMv2-L/X: use DEIMv2/train.py via torchrun (subprocess) — see train_model.py
ALL_MODELS=( rt_detr vit_det deimv2_l deimv2_x)  # yolo26 sme_yolo

# ─── Parse arguments ────────────────────────────────────────────────────────
TEST_MODE=0
TEST_MODE_FLAG=""
MODEL_ARGS=()

for arg in "$@"; do
    if [ "${arg}" = "--test" ]; then
        TEST_MODE=1
    else
        MODEL_ARGS+=("${arg}")
    fi
done

if [ ${#MODEL_ARGS[@]} -gt 0 ]; then
    MODELS=("${MODEL_ARGS[@]}")
else
    MODELS=("${ALL_MODELS[@]}")
fi

if [ ${TEST_MODE} -eq 1 ]; then
    TEST_MODE_FLAG="--test_mode"
    EPOCHS=1
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
# Source conda so we can activate the environment
# Note: set +u is needed because some conda hook scripts reference unset variables
CONDA_BASE="$(conda info --base 2>/dev/null || echo '/home/yuhsuanho/anaconda3')"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

log_separator
log "  PCB Defect Detection — Sequential Model Training"
if [ ${TEST_MODE} -eq 1 ]; then
log "  *** TEST MODE: 1 epoch, 80 train / 40 val / 40 test images ***"
fi
log "  Timestamp:    ${TIMESTAMP}"
log "  Conda Env:    ${CONDA_ENV}"
log "  Python:       $(which python)"
log "  PyTorch:      $(python -c 'import torch; print(torch.__version__)')" 
log "  CUDA:         $(python -c 'import torch; print(torch.cuda.is_available())')"
log "  GPU:          $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
log "  Epochs:       ${EPOCHS}"
log "  Batch Size:   ${BATCH_SIZE}"
log "  Models:       ${MODELS[*]}"
log "  Project Dir:  ${PROJECT_DIR}"
log "  Output Base:  ${OUTPUT_BASE}"
log "  Master Log:   ${MASTER_LOG}"
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

# ─── Train each model ───────────────────────────────────────────────────────
for MODEL_NAME in "${MODELS[@]}"; do
    log_separator
    log ">>> Starting training for: ${MODEL_NAME}"
    log_separator

    MODEL_OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
    MODEL_LOG="${LOG_DIR}/${MODEL_NAME}_${TIMESTAMP}.log"
    mkdir -p "${MODEL_OUTPUT_DIR}"

    MODEL_START=$(date +%s)

    # Run training and capture exit code
    set +e
    python "${PROJECT_DIR}/train_model.py" \
        --model "${MODEL_NAME}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_dir "${MODEL_OUTPUT_DIR}" \
        ${TEST_MODE_FLAG} \
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
log "  FINAL SUMMARY"
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
log "  Log files:"
log "    Master log:     ${MASTER_LOG}"
for MODEL_NAME in "${MODELS[@]}"; do
    log "    ${MODEL_NAME}:$(printf '%*s' $((15 - ${#MODEL_NAME})) '')${LOG_DIR}/${MODEL_NAME}_${TIMESTAMP}.log"
done

log ""
log "  Best model checkpoints:"
for MODEL_NAME in "${MODELS[@]}"; do
    BEST_PT="${OUTPUT_BASE}/${MODEL_NAME}/best_${MODEL_NAME}.pth"
    if [ -f "${BEST_PT}" ]; then
        FILESIZE=$(du -h "${BEST_PT}" | cut -f1)
        log "    ${MODEL_NAME}:$(printf '%*s' $((15 - ${#MODEL_NAME})) '')${BEST_PT} (${FILESIZE})"
    else
        log "    ${MODEL_NAME}:$(printf '%*s' $((15 - ${#MODEL_NAME})) '')NOT FOUND"
    fi
done

log_separator
log "  All done!"
log_separator
