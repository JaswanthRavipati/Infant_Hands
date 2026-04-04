#!/bin/bash
# ============================================================
# Infant Hands — SimCLR Pipeline Part 3 (Classification)
# Stages 7–9: Hand crop → SimCLR pre-train → fine-tune
#
# Prerequisites:
#   - Part 1 + Part 2 complete (YOLO trained)
#   - For Stage 9: labeled_crops/ folder populated with
#     subfolders: baby/, adult/, other_kids/
#
# Usage:
#   bash bash/run_pipeline_part3.sh           # run all stages
#   bash bash/run_pipeline_part3.sh --from 8  # resume from stage 8
# ============================================================

set -euo pipefail

START_STAGE=${2:-7}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ============================================================
# STAGE 7 — Extract 224x224 hand crops using trained YOLO
# ============================================================
if [ "$START_STAGE" -le 7 ]; then
    log "=== STAGE 7: Hand Crop Extraction ==="
    python scripts/hand_crop.py
    log "Stage 7 complete."
fi

# ============================================================
# STAGE 8 — SimCLR self-supervised pre-training
#            Input:  hand_crop/ (unlabeled, all crops)
#            Output: simclr_checkpoints/simclr_resnet50_best.pt
# ============================================================
if [ "$START_STAGE" -le 8 ]; then
    log "=== STAGE 8: SimCLR Pre-training ==="
    python scripts/simclr_pretrain.py
    log "Stage 8 complete."
fi

# ============================================================
# STAGE 9 — Fine-tune linear classifier (baby/adult/other_kids)
#            Input:  labeled_crops/{baby,adult,other_kids}/
#            Output: finetune_checkpoints/finetune_best.pt
# ============================================================
if [ "$START_STAGE" -le 9 ]; then
    log "=== STAGE 9: SimCLR Fine-tuning ==="
    log "Expecting labeled data at: labeled_crops/baby/, adult/, other_kids/"
    python scripts/simclr_finetune.py
    log "Stage 9 complete."
fi

log "=== SIMCLR PIPELINE COMPLETE ==="
