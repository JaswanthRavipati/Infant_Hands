#!/bin/bash
# ============================================================
# Infant Hands — Training Pipeline Part 2 (Post-CVAT)
# Stages 5–6: Train/val split → YOLO training
#
# Run this AFTER completing manual CVAT annotation review
# and exporting corrected labels back to dataset/labels/train/
#
# Usage:
#   bash run_pipeline_part2.sh           # run all stages
#   bash run_pipeline_part2.sh --from 6  # resume from stage 6
# ============================================================

set -euo pipefail

START_STAGE=${2:-5}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ============================================================
# STAGE 5 — Day-stratified train/val split
#            (Day 3 → val, Days 1+2 → train, all 5 subjects)
# ============================================================
if [ "$START_STAGE" -le 5 ]; then
    log "=== STAGE 5: Train/Val Split ==="
    python scripts/split.py
    log "Stage 5 complete."
fi

# ============================================================
# STAGE 6 — Fine-tune YOLOv8m on labeled hand dataset
#            (100 epochs, GPU required)
# ============================================================
if [ "$START_STAGE" -le 6 ]; then
    log "=== STAGE 6: Train YOLO ==="
    python scripts/train_yolo.py
    log "Stage 6 complete."
fi

log "=== TRAINING PIPELINE COMPLETE ==="
