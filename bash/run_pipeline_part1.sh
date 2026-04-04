#!/bin/bash
# ============================================================
# Infant Hands — Training Pipeline Part 1 (Automated)
# Stages 1–4: Frame extraction → label generation → dedup
#
# After this completes, manually review and correct labels
# in CVAT before running run_pipeline_part2.sh
#
# Usage:
#   bash run_pipeline_part1.sh           # run all stages
#   bash run_pipeline_part1.sh --from 3  # resume from stage 3
# ============================================================

set -euo pipefail

START_STAGE=${2:-1}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ============================================================
# STAGE 1 — Extract frames from MP4 videos at 5 FPS
# ============================================================
if [ "$START_STAGE" -le 1 ]; then
    log "=== STAGE 1: Frame Extraction ==="
    python scripts/hands.py
    log "Stage 1 complete."
fi

# ============================================================
# STAGE 2 — Filter frames to only those with detected hands
#            (MediaPipe + CSV annotation filter)
# ============================================================
if [ "$START_STAGE" -le 2 ]; then
    log "=== STAGE 2: Hand Detection Filter ==="
    python scripts/detect_hands.py
    log "Stage 2 complete."
fi

# ============================================================
# STAGE 3 — Generate YOLO bounding box labels from landmarks
# ============================================================
if [ "$START_STAGE" -le 3 ]; then
    log "=== STAGE 3: Generate YOLO Labels ==="
    python scripts/generate_hand_labels.py
    log "Stage 3 complete."
fi

# ============================================================
# STAGE 4 — Remove duplicate bounding boxes from label files
# ============================================================
if [ "$START_STAGE" -le 4 ]; then
    log "=== STAGE 4: Deduplicate Bounding Boxes ==="
    python scripts/delete_dup_box.py
    log "Stage 4 complete."
fi

log ""
log "=== PART 1 COMPLETE ==="
log ""
log "NEXT STEP — Manual CVAT Annotation Review:"
log "  1. Import images from:  dataset/images/train/"
log "  2. Import labels from:  dataset/labels/train/  (YOLO format)"
log "  3. Review and correct bounding boxes (add padding, fix misses)"
log "  4. Export corrected labels back to: dataset/labels/train/  (YOLO format)"
log "  5. Then run: bash bash/run_pipeline_part2.sh"
log ""
