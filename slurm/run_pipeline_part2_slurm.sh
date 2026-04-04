#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A c02114
#SBATCH --job-name=infanthands_part2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/infanthands_part2_%j.out

set -euo pipefail

mkdir -p logs

# Load conda
: "${PS1:=}"
set +u
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hand_env
set -u

# 🔥 FIX: move to project root
cd ~/infanthands

echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Time:   $(date)"
echo ""

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ============================================================
# STAGE 5 — Train/Val Split
# ============================================================
log "=== STAGE 5: Train/Val Split ==="
srun --ntasks=1 python scripts/split.py
log "Stage 5 complete."

# ============================================================
# STAGE 6 — Train YOLO
# ============================================================
log "=== STAGE 6: Train YOLO ==="
srun python scripts/train_yolo.py
log "Stage 6 complete."

log "=== PART 2 COMPLETE ==="