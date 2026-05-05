#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A c02114
#SBATCH --job-name=simclr_finetune
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/simclr_finetune_%j.out

set -euo pipefail

mkdir -p logs

# Load conda
: "${PS1:=}"
set +u
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hand_env
set -u

echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Time:   $(date)"
echo ""

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ============================================================
# STAGE 9 — Fine-tune linear classifier (baby/adult/other_kids)
# ============================================================
log "=== STAGE 9: SimCLR Fine-tuning ==="
srun python scripts/simclr_finetune.py
log "Stage 9 complete."

log "=== FINE-TUNING DONE ==="
log "Best model saved to: finetune_checkpoints/finetune_best.pt"
