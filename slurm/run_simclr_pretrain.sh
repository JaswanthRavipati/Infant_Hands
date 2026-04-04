#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A c02114
#SBATCH --job-name=simclr_pretrain
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/simclr_pretrain_%j.out

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
# STAGE 7 — Extract 224x224 hand crops using trained YOLO
# ============================================================
log "=== STAGE 7: Hand Crop Extraction ==="
srun --ntasks=1 python scripts/hand_crop.py
log "Stage 7 complete."

# ============================================================
# STAGE 8 — SimCLR self-supervised pre-training (200 epochs)
# ============================================================
log "=== STAGE 8: SimCLR Pre-training ==="
srun python scripts/simclr_pretrain.py
log "Stage 8 complete."

log "=== PRE-TRAINING DONE ==="
log "Next: label a subset of crops, then run: sbatch slurm/run_simclr_finetune.sh"
