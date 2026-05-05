#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A c02114
#SBATCH --job-name=yolo_dino_pipeline
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/yolo_dino_%j.out
#SBATCH --error=logs/yolo_dino_%j.err

set -euo pipefail

# =========================
# FIX CONDA (important on HPC)
# =========================
: "${PS1:=}"
set +u
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hand_env
set -u

# =========================
# GO TO PROJECT DIRECTORY
# =========================
cd ~/infanthands

mkdir -p logs

echo "=============================="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo "=============================="

# =========================
# RUN SCRIPT
# =========================
srun python scripts/final.py

echo "=============================="
echo "End time: $(date)"
echo "DONE"
echo "=============================="