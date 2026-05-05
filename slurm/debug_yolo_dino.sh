#!/bin/bash
#SBATCH -p gpu-debug
#SBATCH --gpus=1
#SBATCH -A c02114
#SBATCH --job-name=debug_yolo_dino
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

set -euo pipefail

# =========================
# FIX CONDA
# =========================
: "${PS1:=}"
set +u
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hand_env
set -u

cd ~/infanthands

mkdir -p logs

echo "=============================="
echo "DEBUG JOB STARTED"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Time: $(date)"
echo "=============================="

# =========================
# 🔥 DEBUG MODE SETTINGS
# =========================
export DEBUG=1

# =========================
# RUN CODE
# =========================
srun python scripts/final.py

echo "=============================="
echo "DEBUG JOB FINISHED"
echo "=============================="