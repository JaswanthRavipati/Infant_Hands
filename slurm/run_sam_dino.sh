#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A c02114
#SBATCH --job-name=sam2_dino_pipeline
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sam2_dino_%j.out
#SBATCH --error=logs/sam2_dino_%j.err

set -euo pipefail

# =========================
# FIX CONDA ISSUE
# =========================
: "${PS1:=}"
set +u
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hand_env
set -u

# =========================
# GO TO PROJECT
# =========================
cd ~/infanthands

# =========================
# CREATE REQUIRED FOLDERS
# =========================
mkdir -p logs
mkdir -p /N/slate/veravi/dino_outputs
mkdir -p /N/slate/veravi/dino_outputs/csv

# =========================
# DEBUG INFO
# =========================
echo "================================="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Time: $(date)"
echo "================================="

# =========================
# RUN SCRIPT
# =========================
srun python scripts/sam_dino.py

echo "================================="
echo "JOB FINISHED"
echo "================================="