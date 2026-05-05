#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A c02114
#SBATCH --job-name=sam2_dino
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/sam2_dino_%j.out
#SBATCH --error=logs/sam2_dino_%j.err

set -euo pipefail

# =========================
# FIX PS1 ISSUE
# =========================
: "${PS1:=}"
set +u

# =========================
# LOAD CONDA
# =========================
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hand_env

set -u

# =========================
# MOVE TO PROJECT DIR
# =========================
cd ~/infanthands

# =========================
# DEBUG INFO
# =========================
echo "=============================="
echo "Job started on: $(hostname)"
echo "Date: $(date)"
echo "Working Dir: $(pwd)"
echo "=============================="

echo "Python path:"
which python

echo "CUDA check:"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

echo "GPU Info:"
nvidia-smi

# =========================
# SAFE FILE CHECKS (NO LS)
# =========================
echo "Checking required paths..."

if [ ! -d "/N/slate/veravi/test_videos" ]; then
    echo "❌ ERROR: test_videos folder missing"
    exit 1
fi

if [ ! -d "/N/slate/veravi/model" ]; then
    echo "❌ ERROR: model folder missing"
    exit 1
fi

if [ ! -f "/N/slate/veravi/model/prototypes.pth" ]; then
    echo "❌ ERROR: prototypes.pth missing"
    exit 1
fi

if [ ! -f "/N/slate/veravi/model/sam2_hiera_large.pt" ]; then
    echo "❌ ERROR: SAM2 checkpoint missing"
    exit 1
fi

echo "✅ All required files found"

# =========================
# RUN SCRIPT
# =========================
echo "Starting SAM2 + DINO pipeline..."

srun python scripts/test_sam2.py

echo "=============================="
echo "Job finished at: $(date)"
echo "=============================="