#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A c02114
#SBATCH --job-name=yolo_test
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/yolo_test_%j.out

set -euo pipefail

# Activate conda
: "${PS1:=}"
set +u
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hand_env
set -u

cd ~/infanthands

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

srun python test_yolo.py