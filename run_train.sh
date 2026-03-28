#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A c02114
#SBATCH --job-name=hand_train_v1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/hand_train_v1_%j.out

set -euo pipefail


# Load conda properly (robust way)
: "${PS1:=}"
set +u
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hand_env
set -u

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Time: $(date)"

# Run training
srun python train_yolo.py