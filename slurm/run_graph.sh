#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -A c02114
#SBATCH --job-name=hand_graph
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hand_graph_%j.out

set -euo pipefail

# Activate conda
: "${PS1:=}"
set +u
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hand_env
set -u

# Move to project root
cd ~/infanthands

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# (Optional but recommended for logs)
mkdir -p logs

# Run your script
srun python scripts/graph.py