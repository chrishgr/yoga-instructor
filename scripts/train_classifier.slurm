#!/bin/bash
#SBATCH --job-name=coachnasarus-train
#SBATCH --account=ec403
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

# Train the yoga pose classifier on Fox.
# Submit with:  sbatch scripts/train_classifier.slurm

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0

# Activate project venv (create it once on the login node with:
#   python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt scikit-learn)
source .venv/bin/activate

mkdir -p logs models

python scripts/train_classifier.py \
    --dataset "${DATASET_PATH:-/fp/projects01/ec403/coachnasarus/datasets/yoga_poses}" \
    --output  "${OUTPUT_MODEL:-models/knn_classifier.pkl}" \
    --n-neighbors "${N_NEIGHBORS:-5}"

echo "Done. Model is at ${OUTPUT_MODEL:-models/knn_classifier.pkl}"
