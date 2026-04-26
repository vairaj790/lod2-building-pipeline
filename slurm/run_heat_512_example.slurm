#!/bin/bash
#SBATCH --job-name=heat512
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/heat512_%j.out
#SBATCH --error=slurm_logs/heat512_%j.err

set -euo pipefail

# Change this to your cloned lod2_building_pipeline path
PROJECT_ROOT="/path/to/lod2_building_pipeline"

# Change this to your Singularity image path
SIF_IMAGE="/path/to/heat_with_tensorboard.sif"

HEAT_DIR="${PROJECT_ROOT}/third_party/heat"
LOG_DIR="${PROJECT_ROOT}/slurm_logs"
RUNTIME_DIR="${PROJECT_ROOT}/runtime_info"

mkdir -p "$LOG_DIR"
mkdir -p "$RUNTIME_DIR"

# Load CUDA module if your HPC requires it
# module load nvidia/cuda/12.4.0

hostname > "${RUNTIME_DIR}/heat512_${SLURM_JOB_ID}.node"

singularity exec --cleanenv --nv \
  -B "${HEAT_DIR}:/opt/heat" \
  -B "${PROJECT_ROOT}:/workspace" \
  "$SIF_IMAGE" \
  bash /opt/heat/start_infer.sh
