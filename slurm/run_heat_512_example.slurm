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

PROJECT_ROOT="/path/to/lod2_building_pipeline"
SIF_IMAGE="/path/to/heat_with_tensorboard.sif"

HEAT_DIR="${PROJECT_ROOT}/third_party/heat"

mkdir -p "${PROJECT_ROOT}/slurm_logs"
mkdir -p "${PROJECT_ROOT}/runtime_info"

singularity exec --cleanenv --nv \
  -B "${HEAT_DIR}:/opt/heat" \
  -B "${PROJECT_ROOT}:/workspace" \
  "$SIF_IMAGE" \
  bash /opt/heat/start_infer.sh
