#!/bin/bash
#SBATCH --job-name=heat512
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/fo37nor/assets/heat/slurm_logs/heat512_%j.out
#SBATCH --error=/home/fo37nor/assets/heat/slurm_logs/heat512_%j.err

set -euo pipefail

mkdir -p /home/fo37nor/assets/heat/slurm_logs
mkdir -p /home/fo37nor/assets/heat/runtime_info

module load nvidia/cuda/12.4.0

HOSTNAME_FILE=/home/fo37nor/assets/heat/runtime_info/heat512_${SLURM_JOB_ID}.node
TB_LOG_FILE=/home/fo37nor/assets/heat/slurm_logs/tensorboard_${SLURM_JOB_ID}.log

hostname > "$HOSTNAME_FILE"

# Start TensorBoard in background inside the same container
singularity exec --cleanenv --nv \
  -B /home/fo37nor/assets/heat:/opt/heat \
  -B /cluster:/cluster \
  /home/fo37nor/assets/heat/heat_with_tensorboard.sif \
  bash /opt/heat/start_train_tensorboard_512.sh \
  > "$TB_LOG_FILE" 2>&1 &

TB_PID=$!

# Run training in foreground inside the same container
singularity exec --cleanenv --nv \
  -B /home/fo37nor/assets/heat:/opt/heat \
  -B /cluster:/cluster \
  /home/fo37nor/assets/heat/heat_with_tensorboard.sif \
  bash /opt/heat/start_train_512.sh

# Clean up TensorBoard when training ends
kill $TB_PID || true