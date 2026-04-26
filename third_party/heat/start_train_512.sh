#!/bin/bash
set -euo pipefail

export PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH=/opt/heat/models/ops:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

cd /opt/heat

python -c "import MultiScaleDeformableAttention as m; print('MSDA OK:', m.__file__)"

CUDA_VISIBLE_DEVICES=0 python train_with_backbone_unfreeze_small_LR.py \
  --run_validation \
  --image_size 512 \
  --resume ./checkpoints/main_fine_tuned_my_finetune_512/checkpoint_best.pth \
  --output_dir ./checkpoints/stage2_unfreeze_512 \
  --batch_size 2 \
  --lr 1e-5 \
  --epochs 120 \
  --lr_drop 80