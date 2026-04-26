#!/bin/bash
set -euo pipefail

export PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH=/opt/heat/models/ops:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

cd /opt/heat

python -c "import MultiScaleDeformableAttention as m; print('MSDA OK:', m.__file__)"

CUDA_VISIBLE_DEVICES=0 python infer.py \
  --checkpoint_path ./checkpoints/main_fine_tuned_my_finetune_512/checkpoint_best.pth \
  --dataset outdoor \
  --image_size 512 \
  --corner_thresh 0.003 \
  --edge_thresh 0.50 \
  --infer_times 1 \
  --viz_base ./results/viz_old_best_ct_0.003_et_0.50 \
  --save_base ./results/npy_old_best_ct_0.003_et_0.50 \