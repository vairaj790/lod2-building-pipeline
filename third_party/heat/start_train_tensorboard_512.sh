#!/bin/bash
set -euo pipefail

export PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH=/opt/heat/models/ops:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

cd /opt/heat

tensorboard \
  --logdir /opt/heat/checkpoints/stage2_unfreeze_512/tensorboard_logs \
  --port 6006 \
  --host 0.0.0.0