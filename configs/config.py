# -*- coding: utf-8 -*-
"""
Public example configuration.

Copy this file to:

    configs/config_local.py

Then edit the paths for your own machine/HPC.
"""

from pathlib import Path


# Root of this repository
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Preprocessing input/output
# ---------------------------------------------------------------------

# Full satellite GeoTIFF input
SATELLITE_TIF = Path("/path/to/input/satellite.tif")

# Folder where preprocessing outputs will be written
PREPROCESS_OUTPUT = PROJECT_ROOT / "work" / "preprocessing_output"


# ---------------------------------------------------------------------
# HEAT runtime
# ---------------------------------------------------------------------

# Singularity image or Docker-derived .sif image
SINGULARITY_IMAGE = Path("/path/to/heat_with_tensorboard.sif")

# Folder containing HEAT checkpoint files
CHECKPOINTS_DIR = Path("/path/to/checkpoints")

# Main HEAT checkpoint relative to CHECKPOINTS_DIR
CHECKPOINT_RELATIVE_PATH = Path("main_fine_tuned_my_finetune_512/checkpoint_best.pth")

# Container backend used by scripts/run_heat.py
# Supported values planned:
#   "singularity" = HPC / Singularity runtime
#   "docker"      = local Docker runtime
CONTAINER_BACKEND = "singularity"
DOCKER_IMAGE = "vaibhavrajan79/heat_deformable-detr-image:with_tensorboard"


# ---------------------------------------------------------------------
# HEAT input/output inside this repo
# ---------------------------------------------------------------------

HEAT_INPUT_DIR = PROJECT_ROOT / "work" / "heat_input"
HEAT_OUTPUT_DIR = PROJECT_ROOT / "work" / "heat_output"


# ---------------------------------------------------------------------
# HEAT inference settings
# ---------------------------------------------------------------------

IMAGE_SIZE = 512
CORNER_THRESH = 0.003
EDGE_THRESH = 0.50
INFER_TIMES = 1

HEAT_RESULT_NAME = "old_best_ct_0.003_et_0.50"


# ---------------------------------------------------------------------
# Optional local precompiled HEAT ops
# ---------------------------------------------------------------------

# Public users can ignore this and build ops using docs/build_heat_ops.md later.
LOCAL_HEAT_OPS_SOURCE_DIR = Path("/path/to/existing/heat/models/ops")
