# -*- coding: utf-8 -*-
"""
Then edit the paths for your own machine/HPC.
"""

from pathlib import Path


# Root of this repository
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "input"


def find_single_input_file(patterns, label):
    matches = []
    for pattern in patterns:
        matches.extend(INPUT_DIR.glob(pattern))

    matches = sorted([p for p in matches if p.is_file()])

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {label} found in {INPUT_DIR}. "
            f"Expected exactly one file matching: {patterns}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"More than one {label} found in {INPUT_DIR}. "
            f"Please keep exactly one matching file. Found: {matches}"
        )

    return matches[0]



# ---------------------------------------------------------------------
# Preprocessing input/output
# ---------------------------------------------------------------------

# Full satellite GeoTIFF input
SATELLITE_TIF = find_single_input_file(["*.tif", "*.tiff", "*.TIF", "*.TIFF"], "satellite GeoTIFF")

# Folder where preprocessing outputs will be written
PREPROCESS_OUTPUT = PROJECT_ROOT / "work" / "heat_input"


# ---------------------------------------------------------------------
# HEAT runtime
# ---------------------------------------------------------------------

# Singularity image or Docker-derived .sif image
SINGULARITY_IMAGE = Path("/path/to/heat_with_tensorboard.sif")

# Folder containing HEAT checkpoint files
CHECKPOINTS_DIR = PROJECT_ROOT / "third_party" / "heat" / "checkpoints"

# Main HEAT checkpoint relative to CHECKPOINTS_DIR
CHECKPOINT_RELATIVE_PATH = Path("heat_checkpoint_finetuned_512/checkpoint_best.pth")

# Container backend used by scripts/run_heat.py
# Supported values planned:
#   "singularity" = HPC / Singularity runtime
#   "docker"      = local Docker runtime
CONTAINER_BACKEND = "singularity"
DOCKER_IMAGE = "vaibhavrajan79/heat_deformable-detr-image:with_tensorboard"


# ---------------------------------------------------------------------
# HEAT input/output inside this repo
# ---------------------------------------------------------------------

HEAT_INPUT_DIR = PREPROCESS_OUTPUT
HEAT_OUTPUT_DIR = PROJECT_ROOT / "work" / "heat_output"


# ---------------------------------------------------------------------
# HEAT inference settings
# ---------------------------------------------------------------------

IMAGE_SIZE = 512
CORNER_THRESH = 0.003
EDGE_THRESH = 0.50
INFER_TIMES = 1

HEAT_RESULT_NAME = ""


# ---------------------------------------------------------------------
# Optional local precompiled HEAT ops
# ---------------------------------------------------------------------

# Public users can ignore this and build ops using docs/build_heat_ops.md later.
LOCAL_HEAT_OPS_SOURCE_DIR = Path("/path/to/existing/heat/models/ops")

# ---------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------

POSTPROCESS_OUTPUT_DIR = PROJECT_ROOT / "work" / "georeferenced_output"

# ---------------------------------------------------------------------
# LiDAR preprocessing / cropping
# ---------------------------------------------------------------------

MAIN_LIDAR_PATH = find_single_input_file(["*.laz", "*.LAZ"], "main LiDAR LAZ")
LIDAR_INPUT_DIR = PROJECT_ROOT / "work" / "Lidar_input"

LIDAR_EXTRA_BUFFER_M = 0.0
LIDAR_POINTS_PER_CHUNK = 2_000_000
SAVE_MERGED_LIDAR_DEBUG_FILE = True
MERGED_LIDAR_DEBUG_NAME = "merged_cropped_buildings_from_metadata.laz"


# ---------------------------------------------------------------------
# Interactive LoD2 / 3D reconstruction stage
# ---------------------------------------------------------------------

LOD2_3D_OUTPUT_DIR = PROJECT_ROOT / "work" / "3D_output"
LOD2_SNAPSHOT_DIR = LOD2_3D_OUTPUT_DIR / "snapshots"
LOD2_RMSE_CSV_PATH = LOD2_3D_OUTPUT_DIR / "roof_rmse_results.csv"

# Optional folder used to restrict which buildings are processed.
# If None, all matching GeoJSON + LAZ + GeoTIFF triplets are processed.
PROCESS_ONLY_FROM_DIR = None

# If True, scripts/run_preprocessing.py also crops the main LiDAR file
# into per-building LAZ files using geo_metadata.json.
RUN_LIDAR_CROP_IN_PREPROCESSING = True
