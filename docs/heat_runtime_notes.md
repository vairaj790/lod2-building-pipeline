# HEAT runtime notes

This project uses HEAT as the 2D roof skeleton extraction backend.

The public repository does not include:

- local machine paths
- checkpoints
- compiled CUDA extension files
- Singularity .sif images
- training or inference outputs

## Docker image

Example Docker image:

    docker pull vaibhavrajan79/heat_deformable-detr-image:with_tensorboard

## Running with Singularity on HPC

Create or provide a Singularity image, then set its path in:

    configs/config.py or configs/config_local.py

Example values inside configs/config.py or configs/config_local.py:

    SINGULARITY_IMAGE = Path("/path/to/heat_with_tensorboard.sif")
    CHECKPOINTS_DIR = Path("/path/to/checkpoints")

Then run:

    python scripts/prepare_heat_input.py
    python scripts/setup_heat_ops_from_local.py
    python scripts/run_heat.py

## HEAT CUDA ops

The compiled file:

    MultiScaleDeformableAttention*.so

is not committed to GitHub because it is machine-specific.

Build it inside your Docker/Singularity environment:

    cd third_party/heat/models/ops
    bash make.sh

Or, if you already have a compatible compiled .so locally, set this in configs/config.py or configs/config_local.py:

    LOCAL_HEAT_OPS_SOURCE_DIR = Path("/path/to/existing/heat/models/ops")

Then run:

    python scripts/setup_heat_ops_from_local.py
