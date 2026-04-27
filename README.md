# LoD2 Building Pipeline

This repository is being developed as a reproducible pipeline for generating 2D roof skeletons from satellite imagery and preparing them for downstream LoD2 building reconstruction.

Current implemented stages:

1. Preprocessing satellite imagery into HEAT-compatible image crops
2. Running HEAT inference through Docker or Singularity
3. Saving HEAT outputs into a structured working folder

## Repository structure

    configs/
      config.py              Public configuration file
      config_local.py        Private local override, ignored by Git

    scripts/
      run_preprocessing.py   Run satellite preprocessing
      prepare_heat_input.py  Prepare HEAT-compatible input folder
      build_heat_ops.py      Build HEAT CUDA extension
      setup_heat_ops_from_local.py
                             Optional helper to copy an existing local HEAT ops binary
      run_heat.py            Run HEAT inference
      run_pipeline.py        Run preprocessing/input-prep/HEAT workflow

    src/
      lod2_building_pipeline/
        preprocessing/

    third_party/
      heat/                  Cleaned HEAT code used by the pipeline

    work/                    Local outputs, ignored by Git

## Setup

Copy the public config and edit it for your machine:

    cp configs/config.py configs/config_local.py

Edit:

    configs/config_local.py

Set paths such as:

    SATELLITE_TIF
    PREPROCESS_OUTPUT
    SINGULARITY_IMAGE
    CHECKPOINTS_DIR
    CONTAINER_BACKEND

`config_local.py` is ignored by Git and should contain your private/local paths.

## HEAT runtime

The HEAT backend can be run through:

    CONTAINER_BACKEND = "singularity"

or:

    CONTAINER_BACKEND = "docker"

The Docker image used by default is:

    vaibhavrajan79/heat_deformable-detr-image:with_tensorboard

## Build HEAT CUDA ops

The compiled file `MultiScaleDeformableAttention*.so` is not committed because it is machine-specific.

Build it with:

    python scripts/build_heat_ops.py

If you already have a compatible compiled `.so`, you can copy it locally using:

    python scripts/setup_heat_ops_from_local.py

## Run pipeline

Prepare HEAT input:

    python scripts/prepare_heat_input.py

Run HEAT inference:

    python scripts/run_heat.py

Or run the combined workflow:

    python scripts/run_pipeline.py

## Notes

Large data, checkpoints, container images, compiled binaries, and outputs are intentionally not committed.
