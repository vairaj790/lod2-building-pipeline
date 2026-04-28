# LoD2 Building Pipeline

This repository is being developed as a reproducible pipeline for generating 2D roof skeletons from satellite imagery and preparing them for downstream LoD2 building reconstruction.

Current implemented stages:

1. Preprocessing satellite imagery into HEAT-compatible image crops
2. Cropping a main LiDAR file into per-building `.laz` files using `geo_metadata.json`
3. Running HEAT inference through Docker or Singularity
4. Reassigning georeferencing to HEAT outputs
5. Interactive 2D skeleton correction and LiDAR-based LoD2 reconstruction

## Repository structure

    configs/
      config.py                         Configuration file

    environments/
      environment_pipeline.yml          Environment for preprocessing and postprocessing
      environment_interactive_lod2.yml  Environment for interactive LoD2 reconstruction

    scripts/
      run_preprocessing.py              Run satellite preprocessing and LiDAR cropping
      prepare_heat_input.py             Prepare HEAT-compatible input folder
      build_heat_ops.py                 Build HEAT CUDA extension
      run_heat.py                       Run HEAT inference
      run_postprocessing.py             Reassign georeferencing to HEAT outputs
      run_interactive_lod2.py           Run interactive 2D/3D LoD2 reconstruction
      run_pipeline.py                   Run preprocessing/input-prep/HEAT/postprocessing workflow

    src/
      lod2_building_pipeline/
        preprocessing/
        postprocessing/
        lod2/

    third_party/
      heat/                             Modified HEAT code used by the pipeline

## Setup

Create the main pipeline environment:

    conda env create -f environments/environment_pipeline.yml
    conda activate lod2_pipeline

Copy the config and edit it for your machine:

    cp configs/config.py configs/config_local.py

Edit:

    configs/config_local.py

Set paths such as:

    SATELLITE_TIF
    MAIN_LIDAR_PATH
    SINGULARITY_IMAGE
    CHECKPOINTS_DIR
    CONTAINER_BACKEND

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

## Run main pipeline

Run preprocessing. This creates HEAT image crops, `geo_metadata.json`, and per-building LiDAR crops:

    python scripts/run_preprocessing.py

Prepare HEAT input:

    python scripts/prepare_heat_input.py

Run HEAT inference:

    python scripts/run_heat.py

Reassign georeferencing to HEAT outputs:

    python scripts/run_postprocessing.py

Or run the combined workflow:

    python scripts/run_pipeline.py

The main generated folders are:

    work/heat_input/
    work/Lidar_input/
    work/heat_output/
    work/georeferenced_output/

## Interactive LoD2 reconstruction

The final LoD2 reconstruction stage is interactive. It opens 2D and 3D windows for correcting roof skeletons, validating geometry, and fusing roof/base heights from LiDAR. This stage requires a graphical Python session.

Create the interactive environment:

    conda env create -f environments/environment_interactive_lod2.yml
    conda activate lod2_interactive

Run:

    python scripts/run_interactive_lod2.py

This stage reads:

    work/georeferenced_output/geojson_files/
    work/Lidar_input/
    work/heat_input/rgb_tif_original/

and writes:

    work/3D_output/

### 2D DELETE mode controls

    Left click near geometry     Select nearest edge/vertex
    Left drag                    Rectangle-select vertices/edges
    E                            Delete selected edge(s)
    V                            Delete selected vertex/vertices and incident edges
    U                            Undo last delete
    A                            Accept/save updated 2D skeleton
    D                            Switch to DELETE mode
    T                            Toggle GeoTIFF overlay
    C                            Tag connected components
    3                            Build 3D result and open 3D validator
    Q                            Quit batch without saving
    Close window                 Quit batch without saving

### 2D EDIT mode controls

    Left click                   Add new vertex
    Right click                  Select point/line
    Right double click           Connect two selected points
    X                            Delete selected point/line
    U or Z                       Undo last edit action
    A                            Accept/save edited 2D skeleton
    D or E                       Switch mode

### 3D preview controls

    N                            Save and next
    P                            Save and previous
    R                            Redo this file without saving
    Q                            Quit batch without saving
    L                            Toggle LiDAR/debug overlays
    O                            Toggle previous red roof edges
    F                            Toggle fitted blue roof plane overlays
    G                            Toggle generated roof surfaces
    W                            Toggle wall surfaces
    S                            Save PNG snapshot
    Close window                 Quit without saving

## Notes

Large data, checkpoints, container images, compiled binaries, LiDAR files, and generated outputs are intentionally not committed.
