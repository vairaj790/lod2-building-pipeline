# LoD2 Building Pipeline

This repository is a pipeline for generating georeferenced 2D roof skeletons from satellite imagery and fusing them with LiDAR for downstream LoD2 building reconstruction.

The pipeline takes satellite imagery and LiDAR point-cloud data as input and produces georeferenced roof skeletons and interactive LoD2 building reconstructions. In general, the workflow first identifies building-like regions from satellite imagery, extracts 2D roof structures, restores their map coordinates, and then combines them with LiDAR height information to generate 3D building geometry.

The main external models used in this pipeline are:

- **SAM 3**: used during preprocessing to segment building/roof-like regions from satellite imagery and generate per-building image crops.
- **HEAT**: used to predict 2D roof skeletons from the cropped building images.

Current implemented stages:

1. **Preprocessing**: Cropping satellite imagery into HEAT-compatible per-building crops
2. **Preprocessing**: Cropping a LiDAR point file into per-building crops
3. **HEAT**: Running HEAT inference through Docker or Singularity
4. **Postprocessing**: Reassigning georeferencing to HEAT outputs
5. **Main LoD2 Generation**: Interactive 2D skeleton correction and LiDAR-based LoD2 reconstruction

## Repository structure

    configs/
      config.py                         Configuration file

    environments/
      environment_pipeline.yml          Environment for preprocessing and postprocessing
      environment_interactive_lod2.yml  Environment for interactive LoD2 reconstruction

    scripts/
      run_preprocessing.py              Run satellite preprocessing using SAM3 and LiDAR cropping
      prepare_heat_input.py             Prepare HEAT-compatible input folder
      build_heat_ops.py                 Build HEAT CUDA extension
      run_heat.py                       Run HEAT inference
      run_postprocessing.py             Reassign georeferencing to HEAT outputs
      run_interactive_lod2.py           Run interactive 2D/3D LoD2 reconstruction
      run_pipeline.py                   Run preprocessing + HEAT + postprocessing workflow

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

Place the input files inside:

    input/

The folder should contain exactly one satellite GeoTIFF and exactly one LiDAR LAZ file, for example:

    input/
      your_satellite_image.tif
      your_lidar_file.laz

The file names can be different. The config automatically searches for:

    *.tif / *.tiff
    *.laz

Edit the config file if needed:

    configs/config.py

Important config values include:

    CONTAINER_BACKEND
    SINGULARITY_IMAGE
    DOCKER_IMAGE
    CHECKPOINTS_DIR
    CHECKPOINT_RELATIVE_PATH

## HEAT runtime

The HEAT backend can be run through:

    CONTAINER_BACKEND = "singularity"

or:

    CONTAINER_BACKEND = "docker"

The Docker image used by default is:

    vaibhavrajan79/heat_deformable-detr-image:with_tensorboard

## HEAT checkpoint

The HEAT checkpoint is not committed to this repository.

Download the fine-tuned checkpoint folder from:

    https://cloud.uni-jena.de/public.php/dav/files/eftr8LboGGoSSEQ/?accept=zip

Place or extract it so that the final structure is:

    third_party/heat/checkpoints/
      heat_checkpoint_finetuned_512/
        checkpoint_best.pth

The default config expects:

    CHECKPOINTS_DIR = PROJECT_ROOT / "third_party" / "heat" / "checkpoints"
    CHECKPOINT_RELATIVE_PATH = Path("heat_checkpoint_finetuned_512/checkpoint_best.pth")

If downloading from the terminal, one possible workflow is:

    mkdir -p third_party/heat/checkpoints
    cd third_party/heat/checkpoints
    wget -O heat_checkpoint_finetuned_512.zip "https://cloud.uni-jena.de/public.php/dav/files/eftr8LboGGoSSEQ/?accept=zip"
    unzip heat_checkpoint_finetuned_512.zip

After extraction, check that `checkpoint_best.pth` is inside:

    third_party/heat/checkpoints/heat_checkpoint_finetuned_512/

## Build HEAT CUDA ops

The compiled file `MultiScaleDeformableAttention*.so` is not committed because it is machine-specific.

Build it with:

    python scripts/build_heat_ops.py

## Run main pipeline

Run preprocessing. This creates HEAT per-building crops, `geo_metadata.json`, and per-building LiDAR crops:

    python scripts/run_preprocessing.py

Prepare HEAT input:

    python scripts/prepare_heat_input.py

Run HEAT inference:

    python scripts/run_heat.py

Reassign georeferencing to HEAT outputs:

    python scripts/run_postprocessing.py

Or run the combined workflow:

    python scripts/run_pipeline.py  # preprocessing + HEAT + postprocessing

## Interactive LoD2 reconstruction

The final LoD2 reconstruction stage is interactive. It opens 2D and 3D windows for correcting roof skeletons, validating geometry, and fusing roof/base heights from LiDAR. It uses the georeferenced HEAT output produced in the previous stage together with the cropped LiDAR files to generate 3D building models.

This stage requires a graphical Python session.

Create the interactive environment:

    conda env create -f environments/environment_interactive_lod2.yml
    conda activate lod2_interactive

Run:

    python scripts/run_interactive_lod2.py

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

## Acknowledgements and citations

This pipeline builds on external research code and models. If you use this repository, please also cite the original works listed below.

### SAM 3

This project uses SAM 3 for segmentation-based preprocessing.

Repository:

    https://github.com/facebookresearch/sam3

BibTeX:

    @misc{carion2025sam3segmentconcepts,
          title={SAM 3: Segment Anything with Concepts},
          author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
          year={2025},
          eprint={2511.16719},
          archivePrefix={arXiv},
          primaryClass={cs.CV},
          url={https://arxiv.org/abs/2511.16719},
    }

### HEAT

This project uses a modified version of HEAT for 2D roof skeleton inference.

Repository:

    https://github.com/woodfrog/heat

BibTeX:

    @inproceedings{chen2022heat,
         title={HEAT: Holistic Edge Attention Transformer for Structured Reconstruction},
         author={Chen, Jiacheng and Qian, Yiming and Furukawa, Yasutaka},
         booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
         year={2022}
    }
