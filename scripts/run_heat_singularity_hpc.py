# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path


PROJECT_ROOT = Path("/home/fo37nor/assets/lod2_building_pipeline")

HEAT_CODE_DIR = PROJECT_ROOT / "third_party" / "heat"
HEAT_INPUT_DIR = PROJECT_ROOT / "work" / "heat_input"
HEAT_OUTPUT_DIR = PROJECT_ROOT / "work" / "heat_output"

CHECKPOINTS_DIR = Path("/home/fo37nor/assets/heat/checkpoints")
SINGULARITY_IMAGE = Path("/home/fo37nor/assets/heat/heat_with_tensorboard.sif")

CONTAINER_HEAT_DIR = Path("/opt/heat")


def main():
    if not HEAT_CODE_DIR.exists():
        raise FileNotFoundError(f"Missing HEAT code folder: {HEAT_CODE_DIR}")

    if not HEAT_INPUT_DIR.exists():
        raise FileNotFoundError(f"Missing HEAT input folder: {HEAT_INPUT_DIR}")

    if not CHECKPOINTS_DIR.exists():
        raise FileNotFoundError(f"Missing checkpoints folder: {CHECKPOINTS_DIR}")

    if not SINGULARITY_IMAGE.exists():
        raise FileNotFoundError(f"Missing Singularity image: {SINGULARITY_IMAGE}")

    HEAT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "singularity",
        "exec",
        "--cleanenv",
        "--nv",
        "-B", f"{HEAT_CODE_DIR}:{CONTAINER_HEAT_DIR}",
        "-B", f"{HEAT_INPUT_DIR}:{CONTAINER_HEAT_DIR / 'data/outdoor/cities_dataset'}",
        "-B", f"{HEAT_OUTPUT_DIR}:{CONTAINER_HEAT_DIR / 'results'}",
        "-B", f"{CHECKPOINTS_DIR}:{CONTAINER_HEAT_DIR / 'checkpoints'}",
        "-B", "/cluster:/cluster",
        str(SINGULARITY_IMAGE),
        "bash",
        str(CONTAINER_HEAT_DIR / "start_infer.sh"),
    ]

    print("=" * 80)
    print("Running HEAT inference from clean pipeline repo")
    print("=" * 80)
    print("HEAT code   :", HEAT_CODE_DIR)
    print("HEAT input  :", HEAT_INPUT_DIR)
    print("HEAT output :", HEAT_OUTPUT_DIR)
    print("Checkpoints :", CHECKPOINTS_DIR)
    print("=" * 80)

    subprocess.run(cmd, check=True)

    print("HEAT inference finished.")


if __name__ == "__main__":
    main()
