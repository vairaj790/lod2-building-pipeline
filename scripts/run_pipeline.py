# -*- coding: utf-8 -*-

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from lod2_building_pipeline.config_loader import load_config


# Set True only when you want to regenerate crops from the satellite image.
RUN_PREPROCESSING = False

RUN_PREPARE_HEAT_INPUT = True
RUN_SETUP_HEAT_OPS = True
RUN_HEAT_INFERENCE = True


def run_script(script_relative_path: str):
    script_path = PROJECT_ROOT / script_relative_path

    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print("=" * 80)
    print(f"Running: {script_relative_path}")
    print("=" * 80)

    subprocess.run([sys.executable, str(script_path)], check=True)


def main():
    cfg = load_config()

    print("=" * 80)
    print("LOD2 Building Pipeline - preprocessing + HEAT inference")
    print("=" * 80)
    print("Project root      :", cfg.PROJECT_ROOT)
    print("Preprocess output :", cfg.PREPROCESS_OUTPUT)
    print("HEAT input        :", cfg.HEAT_INPUT_DIR)
    print("HEAT output       :", cfg.HEAT_OUTPUT_DIR)
    print("=" * 80)

    if RUN_PREPROCESSING:
        run_script("scripts/run_preprocessing.py")

    if RUN_PREPARE_HEAT_INPUT:
        run_script("scripts/prepare_heat_input.py")

    if RUN_SETUP_HEAT_OPS:
        run_script("scripts/setup_heat_ops_from_local.py")

    if RUN_HEAT_INFERENCE:
        run_script("scripts/run_heat.py")

    print("=" * 80)
    print("Preprocessing + HEAT inference workflow finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
