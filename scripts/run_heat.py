# -*- coding: utf-8 -*-

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from lod2_building_pipeline.config_loader import load_config


CONTAINER_HEAT_DIR = Path("/opt/heat")


def build_singularity_command(cfg, heat_code_dir, heat_input_dir, heat_output_dir, checkpoints_dir):
    singularity_image = Path(cfg.SINGULARITY_IMAGE)

    if not singularity_image.exists():
        raise FileNotFoundError(f"Missing Singularity image: {singularity_image}")

    return [
        "singularity",
        "exec",
        "--cleanenv",
        "--nv",
        "-B", f"{heat_code_dir}:{CONTAINER_HEAT_DIR}",
        "-B", f"{heat_input_dir}:{CONTAINER_HEAT_DIR / 'data/outdoor/cities_dataset'}",
        "-B", f"{heat_output_dir}:{CONTAINER_HEAT_DIR / 'results'}",
        "-B", f"{checkpoints_dir}:{CONTAINER_HEAT_DIR / 'checkpoints'}",
        "-B", "/cluster:/cluster",
        str(singularity_image),
        "bash",
        str(CONTAINER_HEAT_DIR / "start_infer.sh"),
    ]


def build_docker_command(cfg, heat_code_dir, heat_input_dir, heat_output_dir, checkpoints_dir):
    raise NotImplementedError(
        "Docker backend is planned but not implemented yet. "
        "Use CONTAINER_BACKEND='singularity' for now."
    )


def main():
    cfg = load_config()

    backend = getattr(cfg, "CONTAINER_BACKEND", "singularity").lower()

    heat_code_dir = Path(cfg.PROJECT_ROOT) / "third_party" / "heat"
    heat_input_dir = Path(cfg.HEAT_INPUT_DIR)
    heat_output_dir = Path(cfg.HEAT_OUTPUT_DIR)
    checkpoints_dir = Path(cfg.CHECKPOINTS_DIR)

    if not heat_code_dir.exists():
        raise FileNotFoundError(f"Missing HEAT code folder: {heat_code_dir}")

    if not heat_input_dir.exists():
        raise FileNotFoundError(f"Missing HEAT input folder: {heat_input_dir}")

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints folder: {checkpoints_dir}")

    heat_output_dir.mkdir(parents=True, exist_ok=True)

    if backend == "singularity":
        cmd = build_singularity_command(cfg, heat_code_dir, heat_input_dir, heat_output_dir, checkpoints_dir)
    elif backend == "docker":
        cmd = build_docker_command(cfg, heat_code_dir, heat_input_dir, heat_output_dir, checkpoints_dir)
    else:
        raise ValueError(f"Unsupported CONTAINER_BACKEND: {backend}")

    print("=" * 80)
    print("Running HEAT inference from clean pipeline repo")
    print("=" * 80)
    print("HEAT code   :", heat_code_dir)
    print("HEAT input  :", heat_input_dir)
    print("HEAT output :", heat_output_dir)
    print("Checkpoints :", checkpoints_dir)
    print("Backend     :", backend)
    print("=" * 80)

    subprocess.run(cmd, check=True)

    print("HEAT inference finished.")


if __name__ == "__main__":
    main()
