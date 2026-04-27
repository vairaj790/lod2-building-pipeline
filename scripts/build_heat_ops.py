# -*- coding: utf-8 -*-

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from lod2_building_pipeline.config_loader import load_config


CONTAINER_HEAT_DIR = Path("/opt/heat")


def build_singularity_command(cfg, heat_code_dir: Path):
    singularity_image = Path(cfg.SINGULARITY_IMAGE)

    if not singularity_image.exists():
        raise FileNotFoundError(f"Missing Singularity image: {singularity_image}")

    cmd = [
        "singularity",
        "exec",
        "--cleanenv",
        "--nv",
        "-B", f"{heat_code_dir}:{CONTAINER_HEAT_DIR}",
    ]

    for bind_path in getattr(cfg, "SINGULARITY_EXTRA_BINDS", []):
        cmd.extend(["-B", str(bind_path)])

    cmd.extend([
        str(singularity_image),
        "bash",
        "-lc",
        "cd /opt/heat/models/ops && bash make.sh",
    ])

    return cmd


def build_docker_command(cfg, heat_code_dir: Path):
    docker_image = str(cfg.DOCKER_IMAGE)

    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--shm-size=1g",
        "-v", f"{heat_code_dir}:{CONTAINER_HEAT_DIR}",
        docker_image,
        "bash",
        "-lc",
        "cd /opt/heat/models/ops && bash make.sh",
    ]


def main():
    cfg = load_config()

    backend = getattr(cfg, "CONTAINER_BACKEND", "singularity").lower()
    heat_code_dir = Path(cfg.PROJECT_ROOT) / "third_party" / "heat"

    if not heat_code_dir.exists():
        raise FileNotFoundError(f"Missing HEAT code folder: {heat_code_dir}")

    if backend == "singularity":
        cmd = build_singularity_command(cfg, heat_code_dir)
    elif backend == "docker":
        cmd = build_docker_command(cfg, heat_code_dir)
    else:
        raise ValueError(f"Unsupported CONTAINER_BACKEND: {backend}")

    print("=" * 80)
    print("Building HEAT CUDA ops")
    print("=" * 80)
    print("Backend   :", backend)
    print("HEAT code :", heat_code_dir)
    print("=" * 80)

    subprocess.run(cmd, check=True)

    print("=" * 80)
    print("HEAT CUDA ops build finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
