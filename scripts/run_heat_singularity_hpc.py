# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path


HEAT_HOST_DIR = Path("/home/fo37nor/assets/heat")
HEAT_CONTAINER_DIR = Path("/opt/heat")

SINGULARITY_IMAGE = Path("/home/fo37nor/assets/heat/heat_with_tensorboard.sif")

START_INFER_SCRIPT = HEAT_CONTAINER_DIR / "start_infer.sh"


def main():
    if not HEAT_HOST_DIR.exists():
        raise FileNotFoundError(f"HEAT_HOST_DIR not found: {HEAT_HOST_DIR}")

    if not SINGULARITY_IMAGE.exists():
        raise FileNotFoundError(f"SINGULARITY_IMAGE not found: {SINGULARITY_IMAGE}")

    host_start_script = HEAT_HOST_DIR / "start_infer.sh"
    if not host_start_script.exists():
        raise FileNotFoundError(f"start_infer.sh not found: {host_start_script}")

    cmd = [
        "singularity",
        "exec",
        "--cleanenv",
        "--nv",
        "-B", f"{HEAT_HOST_DIR}:{HEAT_CONTAINER_DIR}",
        "-B", "/cluster:/cluster",
        str(SINGULARITY_IMAGE),
        "bash",
        str(START_INFER_SCRIPT),
    ]

    print("=" * 80)
    print("Running HEAT inference with Singularity")
    print("=" * 80)
    print("HEAT host dir     :", HEAT_HOST_DIR)
    print("Container mount   :", f"{HEAT_HOST_DIR} -> {HEAT_CONTAINER_DIR}")
    print("Singularity image :", SINGULARITY_IMAGE)
    print("Start script      :", START_INFER_SCRIPT)
    print("=" * 80)

    subprocess.run(cmd, check=True)

    print("=" * 80)
    print("HEAT inference finished successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()
