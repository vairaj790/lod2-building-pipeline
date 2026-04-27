# -*- coding: utf-8 -*-

import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from lod2_building_pipeline.config_loader import load_config


CONTAINER_HEAT_DIR = Path("/opt/heat")


def q(value) -> str:
    return shlex.quote(str(value))


def build_inner_heat_command(cfg) -> str:
    checkpoint_path = Path("./checkpoints") / Path(cfg.CHECKPOINT_RELATIVE_PATH)

    result_name = str(getattr(cfg, "HEAT_RESULT_NAME", "")).strip()

    if result_name:
        viz_base = Path("./results") / f"viz_{result_name}"
        save_base = Path("./results") / f"npy_{result_name}"
    else:
        viz_base = Path("./results") / "viz"
        save_base = Path("./results") / "npy"

    cmd = f"""
set -euo pipefail

export PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH=/opt/heat/models/ops:${{PYTHONPATH:-}}
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/torch/lib:${{LD_LIBRARY_PATH:-}}

cd /opt/heat

python -c "import MultiScaleDeformableAttention as m; print('MSDA OK:', m.__file__)"

CUDA_VISIBLE_DEVICES=0 python infer.py \\
  --checkpoint_path {q(checkpoint_path)} \\
  --dataset outdoor \\
  --image_size {int(cfg.IMAGE_SIZE)} \\
  --corner_thresh {float(cfg.CORNER_THRESH)} \\
  --edge_thresh {float(cfg.EDGE_THRESH)} \\
  --infer_times {int(cfg.INFER_TIMES)} \\
  --viz_base {q(viz_base)} \\
  --save_base {q(save_base)}
"""
    return cmd


def build_singularity_command(cfg, heat_code_dir, heat_input_dir, heat_output_dir, checkpoints_dir):
    singularity_image = Path(cfg.SINGULARITY_IMAGE)

    if not singularity_image.exists():
        raise FileNotFoundError(f"Missing Singularity image: {singularity_image}")

    cmd = [
        "singularity",
        "exec",
        "--cleanenv",
        "--nv",
        "-B", f"{heat_code_dir}:{CONTAINER_HEAT_DIR}",
        "-B", f"{heat_input_dir}:{CONTAINER_HEAT_DIR / 'data/outdoor/cities_dataset'}",
        "-B", f"{heat_output_dir}:{CONTAINER_HEAT_DIR / 'results'}",
        "-B", f"{checkpoints_dir}:{CONTAINER_HEAT_DIR / 'checkpoints'}",
    ]

    for bind_path in getattr(cfg, "SINGULARITY_EXTRA_BINDS", []):
        cmd.extend(["-B", str(bind_path)])

    cmd.extend([
        str(singularity_image),
        "bash",
        "-lc",
        build_inner_heat_command(cfg),
    ])

    return cmd


def build_docker_command(cfg, heat_code_dir, heat_input_dir, heat_output_dir, checkpoints_dir):
    docker_image = str(cfg.DOCKER_IMAGE)

    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--shm-size=1g",
        "-v", f"{heat_code_dir}:{CONTAINER_HEAT_DIR}",
        "-v", f"{heat_input_dir}:{CONTAINER_HEAT_DIR / 'data/outdoor/cities_dataset'}",
        "-v", f"{heat_output_dir}:{CONTAINER_HEAT_DIR / 'results'}",
        "-v", f"{checkpoints_dir}:{CONTAINER_HEAT_DIR / 'checkpoints'}",
        docker_image,
        "bash",
        "-lc",
        build_inner_heat_command(cfg),
    ]


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
        cmd = build_singularity_command(
            cfg, heat_code_dir, heat_input_dir, heat_output_dir, checkpoints_dir
        )
    elif backend == "docker":
        cmd = build_docker_command(
            cfg, heat_code_dir, heat_input_dir, heat_output_dir, checkpoints_dir
        )
    else:
        raise ValueError(f"Unsupported CONTAINER_BACKEND: {backend}")

    print("=" * 80)
    print("Running HEAT inference")
    print("=" * 80)
    print("Backend     :", backend)
    print("HEAT code   :", heat_code_dir)
    print("HEAT input  :", heat_input_dir)
    print("HEAT output :", heat_output_dir)
    print("Checkpoints :", checkpoints_dir)
    print("=" * 80)

    subprocess.run(cmd, check=True)

    print("HEAT inference finished.")


if __name__ == "__main__":
    main()
