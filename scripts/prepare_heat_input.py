# -*- coding: utf-8 -*-

import os
import sys
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from lod2_building_pipeline.config_loader import load_config


USE_SYMLINKS = True


def remove_existing(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        remove_existing(dst)

    if USE_SYMLINKS:
        os.symlink(src, dst)
    else:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def main():
    cfg = load_config()

    preprocess_output = Path(cfg.PREPROCESS_OUTPUT)
    image_size = int(cfg.IMAGE_SIZE)

    source_rgb_dir = preprocess_output / f"rgb_jpg_{image_size}"
    source_lists_dir = preprocess_output / "lists"

    heat_input_dir = Path(cfg.HEAT_INPUT_DIR)
    target_rgb_dir = heat_input_dir / "rgb"

    if not source_rgb_dir.exists():
        raise FileNotFoundError(f"Missing source RGB folder: {source_rgb_dir}")

    if not source_lists_dir.exists():
        raise FileNotFoundError(f"Missing source lists folder: {source_lists_dir}")

    heat_input_dir.mkdir(parents=True, exist_ok=True)

    link_or_copy(source_rgb_dir, target_rgb_dir)

    for list_name in ["all_list.txt", "train_list.txt", "valid_list.txt"]:
        src = source_lists_dir / list_name
        dst = heat_input_dir / list_name

        if not src.exists():
            raise FileNotFoundError(f"Missing list file: {src}")

        link_or_copy(src, dst)

    print("HEAT-compatible input prepared:")
    print("  HEAT input :", heat_input_dir)
    print("  RGB folder :", target_rgb_dir)
    print("  Image size :", image_size)


if __name__ == "__main__":
    main()
