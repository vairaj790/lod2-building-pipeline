# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import os


PROJECT_ROOT = Path("/home/fo37nor/assets/lod2_building_pipeline")

PREPROCESS_OUTPUT = Path("/home/fo37nor/assets/sam3/sam3_osm_heat_dataset")

SOURCE_RGB_DIR = PREPROCESS_OUTPUT / "rgb_jpg_512"
SOURCE_LISTS_DIR = PREPROCESS_OUTPUT / "lists"

HEAT_INPUT_DIR = PROJECT_ROOT / "work" / "heat_input"
TARGET_RGB_DIR = HEAT_INPUT_DIR / "rgb"


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
    if not SOURCE_RGB_DIR.exists():
        raise FileNotFoundError(f"Missing source RGB folder: {SOURCE_RGB_DIR}")

    if not SOURCE_LISTS_DIR.exists():
        raise FileNotFoundError(f"Missing source lists folder: {SOURCE_LISTS_DIR}")

    HEAT_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    link_or_copy(SOURCE_RGB_DIR, TARGET_RGB_DIR)

    for list_name in ["all_list.txt", "train_list.txt", "valid_list.txt"]:
        src = SOURCE_LISTS_DIR / list_name
        dst = HEAT_INPUT_DIR / list_name

        if not src.exists():
            raise FileNotFoundError(f"Missing list file: {src}")

        link_or_copy(src, dst)

    print("HEAT-compatible input prepared:")
    print(" ", HEAT_INPUT_DIR)
    print("RGB:")
    print(" ", TARGET_RGB_DIR)


if __name__ == "__main__":
    main()
