# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import glob


SOURCE_DIR = Path("/home/fo37nor/assets/heat/models/ops")
TARGET_DIR = Path("/home/fo37nor/assets/lod2_building_pipeline/third_party/heat/models/ops")


def main():
    so_files = glob.glob(str(SOURCE_DIR / "MultiScaleDeformableAttention*.so"))

    if not so_files:
        raise FileNotFoundError(f"No compiled HEAT ops .so found in: {SOURCE_DIR}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for src in so_files:
        src_path = Path(src)
        dst_path = TARGET_DIR / src_path.name
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {src_path} -> {dst_path}")

    print("Done. These .so files are local runtime artifacts and should not be committed.")


if __name__ == "__main__":
    main()
