# -*- coding: utf-8 -*-

import sys
import shutil
import glob
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from lod2_building_pipeline.config_loader import load_config


def main():
    cfg = load_config()

    source_dir = Path(cfg.LOCAL_HEAT_OPS_SOURCE_DIR)
    target_dir = Path(cfg.PROJECT_ROOT) / "third_party" / "heat" / "models" / "ops"

    so_files = glob.glob(str(source_dir / "MultiScaleDeformableAttention*.so"))

    if not so_files:
        raise FileNotFoundError(f"No compiled HEAT ops .so found in: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    for src in so_files:
        src_path = Path(src)
        dst_path = target_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {src_path} -> {dst_path}")

    print("Done. These .so files are local runtime artifacts and should not be committed.")


if __name__ == "__main__":
    main()
