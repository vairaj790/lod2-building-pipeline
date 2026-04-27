# -*- coding: utf-8 -*-

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from lod2_building_pipeline.config_loader import load_config
from lod2_building_pipeline.postprocessing.reassign_geoinfo import run_postprocessing


def main():
    cfg = load_config()

    heat_output_dir = Path(cfg.HEAT_OUTPUT_DIR)
    metadata_path = Path(cfg.HEAT_INPUT_DIR) / "geo_metadata.json"
    output_dir = Path(cfg.POSTPROCESS_OUTPUT_DIR)

    patch_size = int(cfg.IMAGE_SIZE)
    result_name = str(cfg.HEAT_RESULT_NAME)

    run_postprocessing(
        heat_output_dir=heat_output_dir,
        metadata_path=metadata_path,
        output_dir=output_dir,
        patch_size=patch_size,
        result_name=result_name,
    )


if __name__ == "__main__":
    main()
