# -*- coding: utf-8 -*-

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from lod2_building_pipeline.config_loader import load_config
from lod2_building_pipeline.preprocessing import heat_data_creator
from lod2_building_pipeline.preprocessing.crop_lidar_by_geo_metadata import (
    crop_lidar_by_geo_metadata,
)


def run_image_preprocessing(cfg):
    heat_data_creator.SATELLITE_TIF = str(cfg.SATELLITE_TIF)
    heat_data_creator.OUTPUT_BASE = str(cfg.PREPROCESS_OUTPUT)

    print("=" * 80)
    print("Running image preprocessing stage")
    print("=" * 80)
    print("Satellite input   :", heat_data_creator.SATELLITE_TIF)
    print("Preprocess output :", heat_data_creator.OUTPUT_BASE)
    print("=" * 80)

    heat_data_creator.run()


def run_lidar_preprocessing(cfg):
    main_lidar_path = Path(cfg.MAIN_LIDAR_PATH)
    geo_metadata_json = Path(cfg.HEAT_INPUT_DIR) / "geo_metadata.json"
    output_dir = Path(cfg.LIDAR_INPUT_DIR)

    print("=" * 80)
    print("Running LiDAR cropping stage")
    print("=" * 80)

    crop_lidar_by_geo_metadata(
        main_lidar_path=main_lidar_path,
        geo_metadata_json=geo_metadata_json,
        output_dir=output_dir,
        extra_buffer_m=float(cfg.LIDAR_EXTRA_BUFFER_M),
        points_per_chunk=int(cfg.LIDAR_POINTS_PER_CHUNK),
        save_merged_debug_file=bool(cfg.SAVE_MERGED_LIDAR_DEBUG_FILE),
        merged_debug_name=str(cfg.MERGED_LIDAR_DEBUG_NAME),
    )


def main():
    cfg = load_config()

    run_image_preprocessing(cfg)

    if bool(getattr(cfg, "RUN_LIDAR_CROP_IN_PREPROCESSING", False)):
        run_lidar_preprocessing(cfg)
    else:
        print("LiDAR cropping skipped because RUN_LIDAR_CROP_IN_PREPROCESSING=False")


if __name__ == "__main__":
    main()
