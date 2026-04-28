# -*- coding: utf-8 -*-

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from lod2_building_pipeline.config_loader import load_config
from lod2_building_pipeline.lod2 import interactive_lod2_reconstruction as lod2


def main():
    cfg = load_config()

    geojson_dir = Path(cfg.POSTPROCESS_OUTPUT_DIR) / "geojson_files"
    laz_dir = Path(cfg.LIDAR_INPUT_DIR)
    geotiff_dir = Path(cfg.HEAT_INPUT_DIR) / "rgb_tif_original"

    output_dir = Path(cfg.LOD2_3D_OUTPUT_DIR)
    snapshot_dir = Path(cfg.LOD2_SNAPSHOT_DIR)
    rmse_csv_path = Path(cfg.LOD2_RMSE_CSV_PATH)

    process_only_from_dir = getattr(cfg, "PROCESS_ONLY_FROM_DIR", None)
    if process_only_from_dir is not None:
        process_only_from_dir = str(Path(process_only_from_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    lod2.GEOJSON_DIR = str(geojson_dir)
    lod2.LAZ_DIR = str(laz_dir)
    lod2.GEOTIFF_DIR = str(geotiff_dir)
    lod2.OUTPUT_DIR = str(output_dir)
    lod2.SNAPSHOT_DIR = str(snapshot_dir)
    lod2.RMSE_CSV_PATH = str(rmse_csv_path)
    lod2.PROCESS_ONLY_FROM_DIR = process_only_from_dir

    print("=" * 80)
    print("Running interactive LoD2 reconstruction")
    print("=" * 80)
    print("GeoJSON input :", lod2.GEOJSON_DIR)
    print("LiDAR input   :", lod2.LAZ_DIR)
    print("GeoTIFF input :", lod2.GEOTIFF_DIR)
    print("3D output     :", lod2.OUTPUT_DIR)
    print("Snapshots     :", lod2.SNAPSHOT_DIR)
    print("RMSE CSV      :", lod2.RMSE_CSV_PATH)
    print("Process filter:", lod2.PROCESS_ONLY_FROM_DIR)
    print("=" * 80)

    lod2.main_batch()


if __name__ == "__main__":
    main()
