# -*- coding: utf-8 -*-

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from lod2_building_pipeline.config_loader import load_config
from lod2_building_pipeline.preprocessing import heat_data_creator


def main():
    cfg = load_config()

    # Override hardcoded paths inside heat_data_creator.py using config values.
    heat_data_creator.SATELLITE_TIF = str(cfg.SATELLITE_TIF)
    heat_data_creator.OUTPUT_BASE = str(cfg.PREPROCESS_OUTPUT)

    print("=" * 80)
    print("Running preprocessing stage")
    print("=" * 80)
    print("Satellite input   :", heat_data_creator.SATELLITE_TIF)
    print("Preprocess output :", heat_data_creator.OUTPUT_BASE)
    print("=" * 80)

    heat_data_creator.run()


if __name__ == "__main__":
    main()
