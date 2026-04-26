# -*- coding: utf-8 -*-

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from lod2_building_pipeline.preprocessing import heat_data_creator


def main():
    print("=" * 80)
    print("Running preprocessing stage")
    print("=" * 80)

    heat_data_creator.run()


if __name__ == "__main__":
    main()
