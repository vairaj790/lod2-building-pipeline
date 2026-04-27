# -*- coding: utf-8 -*-

from pathlib import Path
import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"

LOCAL_CONFIG = CONFIGS_DIR / "config_local.py"
EXAMPLE_CONFIG = CONFIGS_DIR / "config_example.py"


def load_config():
    if LOCAL_CONFIG.exists():
        config_path = LOCAL_CONFIG
        print(f"Using local config: {config_path}")
    else:
        config_path = EXAMPLE_CONFIG
        print(f"Using example config: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("pipeline_config", config_path)
    config = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not load config: {config_path}")

    spec.loader.exec_module(config)
    return config
