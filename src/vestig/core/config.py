"""Configuration loading with minimal error handling"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please create a config.yaml file or specify path with --config"
        )

    with open(path) as f:
        config = yaml.safe_load(f)

    # Validate db_path exists in config
    if "storage" not in config or "db_path" not in config["storage"]:
        raise ValueError(f"Invalid config: missing storage.db_path in {config_path}")

    return config
