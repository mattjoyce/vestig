"""Configuration loading using loaden package"""

from pathlib import Path
from typing import Any

from loaden import load_config as loaden_load_config

from vestig.core.entity_ontology import EntityOntology


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    Load configuration from YAML file with loaden.

    Supports:
    - Recursive includes via "loaden_include" key
    - Environment variable expansion: ${VAR} or ${VAR:-default}
    - .env file loading
    - Deep merging of included files

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required configuration keys are missing
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please create a config.yaml file or specify path with --config"
        )

    # Load config with loaden
    # Loaden will:
    # - Process loaden_include directives
    # - Expand ${VAR} environment variables
    # - Load .env file if present
    # - Perform deep merging
    config = loaden_load_config(
        str(path),
        required_keys=[
            "storage.falkordb.host",
            "storage.falkordb.port",
            "storage.falkordb.graph_name",
        ],
    )

    return config


def load_entity_ontology(config: dict) -> EntityOntology:
    """
    Load entity ontology from config.

    Args:
        config: Full configuration dictionary

    Returns:
        EntityOntology instance

    Raises:
        ValueError: If m4.entity_types config is missing or malformed
    """
    if "m4" not in config:
        raise ValueError("Config missing m4 section (required for entity ontology)")

    return EntityOntology.from_config(config["m4"])
