"""Configuration loading with minimal error handling"""

import os
from pathlib import Path
from typing import Any

import yaml

from vestig.core.entity_ontology import EntityOntology


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge overlay into base, with overlay taking precedence.

    Args:
        base: Base configuration dictionary
        overlay: Overlay configuration (overrides base)

    Returns:
        Merged configuration dictionary

    Examples:
        >>> base = {"a": 1, "b": {"c": 2}}
        >>> overlay = {"b": {"d": 3}, "e": 4}
        >>> deep_merge(base, overlay)
        {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
    """
    result = base.copy()

    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Override with overlay value
            result[key] = value

    return result


def load_config(
    config_path: str = "config.yaml",
    _skip_validation: bool = False,
    _include_stack: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load configuration from YAML file with include support.

    Supports recursive includes via "include" key:
        include: base.yaml
        include: [base.yaml, ontology.yaml]

    Included files are merged in order, with later files overriding earlier ones.
    The main config file always takes final precedence.

    Args:
        config_path: Path to config file
        _skip_validation: Internal parameter to skip validation for included files
        _include_stack: Internal parameter to detect circular includes

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If config file is empty/invalid or circular include detected
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please create a config.yaml file or specify path with --config"
        )

    # Detect circular includes
    if _include_stack is None:
        _include_stack = []

    resolved_path = str(path.resolve())
    if resolved_path in _include_stack:
        cycle = " -> ".join(_include_stack + [resolved_path])
        raise ValueError(f"Circular include detected: {cycle}")

    # Add to stack for cycle detection
    _include_stack.append(resolved_path)

    try:
        with open(path) as f:
            config = yaml.safe_load(f)

        # Guard against empty or non-dict configs
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ValueError(
                f"Invalid config file: {config_path}\n"
                f"Config must be a YAML dictionary, got {type(config).__name__}"
            )

        # Process includes recursively
        if "include" in config:
            includes = config.pop("include")
            if isinstance(includes, str):
                includes = [includes]

            # Build base config from all includes (in order)
            base_config: dict[str, Any] = {}
            for include_path in includes:
                # Resolve relative to current config file
                include_full = path.parent / include_path
                # Skip validation for included files (they may be partial configs)
                # Pass copy of include_stack to detect cycles in this branch
                included = load_config(
                    str(include_full),
                    _skip_validation=True,
                    _include_stack=_include_stack.copy(),
                )
                base_config = deep_merge(base_config, included)

            # Merge main config on top (main config takes precedence)
            config = deep_merge(base_config, config)
    finally:
        # Remove from stack after processing (allows diamond dependencies)
        if resolved_path in _include_stack:
            _include_stack.remove(resolved_path)

    # Only validate final merged config (not partial included files)
    if not _skip_validation:
        # Validate storage config exists
        if "storage" not in config:
            raise ValueError(f"Invalid config: missing storage section in {config_path}")

        # FalkorDB validation
        falkor_cfg = config["storage"].get("falkordb", {})
        for key in ["host", "port", "graph_name"]:
            if key not in falkor_cfg:
                raise ValueError(f"Invalid config: missing storage.falkordb.{key} in {config_path}")

        # Set up environment variables from config (if section exists)
        # User's shell environment takes precedence (don't override existing vars)
        if "env" in config:
            for key, value in config["env"].items():
                if key not in os.environ:
                    os.environ[key] = str(value)
                    # Note: We don't print here to avoid noise, but could add verbose flag

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
