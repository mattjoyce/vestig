"""Test config includes feature with loaden."""

import tempfile
from pathlib import Path

import pytest
import yaml

from vestig.core.config import load_config


def test_config_single_include():
    """Test loading config with single include file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create base config (minimal valid config)
        base_config = {
            "storage": {
                "falkordb": {
                    "host": "localhost",
                    "port": 6379,
                    "graph_name": "test_base",
                }
            },
            "embedding": {"model": "base_model", "dimension": 768},
        }
        (tmppath / "base.yaml").write_text(yaml.dump(base_config))

        # Create main config that includes base
        main_config = {
            "loaden_include": "base.yaml",
            "storage": {
                "falkordb": {
                    "graph_name": "test_main",  # Override
                }
            },
        }
        (tmppath / "main.yaml").write_text(yaml.dump(main_config))

        # Load and verify
        result = load_config(str(tmppath / "main.yaml"))

        assert result["storage"]["falkordb"]["host"] == "localhost"
        assert result["storage"]["falkordb"]["port"] == 6379
        assert result["storage"]["falkordb"]["graph_name"] == "test_main"  # Overridden
        assert result["embedding"]["model"] == "base_model"
        assert result["embedding"]["dimension"] == 768


def test_config_multiple_includes():
    """Test loading config with multiple include files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create storage config
        storage_config = {
            "storage": {
                "falkordb": {
                    "host": "localhost",
                    "port": 6379,
                    "graph_name": "test",
                }
            }
        }
        (tmppath / "storage.yaml").write_text(yaml.dump(storage_config))

        # Create embedding config
        embedding_config = {"embedding": {"model": "embeddinggemma", "dimension": 768}}
        (tmppath / "embedding.yaml").write_text(yaml.dump(embedding_config))

        # Create main config that includes both
        main_config = {
            "loaden_include": ["storage.yaml", "embedding.yaml"],
            "m4": {"entity_types": ["Person", "Place"]},
        }
        (tmppath / "main.yaml").write_text(yaml.dump(main_config))

        # Load and verify
        result = load_config(str(tmppath / "main.yaml"))

        assert result["storage"]["falkordb"]["host"] == "localhost"
        assert result["embedding"]["model"] == "embeddinggemma"
        assert result["m4"]["entity_types"] == ["Person", "Place"]


def test_config_nested_includes():
    """Test loading config with nested includes (include in include)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create base config
        base_config = {
            "storage": {
                "falkordb": {
                    "host": "localhost",
                    "port": 6379,
                    "graph_name": "base",
                }
            }
        }
        (tmppath / "base.yaml").write_text(yaml.dump(base_config))

        # Create middle config that includes base
        middle_config = {
            "loaden_include": "base.yaml",
            "embedding": {"model": "middle_model", "dimension": 768},
        }
        (tmppath / "middle.yaml").write_text(yaml.dump(middle_config))

        # Create main config that includes middle (which includes base)
        main_config = {
            "loaden_include": "middle.yaml",
            "storage": {
                "falkordb": {
                    "graph_name": "main",  # Override base
                }
            },
        }
        (tmppath / "main.yaml").write_text(yaml.dump(main_config))

        # Load and verify
        result = load_config(str(tmppath / "main.yaml"))

        assert result["storage"]["falkordb"]["host"] == "localhost"
        assert result["storage"]["falkordb"]["graph_name"] == "main"
        assert result["embedding"]["model"] == "middle_model"


def test_config_include_precedence():
    """Test that include order matters and main config has final precedence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create minimal storage config for validation
        storage_config = {
            "storage": {
                "falkordb": {
                    "host": "localhost",
                    "port": 6379,
                    "graph_name": "test",
                }
            }
        }
        (tmppath / "storage.yaml").write_text(yaml.dump(storage_config))

        # Create first config
        config1 = {"loaden_include": "storage.yaml", "value": 1, "nested": {"a": 1, "b": 1}}
        (tmppath / "config1.yaml").write_text(yaml.dump(config1))

        # Create second config
        config2 = {"loaden_include": "storage.yaml", "value": 2, "nested": {"a": 2, "c": 2}}
        (tmppath / "config2.yaml").write_text(yaml.dump(config2))

        # Create main config that includes both (order matters)
        main_config = {
            "loaden_include": ["config1.yaml", "config2.yaml"],
            "value": 3,  # Main overrides all
            "nested": {"a": 3},  # Partial override
        }
        (tmppath / "main.yaml").write_text(yaml.dump(main_config))

        # Load and verify
        result = load_config(str(tmppath / "main.yaml"))

        assert result["value"] == 3  # Main config wins
        assert result["nested"]["a"] == 3  # Main config wins
        assert result["nested"]["b"] == 1  # From config1
        assert result["nested"]["c"] == 2  # From config2


def test_config_include_file_not_found():
    """Test that missing include file raises clear error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create main config with non-existent include
        main_config = {"loaden_include": "nonexistent.yaml"}
        (tmppath / "main.yaml").write_text(yaml.dump(main_config))

        # Should raise FileNotFoundError with clear message
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config(str(tmppath / "main.yaml"))

        assert "nonexistent.yaml" in str(exc_info.value)


def test_config_empty_file():
    """Test that empty config file is handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create valid storage base
        storage_config = {
            "storage": {
                "falkordb": {
                    "host": "localhost",
                    "port": 6379,
                    "graph_name": "test",
                }
            }
        }
        (tmppath / "storage.yaml").write_text(yaml.dump(storage_config))

        # Create empty config file (yaml.safe_load returns None)
        (tmppath / "empty.yaml").write_text("")

        # Create main config that includes both
        main_config = {"loaden_include": ["storage.yaml", "empty.yaml"]}
        (tmppath / "main.yaml").write_text(yaml.dump(main_config))

        # Should load successfully (empty file treated as {})
        result = load_config(str(tmppath / "main.yaml"))
        assert result["storage"]["falkordb"]["host"] == "localhost"


def test_config_missing_required_key():
    """Test that missing required keys raise clear error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create config missing required falkordb.host
        incomplete_config = {
            "storage": {
                "falkordb": {
                    "port": 6379,
                    "graph_name": "test",
                }
            }
        }
        (tmppath / "incomplete.yaml").write_text(yaml.dump(incomplete_config))

        # Should raise ValueError about missing key
        with pytest.raises(ValueError) as exc_info:
            load_config(str(tmppath / "incomplete.yaml"))

        assert "storage.falkordb.host" in str(exc_info.value)


def test_config_env_variable_expansion():
    """Test that environment variable expansion works."""
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Set test environment variable
        os.environ["TEST_GRAPH_NAME"] = "test_from_env"

        # Create config with env var expansion
        config = {
            "storage": {
                "falkordb": {
                    "host": "localhost",
                    "port": 6379,
                    "graph_name": "${TEST_GRAPH_NAME}",
                }
            }
        }
        (tmppath / "config.yaml").write_text(yaml.dump(config))

        # Load and verify
        result = load_config(str(tmppath / "config.yaml"))
        assert result["storage"]["falkordb"]["graph_name"] == "test_from_env"

        # Clean up
        del os.environ["TEST_GRAPH_NAME"]


def test_config_env_variable_with_default():
    """Test that environment variable expansion with default works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create config with env var expansion and default
        config = {
            "storage": {
                "falkordb": {
                    "host": "${NONEXISTENT_HOST:-localhost}",
                    "port": 6379,
                    "graph_name": "test",
                }
            }
        }
        (tmppath / "config.yaml").write_text(yaml.dump(config))

        # Load and verify default is used
        result = load_config(str(tmppath / "config.yaml"))
        assert result["storage"]["falkordb"]["host"] == "localhost"
