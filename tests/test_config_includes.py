"""Test config includes feature."""

import tempfile
from pathlib import Path

import pytest
import yaml

from vestig.core.config import deep_merge, load_config


def test_deep_merge_basic():
    """Test basic deep merge functionality."""
    base = {"a": 1, "b": {"c": 2}}
    overlay = {"b": {"d": 3}, "e": 4}
    result = deep_merge(base, overlay)

    assert result == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}


def test_deep_merge_override():
    """Test that overlay overrides base values."""
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    overlay = {"a": 10, "b": {"c": 20}}
    result = deep_merge(base, overlay)

    assert result == {"a": 10, "b": {"c": 20, "d": 3}}


def test_deep_merge_nested():
    """Test deep merge with multiple nesting levels."""
    base = {"a": {"b": {"c": 1}}}
    overlay = {"a": {"b": {"d": 2}, "e": 3}}
    result = deep_merge(base, overlay)

    assert result == {"a": {"b": {"c": 1, "d": 2}, "e": 3}}


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
            "include": "base.yaml",
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
            "include": ["storage.yaml", "embedding.yaml"],
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
            "include": "base.yaml",
            "embedding": {"model": "middle_model", "dimension": 768},
        }
        (tmppath / "middle.yaml").write_text(yaml.dump(middle_config))

        # Create main config that includes middle (which includes base)
        main_config = {
            "include": "middle.yaml",
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
        config1 = {"include": "storage.yaml", "value": 1, "nested": {"a": 1, "b": 1}}
        (tmppath / "config1.yaml").write_text(yaml.dump(config1))

        # Create second config
        config2 = {"include": "storage.yaml", "value": 2, "nested": {"a": 2, "c": 2}}
        (tmppath / "config2.yaml").write_text(yaml.dump(config2))

        # Create main config that includes both (order matters)
        main_config = {
            "include": ["config1.yaml", "config2.yaml"],
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
        main_config = {"include": "nonexistent.yaml"}
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
        main_config = {"include": ["storage.yaml", "empty.yaml"]}
        (tmppath / "main.yaml").write_text(yaml.dump(main_config))

        # Should load successfully (empty file treated as {})
        result = load_config(str(tmppath / "main.yaml"))
        assert result["storage"]["falkordb"]["host"] == "localhost"


def test_config_null_file():
    """Test that null config file is handled gracefully."""
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

        # Create null config file
        (tmppath / "null.yaml").write_text("null\n")

        # Create main config that includes both
        main_config = {"include": ["storage.yaml", "null.yaml"]}
        (tmppath / "main.yaml").write_text(yaml.dump(main_config))

        # Should load successfully (null treated as {})
        result = load_config(str(tmppath / "main.yaml"))
        assert result["storage"]["falkordb"]["host"] == "localhost"


def test_config_non_dict():
    """Test that non-dict config raises clear error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create config that's a list instead of dict
        (tmppath / "list.yaml").write_text("[1, 2, 3]\n")

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            load_config(str(tmppath / "list.yaml"))

        assert "must be a YAML dictionary" in str(exc_info.value)
        assert "list" in str(exc_info.value)


def test_config_circular_include_direct():
    """Test that direct circular include is detected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create config that includes itself
        circular_config = {
            "include": "circular.yaml",
            "storage": {
                "falkordb": {
                    "host": "localhost",
                    "port": 6379,
                    "graph_name": "test",
                }
            },
        }
        (tmppath / "circular.yaml").write_text(yaml.dump(circular_config))

        # Should raise ValueError about circular include
        with pytest.raises(ValueError) as exc_info:
            load_config(str(tmppath / "circular.yaml"))

        assert "Circular include detected" in str(exc_info.value)


def test_config_circular_include_indirect():
    """Test that indirect circular include (A->B->A) is detected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create A that includes B
        config_a = {"include": "b.yaml", "value": "A"}
        (tmppath / "a.yaml").write_text(yaml.dump(config_a))

        # Create B that includes A (circular!)
        config_b = {"include": "a.yaml", "value": "B"}
        (tmppath / "b.yaml").write_text(yaml.dump(config_b))

        # Should raise ValueError about circular include
        with pytest.raises(ValueError) as exc_info:
            load_config(str(tmppath / "a.yaml"))

        assert "Circular include detected" in str(exc_info.value)
