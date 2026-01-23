#!/usr/bin/env python3
"""Test embedding engine (Issue #9: Phase 1.2)

Tests for embeddings.py - dimension validation, truncation, consistency.
Requires llm CLI with embedding model available.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vestig.core.config import load_config
from vestig.core.embeddings import EmbeddingEngine, get_llm_model_provider


def llm_available() -> bool:
    """Check if llm CLI is available."""
    try:
        result = subprocess.run(["llm", "--version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_test_config():
    """Load test config for embedding settings."""
    return load_config("config_test.yaml")


# Skip all tests if llm CLI not available
pytestmark = pytest.mark.skipif(not llm_available(), reason="llm CLI not available")


class TestGetLlmModelProvider:
    """Test the get_llm_model_provider helper function."""

    def test_returns_dict_with_required_keys(self):
        """Test that result has provider_name and location keys."""
        result = get_llm_model_provider("some-model", model_type="embedding")
        assert "provider_name" in result
        assert "location" in result

    def test_unknown_model_returns_unknown(self):
        """Test that unknown model returns Unknown provider."""
        result = get_llm_model_provider("nonexistent-model-xyz", model_type="embedding")
        # Should return something, not crash
        assert isinstance(result["provider_name"], str)
        assert isinstance(result["location"], str)


class TestEmbeddingEngineInit:
    """Test EmbeddingEngine initialization."""

    def test_init_with_valid_config(self):
        """Test initialization with valid model and dimension."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
        )
        assert engine.model_name == config["embedding"]["model"]
        assert engine.expected_dimension == config["embedding"]["dimension"]
        assert engine.provider == "llm"

    def test_init_validates_dimension(self):
        """Test that wrong dimension raises ValueError."""
        config = get_test_config()
        with pytest.raises(ValueError, match="dimension mismatch"):
            EmbeddingEngine(
                model_name=config["embedding"]["model"],
                expected_dimension=999,  # Wrong dimension
            )

    def test_init_unknown_provider_raises(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            EmbeddingEngine(
                model_name="test-model",
                expected_dimension=768,
                provider="unknown_provider",
            )

    def test_init_with_timeout(self):
        """Test initialization with custom timeout."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
            timeout=120,
        )
        assert engine.timeout == 120

    def test_init_with_max_length(self):
        """Test initialization with max_length for truncation."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
            max_length=512,
        )
        assert engine.max_length == 512


class TestEmbeddingDimension:
    """Test embedding dimension correctness."""

    def test_embed_text_returns_correct_dimension(self):
        """Test that embed_text returns expected dimension."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
        )
        embedding = engine.embed_text("test text")
        assert len(embedding) == config["embedding"]["dimension"]

    def test_embed_text_returns_list_of_floats(self):
        """Test that embedding is a list of floats."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
        )
        embedding = engine.embed_text("test text")
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)


class TestEmbeddingConsistency:
    """Test embedding consistency and determinism."""

    def test_same_text_same_embedding(self):
        """Test that same text produces same embedding."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
        )
        text = "The quick brown fox"
        emb1 = engine.embed_text(text)
        emb2 = engine.embed_text(text)

        # Embeddings should be identical for same input
        assert emb1 == emb2

    def test_different_text_different_embedding(self):
        """Test that different text produces different embeddings."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
        )
        emb1 = engine.embed_text("Python programming")
        emb2 = engine.embed_text("Cooking recipes")

        # Embeddings should differ
        assert emb1 != emb2


class TestTruncation:
    """Test text truncation behavior."""

    def test_truncation_applied(self):
        """Test that long text is truncated at max_length."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
            max_length=50,
        )
        # Text longer than max_length
        long_text = "A" * 100
        # Should not raise - truncation handles it
        embedding = engine.embed_text(long_text)
        assert len(embedding) == config["embedding"]["dimension"]

    def test_no_truncation_when_no_limit(self):
        """Test that text is not truncated when max_length is None."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
            max_length=None,
        )
        text = "Normal length text"
        embedding = engine.embed_text(text)
        assert len(embedding) == config["embedding"]["dimension"]

    def test_truncation_affects_embedding(self):
        """Test that truncated text produces different embedding than full text."""
        config = get_test_config()
        # Engine with truncation
        engine_truncated = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
            max_length=10,
        )
        # Engine without truncation
        engine_full = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
            max_length=None,
        )
        # Text that will be truncated
        text = "The quick brown fox jumps over the lazy dog"

        emb_truncated = engine_truncated.embed_text(text)
        emb_full = engine_full.embed_text(text)

        # Should produce different embeddings due to truncation
        assert emb_truncated != emb_full


class TestBatchEmbedding:
    """Test batch embedding generation."""

    def test_embed_batch_returns_list(self):
        """Test that embed_batch returns a list of embeddings."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
        )
        texts = ["First text", "Second text", "Third text"]
        embeddings = engine.embed_batch(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3

    def test_embed_batch_correct_dimensions(self):
        """Test that all batch embeddings have correct dimension."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
        )
        texts = ["Text one", "Text two"]
        embeddings = engine.embed_batch(texts)

        for emb in embeddings:
            assert len(emb) == config["embedding"]["dimension"]
            assert all(isinstance(x, float) for x in emb)

    def test_embed_batch_empty_list(self):
        """Test that empty list returns empty list."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
        )
        embeddings = engine.embed_batch([])
        assert embeddings == []

    def test_embed_batch_matches_individual(self):
        """Test that batch embeddings match individual embeddings."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
        )
        texts = ["Hello world", "Goodbye world"]

        batch_embeddings = engine.embed_batch(texts)
        individual_embeddings = [engine.embed_text(t) for t in texts]

        # Should match
        for batch, individual in zip(batch_embeddings, individual_embeddings):
            assert batch == individual


class TestProviderInfo:
    """Test get_provider_info method."""

    def test_provider_info_returns_dict(self):
        """Test that get_provider_info returns dict with required keys."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
        )
        info = engine.get_provider_info()

        assert isinstance(info, dict)
        assert "provider_name" in info
        assert "location" in info

    def test_provider_info_location_valid(self):
        """Test that location is a valid value."""
        config = get_test_config()
        engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
        )
        info = engine.get_provider_info()

        # Location should be one of these values
        valid_locations = {"local", "remote", "unknown"}
        assert info["location"] in valid_locations
