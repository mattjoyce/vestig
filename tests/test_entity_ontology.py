#!/usr/bin/env python3
"""Test entity ontology (Issue #9: Phase 1.3)

Tests for entity_ontology.py - type matching, tier logic, config parsing.
No external dependencies required (pure Python).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vestig.core.entity_ontology import EntityOntology, EntityType


class TestEntityTypeModel:
    """Test EntityType dataclass."""

    def test_entity_type_creation(self):
        """Test basic EntityType creation with defaults."""
        et = EntityType(name="PERSON", description="Named individuals")
        assert et.name == "PERSON"
        assert et.description == "Named individuals"
        assert et.tier == 2  # default
        assert et.synonyms == []
        assert et.examples == []

    def test_entity_type_with_all_fields(self):
        """Test EntityType with all fields populated."""
        et = EntityType(
            name="ORG",
            description="Organizations and companies",
            tier=1,
            synonyms=["organization", "company", "corp"],
            examples=["Acme Inc", "Mozilla Foundation"],
        )
        assert et.name == "ORG"
        assert et.tier == 1
        assert "organization" in et.synonyms
        assert "Acme Inc" in et.examples


class TestEntityOntologyFromConfig:
    """Test EntityOntology.from_config() parsing."""

    def test_ontology_format_basic(self):
        """Test parsing new ontology format with required fields."""
        config = {
            "entity_types": {
                "ontology": [
                    {"name": "PERSON", "description": "Named individuals"},
                    {"name": "ORG", "description": "Organizations"},
                ]
            }
        }
        ontology = EntityOntology.from_config(config)
        assert len(ontology.types) == 2
        assert ontology.types[0].name == "PERSON"
        assert ontology.types[1].name == "ORG"

    def test_ontology_format_full(self):
        """Test parsing ontology format with all optional fields."""
        config = {
            "entity_types": {
                "ontology": [
                    {
                        "name": "system",  # lowercase - should normalize
                        "description": "Software systems",
                        "tier": 1,
                        "synonyms": ["platform", "service"],
                        "examples": ["PostgreSQL", "Redis"],
                    }
                ]
            }
        }
        ontology = EntityOntology.from_config(config)
        assert len(ontology.types) == 1
        et = ontology.types[0]
        assert et.name == "SYSTEM"  # normalized to uppercase
        assert et.tier == 1
        assert "platform" in et.synonyms
        assert "PostgreSQL" in et.examples

    def test_legacy_allowed_types_format(self):
        """Test parsing legacy allowed_types list format."""
        config = {"entity_types": {"allowed_types": ["PERSON", "ORG", "SYSTEM"]}}
        ontology = EntityOntology.from_config(config)
        assert len(ontology.types) == 3
        names = ontology.get_type_names()
        assert "PERSON" in names
        assert "ORG" in names
        assert "SYSTEM" in names

    def test_legacy_format_tier_inference(self):
        """Test that legacy format infers tiers correctly."""
        config = {"entity_types": {"allowed_types": ["PERSON", "TOOL", "CONCEPT"]}}
        ontology = EntityOntology.from_config(config)

        person = ontology.get_type("PERSON")
        assert person.tier == 1  # Tier 1 entity

        tool = ontology.get_type("TOOL")
        assert tool.tier == 2  # Tier 2 entity

        concept = ontology.get_type("CONCEPT")
        assert concept.tier == 3  # Tier 3 entity

    def test_legacy_format_normalizes_case(self):
        """Test that legacy format normalizes type names to uppercase."""
        config = {"entity_types": {"allowed_types": ["person", "Org", "SYSTEM"]}}
        ontology = EntityOntology.from_config(config)
        names = ontology.get_type_names()
        assert "PERSON" in names
        assert "ORG" in names
        assert "SYSTEM" in names

    def test_missing_entity_types_raises(self):
        """Test that missing both formats raises ValueError."""
        config = {"entity_types": {}}
        with pytest.raises(ValueError, match="must specify either"):
            EntityOntology.from_config(config)

    def test_ontology_not_list_raises(self):
        """Test that ontology must be a list."""
        config = {"entity_types": {"ontology": {"PERSON": "description"}}}
        with pytest.raises(ValueError, match="must be a list"):
            EntityOntology.from_config(config)

    def test_missing_name_raises(self):
        """Test that missing name field raises ValueError."""
        config = {"entity_types": {"ontology": [{"description": "No name here"}]}}
        with pytest.raises(ValueError, match="missing required 'name' field"):
            EntityOntology.from_config(config)

    def test_missing_description_raises(self):
        """Test that missing description field raises ValueError."""
        config = {"entity_types": {"ontology": [{"name": "PERSON"}]}}
        with pytest.raises(ValueError, match="missing required 'description' field"):
            EntityOntology.from_config(config)


class TestEntityOntologyMethods:
    """Test EntityOntology instance methods."""

    @pytest.fixture
    def ontology(self):
        """Sample ontology for testing methods."""
        config = {
            "entity_types": {
                "ontology": [
                    {"name": "PERSON", "description": "Named individuals", "tier": 1},
                    {"name": "ORG", "description": "Organizations", "tier": 1},
                    {"name": "TOOL", "description": "Software tools", "tier": 2},
                    {"name": "CONCEPT", "description": "Abstract concepts", "tier": 3},
                ]
            }
        }
        return EntityOntology.from_config(config)

    def test_get_type_names(self, ontology):
        """Test get_type_names returns all uppercase names."""
        names = ontology.get_type_names()
        assert names == ["PERSON", "ORG", "TOOL", "CONCEPT"]

    def test_get_types_by_tier(self, ontology):
        """Test filtering types by tier."""
        tier1 = ontology.get_types_by_tier(1)
        assert len(tier1) == 2
        tier1_names = [t.name for t in tier1]
        assert "PERSON" in tier1_names
        assert "ORG" in tier1_names

        tier2 = ontology.get_types_by_tier(2)
        assert len(tier2) == 1
        assert tier2[0].name == "TOOL"

        tier3 = ontology.get_types_by_tier(3)
        assert len(tier3) == 1
        assert tier3[0].name == "CONCEPT"

    def test_validate_type_exact_match(self, ontology):
        """Test validate_type with exact uppercase match."""
        assert ontology.validate_type("PERSON") is True
        assert ontology.validate_type("ORG") is True

    def test_validate_type_case_insensitive(self, ontology):
        """Test validate_type is case-insensitive."""
        assert ontology.validate_type("person") is True
        assert ontology.validate_type("Person") is True
        assert ontology.validate_type("PERSON") is True

    def test_validate_type_with_whitespace(self, ontology):
        """Test validate_type handles whitespace."""
        assert ontology.validate_type("  PERSON  ") is True
        assert ontology.validate_type("\tORG\n") is True

    def test_validate_type_unknown(self, ontology):
        """Test validate_type returns False for unknown types."""
        assert ontology.validate_type("UNKNOWN") is False
        assert ontology.validate_type("ANIMAL") is False
        assert ontology.validate_type("") is False

    def test_get_type_returns_entity_type(self, ontology):
        """Test get_type returns EntityType object."""
        person = ontology.get_type("PERSON")
        assert person is not None
        assert isinstance(person, EntityType)
        assert person.name == "PERSON"
        assert person.tier == 1

    def test_get_type_case_insensitive(self, ontology):
        """Test get_type is case-insensitive."""
        assert ontology.get_type("person") is not None
        assert ontology.get_type("Person") is not None
        assert ontology.get_type("PERSON") is not None
        # All return the same object
        assert ontology.get_type("person").name == "PERSON"

    def test_get_type_unknown_returns_none(self, ontology):
        """Test get_type returns None for unknown types."""
        assert ontology.get_type("UNKNOWN") is None
        assert ontology.get_type("ANIMAL") is None


class TestDefaultDescriptionsAndTiers:
    """Test default description and tier inference for legacy format."""

    def test_default_descriptions(self):
        """Test that legacy format gets default descriptions."""
        config = {"entity_types": {"allowed_types": ["PERSON", "ORG", "TOOL"]}}
        ontology = EntityOntology.from_config(config)

        person = ontology.get_type("PERSON")
        assert person.description == "Named individuals"

        org = ontology.get_type("ORG")
        assert "Organizations" in org.description

        tool = ontology.get_type("TOOL")
        assert "Software" in tool.description or "hardware" in tool.description.lower()

    def test_unknown_type_default_description(self):
        """Test that unknown types get a generic description."""
        config = {"entity_types": {"allowed_types": ["CUSTOM_TYPE"]}}
        ontology = EntityOntology.from_config(config)

        custom = ontology.get_type("CUSTOM_TYPE")
        assert "CUSTOM_TYPE" in custom.description

    def test_tier_assignment_comprehensive(self):
        """Test tier assignment for various entity types."""
        config = {
            "entity_types": {
                "allowed_types": [
                    "PERSON",
                    "ORG",
                    "SYSTEM",
                    "PROJECT",  # Tier 1
                    "TOOL",
                    "PLACE",
                    "SKILL",
                    "FILE",  # Tier 2
                    "CONCEPT",
                    "CAPABILITY",  # Tier 3
                ]
            }
        }
        ontology = EntityOntology.from_config(config)

        # Tier 1 entities
        for name in ["PERSON", "ORG", "SYSTEM", "PROJECT"]:
            assert ontology.get_type(name).tier == 1, f"{name} should be tier 1"

        # Tier 2 entities
        for name in ["TOOL", "PLACE", "SKILL", "FILE"]:
            assert ontology.get_type(name).tier == 2, f"{name} should be tier 2"

        # Tier 3 entities
        for name in ["CONCEPT", "CAPABILITY"]:
            assert ontology.get_type(name).tier == 3, f"{name} should be tier 3"
