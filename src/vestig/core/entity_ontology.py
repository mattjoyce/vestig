"""Entity ontology: configurable entity type definitions (M4: Graph Layer)"""

from dataclasses import dataclass, field


@dataclass
class EntityType:
    """Single entity type definition from config."""

    name: str  # UPPERCASE entity type name
    description: str  # What this entity represents
    tier: int = 2  # 1=always extract, 2=contextual, 3=high bar
    synonyms: list[str] = field(default_factory=list)  # Alternative terms
    examples: list[str] = field(default_factory=list)  # Example entities


@dataclass
class EntityOntology:
    """
    Entity type ontology loaded from config.

    Supports two config formats:
    1. New ontology format (m4.entity_types.ontology)
    2. Legacy allowed_types format (m4.entity_types.allowed_types)
    """

    types: list[EntityType]

    @classmethod
    def from_config(cls, config: dict) -> "EntityOntology":
        """
        Load entity ontology from m4.entity_types config section.

        Args:
            config: m4 section of config dict

        Returns:
            EntityOntology instance

        Raises:
            ValueError: If config is malformed or missing entity types
        """
        entity_types_config = config.get("entity_types", {})

        # New format: ontology list
        if "ontology" in entity_types_config:
            ontology_list = entity_types_config["ontology"]
            if not isinstance(ontology_list, list):
                raise ValueError(
                    "m4.entity_types.ontology must be a list of entity type definitions"
                )

            types = []
            for type_def in ontology_list:
                if not isinstance(type_def, dict):
                    raise ValueError(f"Invalid entity type definition: {type_def}")

                if "name" not in type_def:
                    raise ValueError(
                        f"Entity type definition missing required 'name' field: {type_def}"
                    )
                if "description" not in type_def:
                    raise ValueError(
                        f"Entity type definition missing required 'description' field: {type_def}"
                    )

                # Normalize name to uppercase
                name = type_def["name"].strip().upper()

                types.append(
                    EntityType(
                        name=name,
                        description=type_def["description"],
                        tier=type_def.get("tier", 2),
                        synonyms=type_def.get("synonyms", []),
                        examples=type_def.get("examples", []),
                    )
                )

            return cls(types=types)

        # Legacy format: allowed_types list (convert to default ontology)
        elif "allowed_types" in entity_types_config:
            allowed_types = entity_types_config["allowed_types"]
            if not isinstance(allowed_types, list):
                raise ValueError("m4.entity_types.allowed_types must be a list")

            # Convert to default ontology with minimal descriptions
            types = []
            for type_name in allowed_types:
                if not isinstance(type_name, str):
                    raise ValueError(f"Invalid entity type (must be string): {type_name}")

                # Normalize to uppercase
                normalized_name = type_name.strip().upper()

                # Default descriptions for common types
                description = cls._get_default_description(normalized_name)

                # Infer tier from type name
                tier = cls._get_default_tier(normalized_name)

                types.append(
                    EntityType(
                        name=normalized_name,
                        description=description,
                        tier=tier,
                        synonyms=[],
                        examples=[],
                    )
                )

            return cls(types=types)

        else:
            raise ValueError(
                "Config must specify either m4.entity_types.ontology or "
                "m4.entity_types.allowed_types"
            )

    @staticmethod
    def _get_default_description(type_name: str) -> str:
        """Get default description for legacy entity types."""
        defaults = {
            "PERSON": "Named individuals",
            "ORG": "Organizations, companies, institutions",
            "SYSTEM": "Software/hardware platforms, technical systems",
            "PROJECT": "Named initiatives, programs, projects",
            "TOOL": "Software, hardware, instruments, applications",
            "PLACE": "Locations, sites, facilities, geographic entities",
            "SKILL": "Specialized capabilities, methodologies, technical approaches",
            "FILE": "Named documents, artifacts, repositories",
            "CONCEPT": "Domain-specific compound terms representing specialized knowledge",
            "CAPABILITY": "System capabilities or features",
        }
        return defaults.get(type_name, f"Entity type: {type_name}")

    @staticmethod
    def _get_default_tier(type_name: str) -> int:
        """Get default tier for legacy entity types."""
        # Tier 1: High-value entities
        tier1 = {"PERSON", "ORG", "SYSTEM", "PROJECT"}
        # Tier 3: Use sparingly
        tier3 = {"CONCEPT", "CAPABILITY"}

        if type_name in tier1:
            return 1
        elif type_name in tier3:
            return 3
        else:
            return 2

    def get_type_names(self) -> list[str]:
        """
        Get list of entity type names (uppercase).

        Returns:
            List of type names
        """
        return [t.name for t in self.types]

    def get_types_by_tier(self, tier: int) -> list[EntityType]:
        """
        Get entity types in a specific tier.

        Args:
            tier: Tier number (1, 2, or 3)

        Returns:
            List of EntityType objects in the specified tier
        """
        return [t for t in self.types if t.tier == tier]

    def validate_type(self, type_name: str) -> bool:
        """
        Check if an entity type exists in the ontology.

        Args:
            type_name: Entity type name (case-insensitive)

        Returns:
            True if type exists in ontology, False otherwise
        """
        normalized = type_name.strip().upper()
        return normalized in self.get_type_names()

    def get_type(self, type_name: str) -> EntityType | None:
        """
        Get EntityType object by name.

        Args:
            type_name: Entity type name (case-insensitive)

        Returns:
            EntityType object if found, None otherwise
        """
        normalized = type_name.strip().upper()
        for entity_type in self.types:
            if entity_type.name == normalized:
                return entity_type
        return None
