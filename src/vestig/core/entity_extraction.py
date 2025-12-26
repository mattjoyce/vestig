"""Entity extraction using LLM (M4: Graph Layer)"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


@dataclass
class ExtractedEntity:
    """Entity extracted by LLM"""

    name: str
    entity_type: str
    confidence: float
    evidence: str


# Pydantic schemas for LLM structured output
class EntitySchema(BaseModel):
    """Schema for a single entity"""

    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type (PERSON, ORG, SYSTEM, PROJECT, PLACE)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    evidence: str = Field(description="Evidence text supporting this entity")


class EntityExtractionResult(BaseModel):
    """Schema for entity extraction response"""

    entities: list[EntitySchema] = Field(description="List of extracted entities")


def load_prompts(prompts_path: str = "prompts.yaml") -> dict[str, str]:
    """Load prompts from YAML file"""
    path = Path(prompts_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    with open(path) as f:
        prompts = yaml.safe_load(f)

    return prompts


def substitute_tokens(template: str, **kwargs) -> str:
    """
    Substitute {{token}} placeholders in template.

    Args:
        template: Template string with {{token}} placeholders
        **kwargs: Token values

    Returns:
        Template with tokens substituted

    Example:
        >>> substitute_tokens("Hello {{name}}", name="Alice")
        'Hello Alice'
    """
    result = template
    for key, value in kwargs.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, str(value))
    return result


def call_llm(prompt: str, model: str, schema: type[BaseModel] | None = None):
    """
    Call LLM with prompt using llm module.

    Args:
        prompt: Full prompt text
        model: Model name (e.g., "claude-haiku-4.5")
        schema: Optional Pydantic model for structured output

    Returns:
        If schema provided: Pydantic model instance
        If no schema: str (response text)

    Raises:
        ImportError: If llm module not installed
        Exception: If LLM call fails
    """
    try:
        import llm
    except ImportError:
        raise ImportError("llm module not installed. Install with: pip install llm")

    try:
        # Get model and call it
        llm_model = llm.get_model(model)

        if schema:
            # Use structured output with schema
            response = llm_model.prompt(prompt, schema=schema)
            # Parse JSON response and validate with schema
            json_text = response.text()
            data = json.loads(json_text)
            return schema.model_validate(data)
        else:
            # Return raw text
            response = llm_model.prompt(prompt)
            return response.text()

    except Exception as e:
        raise Exception(f"LLM call failed: {e}")


def validate_extraction_result(
    result: dict[str, Any], allowed_types: list[str]
) -> list[ExtractedEntity]:
    """
    Validate and parse LLM extraction result.

    Args:
        result: Parsed JSON from LLM
        allowed_types: List of allowed entity types

    Returns:
        List of validated ExtractedEntity objects

    Raises:
        ValueError: If validation fails
    """
    if "entities" not in result:
        raise ValueError("Missing 'entities' key in extraction result")

    if not isinstance(result["entities"], list):
        raise ValueError("'entities' must be a list")

    entities = []

    for i, entity_dict in enumerate(result["entities"]):
        # Validate required fields
        if "name" not in entity_dict:
            raise ValueError(f"Entity {i}: missing 'name' field")
        if "type" not in entity_dict:
            raise ValueError(f"Entity {i}: missing 'type' field")
        if "confidence" not in entity_dict:
            raise ValueError(f"Entity {i}: missing 'confidence' field")
        if "evidence" not in entity_dict:
            raise ValueError(f"Entity {i}: missing 'evidence' field")

        # Validate types
        name = str(entity_dict["name"]).strip()
        entity_type = str(entity_dict["type"]).upper()
        evidence = str(entity_dict["evidence"]).strip()

        if not name:
            raise ValueError(f"Entity {i}: 'name' cannot be empty")

        if entity_type not in allowed_types:
            raise ValueError(f"Entity {i}: invalid type '{entity_type}'. Allowed: {allowed_types}")

        # Validate confidence
        try:
            confidence = float(entity_dict["confidence"])
        except (ValueError, TypeError):
            raise ValueError(
                f"Entity {i}: 'confidence' must be a float, got {type(entity_dict['confidence'])}"
            )

        if not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"Entity {i}: 'confidence' must be between 0.0 and 1.0, got {confidence}"
            )

        # Truncate evidence to max 200 chars
        if len(evidence) > 200:
            evidence = evidence[:197] + "..."

        entities.append(
            ExtractedEntity(
                name=name,
                entity_type=entity_type,
                confidence=confidence,
                evidence=evidence,
            )
        )

    return entities


def apply_heuristic_cleanup(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """
    Apply optional heuristic cleanup to extracted entities.

    Cleanup steps:
    - Strip titles (Dr, Mr, Ms, etc.)
    - Normalize org suffixes (Ltd, Limited, Inc, etc.)
    - Reject garbage (all punctuation, too short, numeric-only)

    Args:
        entities: List of extracted entities

    Returns:
        Cleaned list of entities
    """
    cleaned = []

    # Title patterns to strip
    title_pattern = r"^(Dr\.|Dr|Mr\.|Mr|Ms\.|Ms|Mrs\.|Mrs|Prof\.|Prof)\s+"

    # Org suffix patterns to normalize
    org_suffix_pattern = (
        r"\s+(Ltd\.?|Limited|Inc\.?|Incorporated|Corp\.?|Corporation|LLC|L\.L\.C\.)$"
    )

    for entity in entities:
        name = entity.name

        # Strip titles (PERSON only)
        if entity.entity_type == "PERSON":
            name = re.sub(title_pattern, "", name, flags=re.IGNORECASE)

        # Normalize org suffixes (ORG only)
        if entity.entity_type == "ORG":
            name = re.sub(org_suffix_pattern, "", name, flags=re.IGNORECASE)

        # Reject garbage
        name = name.strip()

        # Too short
        if len(name) < 2:
            continue

        # All punctuation
        if all(c in ".,;:!?-_()[]{}\"'" for c in name):
            continue

        # Numeric only
        if name.isdigit():
            continue

        # Update entity with cleaned name
        cleaned.append(
            ExtractedEntity(
                name=name,
                entity_type=entity.entity_type,
                confidence=entity.confidence,
                evidence=entity.evidence,
            )
        )

    return cleaned


def extract_entities_llm(
    content: str,
    allowed_types: list[str],
    model: str,
    apply_heuristics: bool = True,
    prompts_path: str = "prompts.yaml",
) -> list[ExtractedEntity]:
    """
    Extract entities from content using LLM.

    Args:
        content: Memory content text
        allowed_types: List of allowed entity types
        model: LLM model name
        apply_heuristics: Apply post-LLM heuristic cleanup
        prompts_path: Path to prompts.yaml

    Returns:
        List of validated ExtractedEntity objects

    Raises:
        ValueError: If extraction/validation fails
        FileNotFoundError: If prompts file not found
    """
    # Load prompts
    prompts = load_prompts(prompts_path)
    template = prompts.get("extract_entities")

    if not template:
        raise ValueError("'extract_entities' prompt not found in prompts.yaml")

    # Substitute tokens
    prompt = substitute_tokens(
        template,
        allowed_types=", ".join(allowed_types),
        content=content,
    )

    # Call LLM with schema for structured output
    try:
        result = call_llm(prompt, model=model, schema=EntityExtractionResult)
    except NotImplementedError:
        # For testing purposes, return empty list if LLM not implemented
        return []
    except Exception as e:
        raise ValueError(f"LLM extraction failed: {e}")

    # Convert schema response to ExtractedEntity objects
    entities = []
    for entity_schema in result.entities:
        # Validate type against allowed types
        entity_type_upper = entity_schema.type.upper()
        if entity_type_upper not in allowed_types:
            continue  # Skip invalid types

        entities.append(
            ExtractedEntity(
                name=entity_schema.name.strip(),
                entity_type=entity_type_upper,
                confidence=entity_schema.confidence,
                evidence=entity_schema.evidence.strip()[:200],  # Truncate evidence
            )
        )

    # Apply optional heuristic cleanup
    if apply_heuristics:
        entities = apply_heuristic_cleanup(entities)

    return entities


def compute_prompt_hash(template: str) -> str:
    """
    Compute hash of prompt template for reproducibility tracking.

    Args:
        template: Prompt template string

    Returns:
        First 16 chars of SHA256 hash
    """
    return hashlib.sha256(template.encode()).hexdigest()[:16]


def extract_and_store_entities(
    content: str,
    memory_id: str,
    storage,  # MemoryStorage instance
    config: dict[str, Any],
    artifact_ref: str | None = None,
) -> list[tuple]:
    """
    Extract entities from content and store with deduplication.

    This is the main integration function that:
    1. Calls LLM to extract entities
    2. Deduplicates via norm_key
    3. Stores entities in database
    4. Logs ENTITY_EXTRACTED event

    Args:
        content: Memory content to extract from
        memory_id: Memory ID (for event logging)
        storage: MemoryStorage instance
        config: M4 config dict
        artifact_ref: Optional artifact reference for event

    Returns:
        List of (entity_id, entity_type, confidence, evidence) tuples
        for entities that passed confidence threshold

    Raises:
        Exception: If extraction fails
    """
    from vestig.core.models import EntityNode, EventNode, compute_norm_key

    # Get config
    allowed_types = config.get("entity_types", {}).get("allowed_types", [])
    llm_config = config.get("entity_extraction", {}).get("llm", {})
    heuristics_config = config.get("entity_extraction", {}).get("heuristics", {})

    model = llm_config.get("model")
    if not model:
        raise ValueError("m4.entity_extraction.llm.model must be specified in config")

    min_confidence = llm_config.get("min_confidence", 0.75)
    apply_heuristics = heuristics_config.get("strip_titles", True)

    # Extract entities
    try:
        extracted = extract_entities_llm(
            content=content,
            allowed_types=allowed_types,
            model=model,
            apply_heuristics=apply_heuristics,
        )
    except NotImplementedError:
        # LLM not available - return empty list
        return []
    except Exception as e:
        print(f"Warning: Entity extraction failed: {e}")
        return []

    # Log extraction event
    prompts = load_prompts()
    template = prompts.get("extract_entities", "")
    prompt_hash = compute_prompt_hash(template)

    # Compute content hash for determinism tracking
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    # Import here to avoid circular dependency
    from vestig.core.event_storage import MemoryEventStorage

    event_storage = MemoryEventStorage(storage.conn)

    event = EventNode.create(
        memory_id=memory_id,
        event_type="ENTITY_EXTRACTED",
        source="llm",
        artifact_ref=artifact_ref,
        payload={
            "model_name": model,
            "prompt_hash": prompt_hash,
            "content_hash": content_hash,
            "entity_count": len(extracted),
            "min_confidence": min_confidence,
        },
    )
    event_storage.add_event(event)

    # Store entities with deduplication
    stored_entities = []

    for entity in extracted:
        # Apply confidence threshold
        if entity.confidence < min_confidence:
            continue

        # Compute norm_key for deduplication
        norm_key = compute_norm_key(entity.name, entity.entity_type)

        # Find or create entity
        existing = storage.find_entity_by_norm_key(norm_key, include_expired=False)

        if existing:
            # Entity already exists - use existing ID
            entity_id = existing.id
        else:
            # Create new entity
            new_entity = EntityNode.create(
                entity_type=entity.entity_type,
                canonical_name=entity.name,
            )
            entity_id = storage.store_entity(new_entity)

        # Return entity info for edge creation
        stored_entities.append((entity_id, entity.entity_type, entity.confidence, entity.evidence))

    return stored_entities
