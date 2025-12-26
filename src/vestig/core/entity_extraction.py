"""Entity extraction using LLM (M4: Graph Layer)"""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


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


def store_entities(
    entities: list[tuple[str, str, float, str]],
    memory_id: str,
    storage,  # MemoryStorage instance
    config: dict[str, Any],
) -> list[tuple]:
    """
    Store pre-extracted entities with deduplication.

    Used when entities are already extracted (e.g., combined extraction during ingestion).

    Args:
        entities: List of (name, type, confidence, evidence) tuples
        memory_id: Memory ID (for event logging)
        storage: MemoryStorage instance
        config: M4 config dict

    Returns:
        List of (entity_id, entity_type, confidence, evidence) tuples
        for entities that passed confidence threshold
    """
    from vestig.core.models import EntityNode, compute_norm_key

    # Get config
    min_confidence = config.get("entity_extraction", {}).get("llm", {}).get("min_confidence", 0.75)

    # Store entities with deduplication
    stored_entities = []

    for name, entity_type, confidence, evidence in entities:
        # Apply confidence threshold
        if confidence < min_confidence:
            continue

        # Compute norm_key for deduplication
        norm_key = compute_norm_key(name, entity_type)

        # Find or create entity
        existing = storage.find_entity_by_norm_key(norm_key, include_expired=False)

        if existing:
            # Entity already exists - use existing ID
            entity_id = existing.id
        else:
            # Create new entity
            new_entity = EntityNode.create(
                entity_type=entity_type,
                canonical_name=name,
            )
            entity_id = storage.store_entity(new_entity)

        # Return entity info for edge creation
        stored_entities.append((entity_id, entity_type, confidence, evidence))

    return stored_entities
