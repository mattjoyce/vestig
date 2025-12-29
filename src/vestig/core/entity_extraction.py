"""Entity extraction using LLM (M4: Graph Layer)"""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def load_prompts(prompts_path: str | None = None) -> dict[str, Any]:
    """
    Load prompts from YAML file.

    Args:
        prompts_path: Optional path to prompts.yaml. If None, uses prompts.yaml
                     in the same directory as this module.

    Returns:
        Dictionary of prompt templates (supports both string and dict formats)
        - String format (legacy): prompt_name: "prompt text"
        - Dict format (M4+): prompt_name: {system: "...", user: "...", description: "..."}
    """
    if prompts_path is None:
        # Default: prompts.yaml in same directory as this module
        module_dir = Path(__file__).parent
        path = module_dir / "prompts.yaml"
    else:
        path = Path(prompts_path)

    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    with open(path) as f:
        prompts = yaml.safe_load(f)

    return prompts


def substitute_tokens(template: str | dict[str, str], **kwargs) -> str | dict[str, str]:
    """
    Substitute {{token}} placeholders in template.

    Args:
        template: Template string or dict with 'system'/'user' keys
        **kwargs: Token values

    Returns:
        Template with tokens substituted (same type as input)

    Examples:
        >>> substitute_tokens("Hello {{name}}", name="Alice")
        'Hello Alice'
        >>> substitute_tokens({"system": "You are {{role}}", "user": "Do {{task}}"}, role="helper", task="summarize")
        {'system': 'You are helper', 'user': 'Do summarize'}
    """
    if isinstance(template, dict):
        # M4+: Handle dict format with system/user prompts
        result = {}
        if "system" in template:
            result["system"] = template["system"]
            for key, value in kwargs.items():
                placeholder = f"{{{{{key}}}}}"
                result["system"] = result["system"].replace(placeholder, str(value))
        if "user" in template:
            result["user"] = template["user"]
            for key, value in kwargs.items():
                placeholder = f"{{{{{key}}}}}"
                result["user"] = result["user"].replace(placeholder, str(value))
        return result
    else:
        # Legacy: Handle string format
        result = template
        for key, value in kwargs.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        return result


def call_llm(
    prompt: str | dict[str, str],
    model: str,
    schema: type[BaseModel] | None = None
):
    """
    Call LLM with prompt using llm module.

    Args:
        prompt: Full prompt text (string) or dict with 'system'/'user' keys (M4+)
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

        # Handle both string and dict formats
        if isinstance(prompt, dict):
            # M4+: Dict format with system/user prompts
            system_text = prompt.get("system", "")
            user_text = prompt.get("user", "")

            if schema:
                # Use structured output with schema
                response = llm_model.prompt(user_text, system=system_text, schema=schema)
                # Parse JSON response and validate with schema
                json_text = response.text()
                data = json.loads(json_text)
                return schema.model_validate(data)
            else:
                # Return raw text
                response = llm_model.prompt(user_text, system=system_text)
                return response.text()
        else:
            # Legacy: String format (no system prompt)
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


def compute_prompt_hash(template: str) -> str:
    """Compute stable hash for prompt template tracking."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


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
