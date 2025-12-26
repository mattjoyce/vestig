"""Entity extraction using LLM (M4: Graph Layer)"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml


@dataclass
class ExtractedEntity:
    """Entity extracted by LLM"""

    name: str
    entity_type: str
    confidence: float
    evidence: str


def load_prompts(prompts_path: str = "prompts.yaml") -> Dict[str, str]:
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


def call_llm(prompt: str, model: str = "claude-sonnet-4.5") -> str:
    """
    Call LLM with prompt using Anthropic SDK.

    Args:
        prompt: Full prompt text
        model: Model name (e.g., "claude-sonnet-4.5")

    Returns:
        LLM response text

    Raises:
        ImportError: If anthropic SDK not installed
        Exception: If LLM call fails
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic SDK not installed. Install with: pip install anthropic"
        )

    # Get API key from environment
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Get your API key from https://console.anthropic.com/"
        )

    # Map model name to Anthropic model ID
    model_map = {
        "claude-sonnet-4.5": "claude-sonnet-4-20250514",
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "claude-opus-4": "claude-opus-4-20250514",
        "claude-haiku-4": "claude-4-haiku-20250107",
    }

    model_id = model_map.get(model, model)

    # Call Anthropic API
    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model=model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from response
        return message.content[0].text

    except Exception as e:
        raise Exception(f"LLM call failed: {e}")


def validate_extraction_result(
    result: Dict[str, Any], allowed_types: List[str]
) -> List[ExtractedEntity]:
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
            raise ValueError(
                f"Entity {i}: invalid type '{entity_type}'. Allowed: {allowed_types}"
            )

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


def apply_heuristic_cleanup(entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
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
    org_suffix_pattern = r"\s+(Ltd\.?|Limited|Inc\.?|Incorporated|Corp\.?|Corporation|LLC|L\.L\.C\.)$"

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
    allowed_types: List[str],
    model: str = "claude-sonnet-4.5",
    apply_heuristics: bool = True,
    prompts_path: str = "prompts.yaml",
) -> List[ExtractedEntity]:
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

    # Call LLM
    try:
        response = call_llm(prompt, model=model)
    except NotImplementedError:
        # For testing purposes, return empty list if LLM not implemented
        return []

    # Parse JSON response
    try:
        result = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from LLM: {e}")

    # Validate extraction result
    entities = validate_extraction_result(result, allowed_types)

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
    config: Dict[str, Any],
    artifact_ref: Optional[str] = None,
) -> List[tuple]:
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

    model = llm_config.get("model", "claude-sonnet-4.5")
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
        stored_entities.append(
            (entity_id, entity.entity_type, entity.confidence, entity.evidence)
        )

    return stored_entities
