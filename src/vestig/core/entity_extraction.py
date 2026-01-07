"""Entity extraction using LLM (M4: Graph Layer)"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from vestig.core.entity_ontology import EntityOntology, EntityType

logger = logging.getLogger(__name__)


# Cache for loaded prompts to avoid file descriptor leak
_PROMPTS_CACHE: dict[str, dict[str, Any]] = {}


def load_prompts(prompts_path: str | None = None) -> dict[str, Any]:
    """
    Load prompts from YAML file with caching to prevent file descriptor leaks.

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

    # Convert to string for cache key
    path_str = str(path.absolute())

    # Return cached prompts if available
    if path_str in _PROMPTS_CACHE:
        return _PROMPTS_CACHE[path_str]

    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    with open(path) as f:
        prompts = yaml.safe_load(f)

    # Cache the loaded prompts
    _PROMPTS_CACHE[path_str] = prompts

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


def build_entity_type_section(entity_type: EntityType) -> str:
    """
    Format single entity type for prompt inclusion.

    Args:
        entity_type: EntityType to format

    Returns:
        Formatted string with type name, description, synonyms, examples
    """
    lines = [f"- {entity_type.name}: {entity_type.description}"]

    if entity_type.synonyms:
        lines.append(f"  Synonyms: {', '.join(entity_type.synonyms)}")

    if entity_type.examples:
        lines.append(f"  Examples: {', '.join(entity_type.examples)}")

    return "\n".join(lines)


def build_entity_guidelines_block(ontology: EntityOntology) -> str:
    """
    Generate tiered entity extraction guidelines from ontology.

    Groups entity types by tier (1=always extract, 2=contextual, 3=high bar)
    and formats them with descriptions, synonyms, and examples.

    Args:
        ontology: EntityOntology instance

    Returns:
        Formatted multi-line string for prompt inclusion
    """
    tier_names = {
        1: "TIER 1 - Always Extract (High-value entities)",
        2: "TIER 2 - Contextual Extraction (Extract when relevant)",
        3: "TIER 3 - High Bar (Extract only when highly significant)",
    }

    sections = []

    for tier in [1, 2, 3]:
        types_in_tier = ontology.get_types_by_tier(tier)

        if types_in_tier:
            sections.append(tier_names[tier])
            for entity_type in types_in_tier:
                sections.append(build_entity_type_section(entity_type))
            sections.append("")  # Blank line between tiers

    return "\n".join(sections).rstrip()


def substitute_entity_types(
    template: str | dict[str, str], ontology: EntityOntology, **kwargs
) -> str | dict[str, str]:
    """
    Substitute entity type tokens in template along with other tokens.

    Handles two special tokens:
    - {{entity_types}}: Full tiered guidelines block
    - {{entity_type_list}}: Pipe-separated list of type names (e.g., "PERSON|ORG|SYSTEM")

    Args:
        template: Template string or dict with 'system'/'user' keys
        ontology: EntityOntology instance
        **kwargs: Additional token values

    Returns:
        Template with all tokens substituted (same type as input)
    """
    # Generate entity type tokens
    entity_tokens = {
        "entity_types": build_entity_guidelines_block(ontology),
        "entity_type_list": "|".join(ontology.get_type_names()),
    }

    # Merge with user-provided tokens
    all_tokens = {**entity_tokens, **kwargs}

    # Delegate to existing substitute_tokens function
    return substitute_tokens(template, **all_tokens)


def call_llm(
    prompt: str | dict[str, str],
    model: str,
    schema: type[BaseModel] | None = None,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
):
    """
    Call LLM with prompt using llm module, with retry logic for JSON parsing failures.

    Args:
        prompt: Full prompt text (string) or dict with 'system'/'user' keys (M4+)
        model: Model name (e.g., "claude-haiku-4.5")
        schema: Optional Pydantic model for structured output
        max_retries: Maximum number of retry attempts on JSON/validation failures (default: 3)
        backoff_seconds: Initial backoff delay in seconds, doubles on each retry (default: 1.0)

    Returns:
        If schema provided: Pydantic model instance
        If no schema: str (response text)

    Raises:
        ImportError: If llm module not installed
        Exception: If LLM call fails after all retries
    """
    try:
        import llm
    except ImportError:
        raise ImportError("llm module not installed. Install with: pip install llm")

    attempt = 0
    last_error = None

    while attempt < max_retries:
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
                    if not json_text or not json_text.strip():
                        raise json.JSONDecodeError("Empty response from LLM", json_text or "", 0)
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

        except (json.JSONDecodeError, ValidationError, ConnectionError, OSError, TimeoutError) as e:
            # Retryable errors: JSON/validation failures, network issues, timeouts
            attempt += 1
            last_error = e

            if attempt < max_retries:
                # Calculate backoff with exponential increase
                delay = backoff_seconds * (2 ** (attempt - 1))
                error_type = type(e).__name__

                # Show more detail for JSON errors
                error_detail = str(e)
                if isinstance(e, json.JSONDecodeError):
                    try:
                        response_preview = json_text[:200] if "json_text" in locals() else "N/A"
                        error_detail = f"{e} | Response preview: {repr(response_preview)}"
                    except Exception:
                        pass

                print(
                    f"{error_type} (attempt {attempt}/{max_retries}). "
                    f"Retrying in {delay:.1f}s... Error: {error_detail}"
                )
                time.sleep(delay)
            else:
                # All retries exhausted
                raise Exception(
                    f"LLM call failed after {max_retries} attempts. Last error: {last_error}"
                )

        except Exception as e:
            # Non-retryable error (unexpected failures)
            raise Exception(f"LLM call failed: {e}")

    # Should never reach here, but satisfy type checker
    raise Exception(f"LLM call failed after {max_retries} attempts. Last error: {last_error}")


def validate_and_log_entity_type(
    entity_type: str, ontology: EntityOntology, source: str = "extraction"
) -> str:
    """
    Validate entity type against ontology (soft validation).

    Logs warning for unknown types but doesn't reject them.
    This allows the LLM to discover new entity types while providing
    observability into type drift.

    Args:
        entity_type: Entity type to validate (case-insensitive)
        ontology: EntityOntology instance
        source: Description of where entity came from (for logging)

    Returns:
        Normalized (uppercase) entity type
    """
    normalized = entity_type.strip().upper()

    if not ontology.validate_type(normalized):
        known_types = ", ".join(ontology.get_type_names())
        logger.warning(
            f"Unknown entity type: '{normalized}' (source: {source}). Known types: {known_types}"
        )

    return normalized


def compute_prompt_hash(template: str) -> str:
    """Compute stable hash for prompt template tracking."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


class EntityExtractionResult(BaseModel):
    """Schema for entity extraction response"""

    entities: list[dict[str, Any]]


def extract_entities_from_text(
    text: str,
    model: str,
    ontology: EntityOntology,
    prompt_name: str = "extract_entities",
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> list[tuple[str, str, float, str]]:
    """
    Extract entities from any text using the extract_entities prompt.

    Reusable for memory content, user queries, or arbitrary text.

    Args:
        text: Text to extract entities from
        model: LLM model to use
        ontology: EntityOntology instance for prompt generation and validation
        prompt_name: Name of prompt template in prompts.yaml (default: extract_entities)
        max_retries: Maximum retry attempts on JSON failures
        backoff_seconds: Initial backoff delay for retries

    Returns:
        List of (name, type, confidence, evidence) tuples

    Raises:
        ValueError: If prompt not found or LLM call fails
    """
    # Load and substitute prompt
    prompts = load_prompts()
    template = prompts.get(prompt_name)

    if not template:
        raise ValueError(f"'{prompt_name}' prompt not found in prompts.yaml")

    # Use substitute_entity_types to inject ontology-based entity types
    prompt = substitute_entity_types(template, ontology, text=text)

    # Call LLM with retry logic
    try:
        result = call_llm(
            prompt,
            model=model,
            schema=EntityExtractionResult,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
    except Exception as e:
        raise ValueError(f"Entity extraction failed: {e}")

    # Convert to tuple format with validation
    entities = []
    for entity_dict in result.entities:
        name = entity_dict.get("name", "").strip()
        entity_type = entity_dict.get("type", "").strip()
        confidence = entity_dict.get("confidence", 0.0)
        evidence = entity_dict.get("evidence", "").strip()

        # Skip empty or invalid entities
        if not name or not entity_type:
            continue

        # Validate and normalize entity type (soft validation with logging)
        entity_type = validate_and_log_entity_type(
            entity_type, ontology, source=f"extract_entities_from_text({prompt_name})"
        )

        entities.append((name, entity_type, confidence, evidence))

    return entities


def store_entities(
    entities: list[tuple[str, str, float, str]],
    memory_id: str,
    storage,  # MemoryStorage instance
    config: dict[str, Any],
    embedding_engine=None,  # Optional EmbeddingEngine instance
    chunk_id: str | None = None,  # M5: Optional chunk ID for hub-and-spoke provenance
) -> list[tuple]:
    """
    Store pre-extracted entities with deduplication and embeddings.

    Used when entities are already extracted (e.g., combined extraction during ingestion).

    Args:
        entities: List of (name, type, confidence, evidence) tuples
        memory_id: Memory ID (for event logging)
        storage: MemoryStorage instance
        config: Full config dict (or M4 config dict if embedding_engine provided)
        embedding_engine: Optional EmbeddingEngine for generating entity embeddings
            If None, will be created from config.embedding settings

    Returns:
        List of (entity_id, entity_type, confidence, evidence) tuples
        for entities that passed confidence threshold
    """
    from vestig.core.models import EntityNode, compute_norm_key

    # Get config - handle both full config and m4-only config
    if "m4" in config:
        # Full config
        min_confidence = (
            config.get("m4", {})
            .get("entity_extraction", {})
            .get("llm", {})
            .get("min_confidence", 0.75)
        )
    else:
        # M4-only config
        min_confidence = (
            config.get("entity_extraction", {}).get("llm", {}).get("min_confidence", 0.75)
        )

    # Import embedding engine if needed
    if embedding_engine is None:
        from vestig.core.embeddings import EmbeddingEngine

        embedding_config = config.get("embedding", {})
        if not embedding_config or "model" not in embedding_config:
            raise ValueError(
                "store_entities requires either an embedding_engine parameter "
                "or full config with embedding settings (not just m4 config)"
            )

        embedding_engine = EmbeddingEngine(
            model_name=embedding_config["model"],
            expected_dimension=embedding_config["dimension"],
            normalize=embedding_config["normalize"],
            provider=embedding_config.get("provider", "llm"),
            max_length=embedding_config.get("max_length"),
        )

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

            # M5: Set chunk_id for hub-and-spoke provenance
            if chunk_id:
                new_entity.chunk_id = chunk_id

            # Generate embedding for entity (lowercase for consistency)
            embedding_text = name.lower()
            embedding_vector = embedding_engine.embed_text(embedding_text)
            new_entity.embedding = json.dumps(embedding_vector)

            entity_id = storage.store_entity(new_entity)

        # Return entity info for edge creation
        stored_entities.append((entity_id, entity_type, confidence, evidence))

    return stored_entities


def process_memories_for_entities(
    storage,
    config: dict[str, Any],
    ontology: EntityOntology,
    reprocess: bool = False,
    batch_size: int = 1,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Extract entities from memories that don't have entities yet.

    Args:
        storage: MemoryStorage instance
        config: Full config dict with m4 settings
        ontology: EntityOntology instance for entity extraction
        reprocess: If True, re-extract entities for ALL memories
        batch_size: Number of memories to process per batch (future: batch extraction)
        verbose: Print progress messages

    Returns:
        Dict with stats: {"memories_processed", "entities_created", "edges_created"}
    """
    # Get config
    m4_config = config.get("m4", {})
    entity_config = m4_config.get("entity_extraction", {})
    llm_config = entity_config.get("llm", {})

    model = llm_config.get("model", "claude-haiku-4.5")
    max_retries = config.get("ingestion", {}).get("retry", {}).get("max_attempts", 3)
    backoff = config.get("ingestion", {}).get("retry", {}).get("backoff_seconds", 1.0)
    min_confidence = llm_config.get("min_confidence", 0.75)

    # Find memories to process
    if reprocess:
        # Process all memories
        if verbose:
            print("Finding all memories for entity re-extraction...")

        # Query all memories
        cursor = storage.conn.execute(
            "SELECT id, content FROM memories WHERE kind = 'MEMORY' ORDER BY created_at"
        )
    else:
        # Process only memories without entity links
        if verbose:
            # Count total memories and those with entities
            total_memories = storage.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE kind = 'MEMORY'"
            ).fetchone()[0]
            memories_with_entities = storage.conn.execute("""
                SELECT COUNT(DISTINCT m.id)
                FROM memories m
                INNER JOIN edges e ON e.from_node = m.id
                WHERE m.kind = 'MEMORY' AND e.edge_type = 'MENTIONS'
            """).fetchone()[0]

            print("Finding memories without entities...")
            print(f"  Total memories: {total_memories}")
            print(f"  Memories with entities (skipping): {memories_with_entities}")
            print(
                f"  Memories without entities (to process): {total_memories - memories_with_entities}"
            )

        # Find memories with no MENTIONS edges
        cursor = storage.conn.execute("""
            SELECT m.id, m.content
            FROM memories m
            WHERE m.kind = 'MEMORY'
              AND NOT EXISTS (
                  SELECT 1 FROM edges e
                  WHERE e.from_node = m.id AND e.edge_type = 'MENTIONS'
              )
            ORDER BY m.created_at
        """)

    memories = cursor.fetchall()

    if verbose:
        print(f"Found {len(memories)} memories to process")
        print()

    # Process each memory
    stats = {"memories_processed": 0, "entities_created": 0, "edges_created": 0}
    entities_before = storage.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    for memory_id, content in memories:
        try:
            if verbose:
                # Show progress periodically, or always if extracting entities
                if stats["memories_processed"] % 10 == 0 or True:
                    print(
                        f"\n[{stats['memories_processed'] + 1}/{len(memories)}] Memory {memory_id}"
                    )
                    # Truncate content for display
                    display_content = content[:200] + "..." if len(content) > 200 else content
                    print(f"  Content: {display_content}")

            # Extract entities from memory content
            entities = extract_entities_from_text(
                text=content,
                model=model,
                ontology=ontology,
                max_retries=max_retries,
                backoff_seconds=backoff,
            )

            if verbose:
                if entities:
                    print(f"  Extracted {len(entities)} entities:")
                    for name, entity_type, confidence, evidence in entities:
                        print(f"    • {name} ({entity_type}) - confidence: {confidence:.2f}")
                        if evidence:
                            # Truncate evidence for display
                            display_evidence = (
                                evidence[:100] + "..." if len(evidence) > 100 else evidence
                            )
                            print(f'      Evidence: "{display_evidence}"')
                else:
                    print("  No entities extracted")

            if not entities:
                stats["memories_processed"] += 1
                continue

            # If reprocessing, delete existing MENTIONS edges for this memory
            if reprocess:
                storage.conn.execute(
                    "DELETE FROM edges WHERE from_node = ? AND edge_type = 'MENTIONS'", (memory_id,)
                )

            # Store entities and create MENTIONS edges
            stored_entities = store_entities(
                entities=entities, memory_id=memory_id, storage=storage, config=config
            )

            # Create MENTIONS edges
            from vestig.core.models import EdgeNode

            for entity_id, entity_type, confidence, evidence in stored_entities:
                if confidence >= min_confidence:
                    edge = EdgeNode.create(
                        from_node=memory_id,
                        to_node=entity_id,
                        edge_type="MENTIONS",
                        weight=1.0,
                        confidence=confidence,
                        evidence=evidence,
                    )
                    storage.store_edge(edge)
                    stats["edges_created"] += 1

            if verbose and stored_entities:
                print(
                    f"  Stored {len(stored_entities)} entities, created {len(stored_entities)} edges"
                )

            stats["memories_processed"] += 1

        except Exception as e:
            if verbose:
                print(f"  ✗ Error processing memory {memory_id}: {e}")
            continue

    # Count new entities
    entities_after = storage.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    stats["entities_created"] = entities_after - entities_before

    if verbose:
        print("\nEntity extraction complete:")
        print(f"  Memories processed: {stats['memories_processed']}")
        print(f"  Entities created: {stats['entities_created']}")
        print(f"  Edges created: {stats['edges_created']}")

    return stats
