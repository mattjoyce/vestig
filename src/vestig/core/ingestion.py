"""Document ingestion with LLM-based memory extraction"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vestig.core.commitment import commit_memory
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.entity_extraction import (
    call_llm,
    load_prompts,
    substitute_tokens,
)
from vestig.core.event_storage import MemoryEventStorage
from vestig.core.ingest_sources import (
    extract_claude_session_chunks,
    normalize_document_text,
)
from vestig.core.models import MemoryNode
from vestig.core.storage import MemoryStorage


@dataclass
class TemporalHints:
    """
    Temporal metadata extracted by parsers.

    Represents the temporal context for a document/chunk being ingested.
    These hints flow from parser → ExtractedMemory → MemoryNode.
    """

    # Primary timestamp for this content
    t_valid: str | None = None  # ISO 8601 timestamp

    # Temporal classification
    stability: str = "unknown"  # "static" | "dynamic" | "ephemeral" | "unknown"

    # Evidence for debugging
    extraction_method: str = (
        "default"  # "jsonl_timestamp" | "file_mtime" | "filename_pattern" | "default"
    )
    evidence: str | None = None  # Human-readable explanation

    @classmethod
    def from_now(cls) -> "TemporalHints":
        """Create hints with current time (fallback)."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        return cls(
            t_valid=now,
            stability="unknown",
            extraction_method="default",
            evidence="No temporal metadata available; using current time",
        )

    @classmethod
    def from_file_mtime(cls, path: Path) -> "TemporalHints":
        """Extract temporal hints from file modification time."""
        from datetime import datetime, timezone

        mtime = path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return cls(
            t_valid=dt.isoformat(),
            stability="unknown",
            extraction_method="file_mtime",
            evidence=f"Extracted from file modification time: {path.name}",
        )

    @classmethod
    def from_timestamp(cls, timestamp: str, evidence: str) -> "TemporalHints":
        """Create hints from explicit timestamp (e.g., JSONL event)."""
        return cls(
            t_valid=timestamp,
            stability="unknown",
            extraction_method="jsonl_timestamp",
            evidence=evidence,
        )


@dataclass
class ExtractedMemory:
    """Memory extracted from document by LLM with temporal hints"""

    content: str
    confidence: float
    rationale: str
    entities: list[tuple[str, str, float, str]]  # (name, type, confidence, evidence)

    # Temporal hints (optional, None = use defaults)
    t_valid_hint: str | None = None  # When fact became true (ISO 8601)
    temporal_stability_hint: str | None = None  # "static" | "dynamic" | "unknown"
    temporal_evidence: str | None = None  # Brief explanation of temporal extraction


# Pydantic schemas for LLM structured output
class EntitySchema(BaseModel):
    """Schema for an extracted entity"""

    name: str = Field(description="Entity name or identifier")
    type: str = Field(
        description="Entity type: PERSON, ORG, SYSTEM, PROJECT, PLACE, SKILL, TOOL, FILE, or CONCEPT"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    evidence: str = Field(description="Text snippet from memory that supports this entity")


class MemorySchema(BaseModel):
    """Schema for a single memory with entities"""

    content: str = Field(
        description="The full memory text with enough context to be self-contained"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    rationale: str = Field(description="Brief explanation of why this is worth remembering")
    temporal_stability: str = Field(
        default="unknown",
        description="Temporal stability: static (permanent), dynamic (changing), ephemeral (expected to change soon), or unknown",
    )
    entities: list[EntitySchema] = Field(
        default_factory=list,
        description="Entities mentioned in this memory (people, orgs, systems, projects, places)",
    )


class MemoryExtractionResult(BaseModel):
    """Schema for memory extraction response"""

    memories: list[MemorySchema] = Field(description="List of extracted memories")


class SummaryBullet(BaseModel):
    """A single bullet point in a summary with citations"""

    text: str = Field(description="The bullet point text")
    memory_ids: list[str] = Field(description="Memory IDs supporting this bullet (1-4 IDs)")


class SummaryData(BaseModel):
    """The summary content"""

    title: str = Field(description="Short descriptive title")
    overview: str = Field(description="2-4 sentence overview in plain language")
    bullets: list[SummaryBullet] = Field(description="6-12 key insights with citations")
    themes: list[str] = Field(description="3-8 short theme tags")
    open_questions: list[str] = Field(
        default_factory=list, description="Genuine gaps or ambiguities"
    )


class SummaryResult(BaseModel):
    """Schema for summary generation response (M4)"""

    summary: SummaryData = Field(description="The generated summary")


@dataclass
class IngestionResult:
    """Result of document ingestion"""

    document_path: str
    chunks_processed: int
    memories_extracted: int
    memories_committed: int
    memories_deduplicated: int
    entities_created: int
    errors: list[str]


def parse_force_entities(force_entities: list[str] | None) -> list[tuple[str, str, float, str]]:
    if not force_entities:
        return []

    parsed: list[tuple[str, str, float, str]] = []
    for raw in force_entities:
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"Forced entity must be TYPE:Name, got: {raw}")
        entity_type, name = raw.split(":", 1)
        entity_type = entity_type.strip().upper()
        name = name.strip()
        if not entity_type or not name:
            raise ValueError(f"Forced entity must be TYPE:Name, got: {raw}")
        parsed.append((name, entity_type, 1.0, "forced_ingest"))

    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str, float, str]] = []
    for name, entity_type, confidence, evidence in parsed:
        key = (name.lower(), entity_type)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, entity_type, confidence, evidence))

    return unique


def chunk_text_by_chars(
    text: str, chunk_size: int = 20000, overlap: int = 500
) -> list[tuple[str, int, int]]:
    """
    Chunk text by character count with overlap, tracking positions.

    Simple chunking strategy that tries to break at paragraph boundaries.
    Uses character count as proxy for tokens (roughly 1 token = 4 chars).

    Args:
        text: Text to chunk
        chunk_size: Max characters per chunk (default 20k chars ≈ 5k tokens)
        overlap: Character overlap between chunks (for context continuity)

    Returns:
        List of (chunk_text, start_position, length) tuples
    """
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]

    chunks = []
    start = 0

    while start < len(text):
        # Find end of chunk
        end = start + chunk_size

        if end >= len(text):
            # Last chunk
            chunk_text = text[start:]
            chunks.append((chunk_text, start, len(chunk_text)))
            break

        # Try to break at paragraph boundary (double newline)
        break_point = text.rfind("\n\n", start, end)
        if break_point == -1:
            # Try single newline
            break_point = text.rfind("\n", start, end)
        if break_point == -1:
            # Try space
            break_point = text.rfind(" ", start, end)
        if break_point == -1:
            # Hard break
            break_point = end

        chunk_text = text[start:break_point]
        chunks.append((chunk_text, start, len(chunk_text)))

        # Next chunk starts with overlap
        start = max(start + 1, break_point - overlap)

    return chunks


def link_entities_to_memories(
    memories: list[tuple[str, float, str]],  # (content, confidence, rationale)
    entities: list[tuple[str, str, float, str]],  # (name, type, confidence, evidence)
) -> dict[int, list[tuple[str, str, float, str]]]:
    """
    Link chunk-level entities to memories via substring matching.

    Args:
        memories: List of (content, confidence, rationale) tuples
        entities: List of (name, type, confidence, evidence) tuples from chunk

    Returns:
        Dict mapping memory index to list of entities that match that memory
    """
    memory_entities = {}

    for mem_idx, (content, _, _) in enumerate(memories):
        matched = []
        content_lower = content.lower()

        for entity_name, entity_type, confidence, evidence in entities:
            # Case-insensitive substring match
            if entity_name.lower() in content_lower:
                matched.append((entity_name, entity_type, confidence, evidence))

        memory_entities[mem_idx] = matched

    return memory_entities


def extract_memories_from_chunk(
    chunk: str,
    model: str,
    min_confidence: float = 0.6,
    temporal_hints: TemporalHints | None = None,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    extraction_config: dict[str, Any] | None = None,
    ontology=None,
) -> list[ExtractedMemory]:
    """
    Extract memories from a text chunk using LLM.

    Supports both single-pass and two-pass extraction:
    - Single-pass: Extract memories with entities in one LLM call
    - Two-pass: Extract memories first, then entities separately (uses ontology)

    Args:
        chunk: Text chunk to extract from
        model: LLM model to use
        min_confidence: Minimum confidence threshold
        temporal_hints: Optional temporal context for this chunk
        max_retries: Max retry attempts on JSON failures
        backoff_seconds: Initial backoff delay for retries
        extraction_config: Optional ingestion.two_pass_extraction config
        ontology: Optional EntityOntology for two-pass entity extraction

    Returns:
        List of ExtractedMemory objects with temporal hints

    Raises:
        ValueError: If LLM returns invalid JSON
    """
    # Check if two-pass extraction is enabled
    two_pass_enabled = False
    memory_prompt_name = "extract_memories_from_session"
    entity_prompt_name = "extract_entities"

    if extraction_config:
        two_pass_enabled = extraction_config.get("enabled", False)
        if two_pass_enabled:
            memory_prompt_name = extraction_config.get("memory_prompt", "extract_memories_simple")
            entity_prompt_name = extraction_config.get("entity_prompt", "extract_entities")

    # Load and substitute prompt
    prompts = load_prompts()
    template = prompts.get(memory_prompt_name)

    if not template:
        raise ValueError(f"'{memory_prompt_name}' prompt not found in prompts.yaml")

    prompt = substitute_tokens(template, content=chunk)

    # Debug: Show extraction mode
    print(f"    Using prompt: {memory_prompt_name} (two-pass: {two_pass_enabled})", flush=True)
    print(f"    Calling LLM model: {model}...", flush=True)

    # Call LLM with schema for structured output and retry logic
    try:
        result = call_llm(
            prompt,
            model=model,
            schema=MemoryExtractionResult,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds
        )
    except Exception as e:
        raise ValueError(f"LLM call failed: {e}")

    # Convert schema response to ExtractedMemory objects
    # Note: For two-pass extraction, entities will be extracted later (after dedup check)
    memories = []
    for memory_schema in result.memories:
        content = memory_schema.content.strip()
        confidence = memory_schema.confidence
        rationale = memory_schema.rationale.strip()

        # Apply confidence threshold
        if confidence < min_confidence:
            continue

        # Skip empty or too-short content
        if not content or len(content) < 10:
            continue

        # Get entities (only for single-pass)
        if two_pass_enabled:
            # Two-pass: Skip entity extraction here (done after commit in caller)
            entities = []
        else:
            # Single-pass: Extract entities from schema (if present)
            entities = [
                (
                    entity.name.strip(),
                    entity.type.strip().upper(),
                    entity.confidence,
                    entity.evidence.strip(),
                )
                for entity in memory_schema.entities
            ]

        # Attach temporal hints to each extracted memory
        t_valid_hint = None
        temporal_stability_hint = "unknown"
        temporal_evidence = None

        if temporal_hints:
            t_valid_hint = temporal_hints.t_valid
            # Use LLM-classified stability, fallback to parser hints
            temporal_stability_hint = memory_schema.temporal_stability
            if not temporal_stability_hint or temporal_stability_hint == "unknown":
                temporal_stability_hint = temporal_hints.stability
            temporal_evidence = temporal_hints.evidence

        # If no temporal hints from parser, use LLM classification only
        if not temporal_hints and memory_schema.temporal_stability:
            temporal_stability_hint = memory_schema.temporal_stability

        memories.append(
            ExtractedMemory(
                content=content,
                confidence=confidence,
                rationale=rationale,
                entities=entities,
                t_valid_hint=t_valid_hint,
                temporal_stability_hint=temporal_stability_hint,
                temporal_evidence=temporal_evidence,
            )
        )

    return memories


def generate_summary(
    memories: list[MemoryNode],
    model: str,
    source_label: str,
    ingest_run_id: str,
    prompt_name: str = "summary_v1",
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> SummaryResult:
    """
    Generate summary from extracted memories using LLM (M4).

    Args:
        memories: List of MemoryNode objects to summarize
        model: LLM model to use
        source_label: Human-readable source label (filename/session name)
        ingest_run_id: Ingest run identifier (artifact_ref)
        prompt_name: Name of prompt template in prompts.yaml (default: summary_v1)

    Returns:
        SummaryResult with structured summary data

    Raises:
        ValueError: If LLM returns invalid response or prompts not found
    """
    from vestig.core.entity_extraction import call_llm, load_prompts, substitute_tokens

    # Load and get prompt
    prompts = load_prompts()
    summary_prompt = prompts.get(prompt_name)

    if not summary_prompt:
        raise ValueError(f"'{prompt_name}' prompt not found in prompts.yaml")

    # Format memories list for prompt
    memory_items_list = []
    for mem in memories:
        # Truncate long memories for prompt efficiency (keep under 500 chars)
        content = mem.content[:500] + "..." if len(mem.content) > 500 else mem.content
        memory_items_list.append(f"[{mem.id}] {content}")

    memory_items_text = "\n\n".join(memory_items_list)

    # Substitute tokens in prompt
    prompt = substitute_tokens(
        summary_prompt,
        source_label=source_label,
        ingest_run_id=ingest_run_id,
        memory_items=memory_items_text,
    )

    # Call LLM with schema and retry logic
    try:
        result = call_llm(
            prompt,
            model=model,
            schema=SummaryResult,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds
        )
        return result
    except Exception as e:
        raise ValueError(f"Summary generation failed: {e}")


def commit_summary(
    summary_result: SummaryResult,
    memory_ids: list[str],
    artifact_ref: str,
    source_label: str,
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    event_storage: MemoryEventStorage | None = None,
    chunk_id: str | None = None,  # M5: Link summary to chunk
) -> str | None:
    """
    Commit summary node and create SUMMARIZES edges (M4/M5).

    Idempotency: Uses deterministic lookup by (artifact_ref in metadata)
    to prevent duplicate summaries on re-runs.

    Args:
        summary_result: Generated summary from LLM
        memory_ids: List of memory IDs being summarized
        artifact_ref: Source artifact reference (filename/session ID)
        source_label: Human-readable source label
        storage: Storage instance
        embedding_engine: Embedding engine
        event_storage: Optional event storage
        chunk_id: M5 chunk ID (for per-chunk summaries)

    Returns:
        Summary memory ID if created/found, None if skipped
    """
    from datetime import datetime, timezone
    import hashlib
    import uuid
    from vestig.core.models import MemoryNode, EdgeNode, EventNode

    # M5: Check if summary already exists (idempotency)
    # For per-chunk summaries, check by chunk_id; otherwise check by artifact_ref
    if chunk_id:
        existing_summary = storage.get_summary_for_chunk(chunk_id)
    else:
        existing_summary = storage.get_summary_for_artifact(artifact_ref)

    if existing_summary:
        print(f"  Summary already exists: {existing_summary.id} (idempotent)")
        return existing_summary.id

    # Format summary content from bullets
    summary_data = summary_result.summary
    content_lines = [
        f"# {summary_data.title}",
        "",
        f"**Overview:** {summary_data.overview}",
        "",
        "**Key Insights:**",
    ]

    for bullet in summary_data.bullets:
        # Citations are stored as SUMMARIZES edges, not in content
        content_lines.append(f"- {bullet.text}")

    if summary_data.themes:
        content_lines.append("")
        content_lines.append(f"**Themes:** {', '.join(summary_data.themes)}")

    if summary_data.open_questions:
        content_lines.append("")
        content_lines.append("**Open Questions:**")
        for q in summary_data.open_questions:
            content_lines.append(f"- {q}")

    summary_content = "\n".join(content_lines)

    # Create summary memory node
    summary_id = f"mem_{uuid.uuid4()}"

    # Generate embedding for summary
    embedding = embedding_engine.embed_text(summary_content)

    # Compute content hash
    content_hash = hashlib.sha256(summary_content.encode("utf-8")).hexdigest()

    # Build metadata
    metadata = {
        "source": "summary_generation",
        "artifact_ref": artifact_ref,
        "source_label": source_label,
        "title": summary_data.title,
        "themes": summary_data.themes,
        "memory_count": len(memory_ids),
        "summarized_ids": memory_ids,
    }

    now = datetime.now(timezone.utc).isoformat()

    # Create MemoryNode with kind=SUMMARY
    summary_node = MemoryNode(
        id=summary_id,
        content=summary_content,
        content_embedding=embedding,
        content_hash=content_hash,
        created_at=now,
        metadata=metadata,
        t_valid=now,
        t_invalid=None,
        t_created=now,
        t_expired=None,
        temporal_stability="static",  # Summaries are snapshots
        last_seen_at=None,
        reinforce_count=0,
        chunk_id=chunk_id,  # M5: Link summary to chunk
    )

    # Atomic transaction for summary + edges + event
    with storage.conn:
        # Store summary with kind=SUMMARY
        storage.store_memory(summary_node, kind="SUMMARY")

        # Create SUMMARIZES edges (Summary → Memory)
        for memory_id in memory_ids:
            edge = EdgeNode.create(
                from_node=summary_id,
                to_node=memory_id,
                edge_type="SUMMARIZES",
                weight=1.0,
                confidence=None,  # Summaries don't have confidence scores
                evidence=f"summary_of_{len(memory_ids)}_memories",
            )
            storage.store_edge(edge)

        # Log SUMMARY_CREATED event
        if event_storage:
            event = EventNode.create(
                memory_id=summary_id,
                event_type="SUMMARY_CREATED",
                source="summary_generation",
                artifact_ref=artifact_ref,
                payload={
                    "memory_count": len(memory_ids),
                    "summarized_ids": memory_ids,
                    "title": summary_data.title,
                    "themes": summary_data.themes,
                },
            )
            event_storage.add_event(event)

    print(f"  Summary created: {summary_id} (summarizes {len(memory_ids)} memories)")
    return summary_id


def ingest_document(
    document_path: str,
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    extraction_model: str,
    event_storage: MemoryEventStorage | None = None,
    m4_config: dict[str, Any] | None = None,
    prompts_config: dict[str, Any] | None = None,
    ingestion_config: dict[str, Any] | None = None,
    chunk_size: int = 20000,
    chunk_overlap: int = 500,
    min_confidence: float = 0.6,
    source: str = "document_ingest",
    source_format: str = "auto",
    format_config: dict[str, Any] | None = None,
    force_entities: list[str] | None = None,
    verbose: bool = False,
) -> IngestionResult:
    """
    Ingest document by extracting memories with LLM and committing them.

    Args:
        document_path: Path to document file
        storage: Storage instance
        embedding_engine: Embedding engine
        event_storage: Optional event storage
        m4_config: Optional M4 config (for entity extraction)
        prompts_config: Optional prompts config (for selecting prompt versions)
        ingestion_config: Optional ingestion config (for two-pass extraction)
        chunk_size: Characters per chunk (default 20k ≈ 5k tokens)
        chunk_overlap: Character overlap between chunks
        extraction_model: LLM model for memory extraction
        min_confidence: Minimum confidence for extracted memories
        source: Source tag for committed memories
        source_format: Input format (auto|plain|claude-session)
        format_config: Format-specific configuration
        force_entities: Entity list to attach to every memory
        verbose: Print detailed extraction output

    Returns:
        IngestionResult with statistics

    Raises:
        FileNotFoundError: If document doesn't exist
        ValueError: If extraction fails
    """
    # Read document
    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    # Load ontology for entity extraction (if M4 enabled)
    ontology = None
    if m4_config and m4_config.get("entity_extraction", {}).get("enabled", False):
        print("Loading entity ontology...", flush=True)
        from vestig.core.entity_ontology import EntityOntology

        try:
            ontology = EntityOntology.from_config(m4_config)
            print(f"Ontology loaded: {len(ontology.types)} entity types", flush=True)
        except Exception as e:
            print(f"Warning: Failed to load entity ontology: {e}")

    # Extract two-pass extraction config
    extraction_config = None
    if ingestion_config:
        two_pass = ingestion_config.get("two_pass_extraction")
        if two_pass and two_pass.get("enabled", False):
            extraction_config = two_pass
            print(f"Two-pass extraction enabled: {two_pass.get('memory_prompt')} + {two_pass.get('entity_prompt')}", flush=True)

    raw_text = path.read_text(encoding="utf-8")

    text, resolved_format, document_temporal_hints = normalize_document_text(
        raw_text,
        source_format=source_format,
        format_config=format_config,
        path=path,
    )
    forced_entities = parse_force_entities(force_entities)
    if not text.strip():
        raise ValueError(f"No ingestible content found in {document_path}")
    if verbose:
        print(f"Normalized input using format: {resolved_format}")
        print(f"Temporal extraction: {document_temporal_hints.extraction_method}")
        if document_temporal_hints.t_valid:
            print(f"  t_valid: {document_temporal_hints.t_valid}")
        if document_temporal_hints.evidence:
            print(f"  Evidence: {document_temporal_hints.evidence}")

    # M5: Create FILE record (hub-and-spoke model)
    from vestig.core.models import FileNode
    import hashlib
    from datetime import datetime, timezone

    document_path_abs = str(path.absolute())

    # Compute file hash for change detection
    file_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    # Get file creation time (use mtime as fallback)
    file_created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

    # Create FILE node
    file_node = FileNode.create(
        path=document_path_abs,
        file_hash=file_hash,
        file_created_at=file_created_at,
        metadata={
            "format": resolved_format,
            "size": len(raw_text),
            "ingestion_source": source,
        },
    )
    file_id = storage.store_file(file_node)
    print(f"Created file record: {file_id}")

    # Chunk text
    if resolved_format == "claude-session":
        chunks_with_hints = extract_claude_session_chunks(
            raw_text,
            format_config or {},
            chunk_size,
            chunk_overlap,
        )
        chunks = [chunk for chunk, _ in chunks_with_hints]
        # Add chunk metadata (text, start, length) with hints
        chunks_with_metadata = []
        for chunk_text, hints in chunks_with_hints:
            # For claude-session, we don't have position tracking yet
            # Use placeholder positions (will be improved later)
            chunks_with_metadata.append((chunk_text, 0, len(chunk_text), hints))
    else:
        chunks_with_positions = chunk_text_by_chars(text, chunk_size=chunk_size, overlap=chunk_overlap)
        chunks = [chunk_text for chunk_text, _, _ in chunks_with_positions]
        # Create (text, start, length, hints) tuples
        chunks_with_metadata = [
            (chunk_text, start_pos, length, document_temporal_hints)
            for chunk_text, start_pos, length in chunks_with_positions
        ]

    print(f"Processing {path.name}: {len(text):,} chars → {len(chunks)} chunks")

    # Track results
    memories_extracted = 0
    memories_committed = 0
    memories_deduplicated = 0
    errors = []

    # Track entities before and after
    entities_before = len(storage.get_all_entities())

    # Process each chunk
    for i, (chunk_text, start_pos, length, chunk_hints) in enumerate(chunks_with_metadata, 1):
        # M5: Create CHUNK record (hub node with location pointer)
        from vestig.core.models import ChunkNode

        chunk_node = ChunkNode.create(
            file_id=file_id,
            start=start_pos,
            length=length,
            sequence=i,  # 1-indexed sequence
        )
        chunk_id = storage.store_chunk(chunk_node)

        print(f"  Chunk {i}/{len(chunks)} (chunk_id={chunk_id}): ", end="", flush=True)

        try:
            # Extract memories from chunk
            extracted = extract_memories_from_chunk(
                chunk_text,
                model=extraction_model,
                min_confidence=min_confidence,
                temporal_hints=chunk_hints,
                extraction_config=extraction_config,
                ontology=ontology,
            )

            print(f"{len(extracted)} memories extracted", flush=True)
            memories_extracted += len(extracted)

            # Show extracted memories in verbose mode (without entities for two-pass)
            if verbose and extracted:
                for idx, memory in enumerate(extracted, 1):
                    print(f"    Memory {idx}:")
                    print(
                        f"      Content: {memory.content[:100]}..."
                        if len(memory.content) > 100
                        else f"      Content: {memory.content}"
                    )
                    print(f"      Confidence: {memory.confidence:.2f}")
                    print(f"      Rationale: {memory.rationale}")

            # Commit each memory (without entities for two-pass)
            # Track which memories were actually inserted (not duplicates)
            committed_memories = []  # List of (memory_id, memory_content) tuples

            for idx, memory in enumerate(extracted, 1):
                try:
                    combined_entities = []
                    if memory.entities:
                        combined_entities.extend(memory.entities)
                    if forced_entities:
                        combined_entities.extend(forced_entities)
                    if combined_entities:
                        seen_entities: set[tuple[str, str]] = set()
                        deduped_entities = []
                        for name, entity_type, confidence, evidence in combined_entities:
                            key = (name.lower(), entity_type)
                            if key in seen_entities:
                                continue
                            seen_entities.add(key)
                            deduped_entities.append((name, entity_type, confidence, evidence))
                        combined_entities = deduped_entities

                    outcome = commit_memory(
                        content=memory.content,
                        storage=storage,
                        embedding_engine=embedding_engine,
                        source=source,
                        event_storage=event_storage,
                        m4_config=m4_config,
                        artifact_ref=path.name,
                        pre_extracted_entities=combined_entities or None,
                        temporal_hints=memory,  # Pass ExtractedMemory with temporal fields
                        chunk_id=chunk_id,  # M5: Link to chunk hub node
                    )

                    if outcome.outcome == "INSERTED_NEW":
                        memories_committed += 1
                        # Track committed memory for later entity extraction
                        committed_memories.append((outcome.memory_id, memory.content))

                        if verbose:
                            print(f"    Memory {idx} - COMMITTED")
                    else:
                        memories_deduplicated += 1
                        if verbose:
                            print(f"    Memory {idx} - DUPLICATE (not committed)")

                except Exception as e:
                    error_msg = f"Failed to commit memory: {e}"
                    errors.append(error_msg)
                    print(f"    Error: {error_msg}")

            # Extract and link entities from chunk (only if we have committed memories)
            if extraction_config and extraction_config.get("enabled") and ontology and committed_memories:
                try:
                    print(f"    Extracting entities from chunk ({len(committed_memories)} memories committed)...", flush=True)
                    from vestig.core.entity_extraction import extract_entities_from_text

                    entity_prompt = extraction_config.get("entity_prompt", "extract_entities")
                    chunk_entities = extract_entities_from_text(
                        text=chunk_text,
                        model=extraction_model,
                        ontology=ontology,
                        prompt_name=entity_prompt,
                        max_retries=3,
                        backoff_seconds=1.0,
                    )
                    print(f"    Found {len(chunk_entities)} entities in chunk", flush=True)

                    # Link entities to committed memories
                    memory_contents = [(mem_content, 0.0, "") for _, mem_content in committed_memories]
                    memory_entity_map = link_entities_to_memories(memory_contents, chunk_entities)

                    # Store entities for each committed memory
                    from vestig.core.entity_extraction import store_entities

                    for mem_idx, (memory_id, _) in enumerate(committed_memories):
                        entities = memory_entity_map.get(mem_idx, [])
                        if entities:
                            try:
                                store_entities(
                                    entities=entities,
                                    memory_id=memory_id,
                                    storage=storage,
                                    config=m4_config,
                                    embedding_engine=embedding_engine,
                                    chunk_id=chunk_id,  # M5: Link entities to chunk hub
                                )
                                if verbose:
                                    print(f"    Memory {mem_idx + 1} - Entities committed ({len(entities)}):")
                                    for name, entity_type, conf, evidence in entities:
                                        evid_str = (
                                            f', evidence="{evidence[:50]}..."'
                                            if len(evidence) > 50
                                            else f', evidence="{evidence}"'
                                        )
                                        print(f"      - {name} ({entity_type}, confidence={conf:.2f}{evid_str})")
                            except Exception as e:
                                print(f"    Warning: Failed to store entities for memory {memory_id}: {e}")

                except Exception as e:
                    print(f"    Warning: Entity extraction failed for chunk: {e}", flush=True)

            # M5: Generate per-chunk summary (if chunk has ≥2 committed memories)
            chunk_memory_count = len(committed_memories)
            if chunk_memory_count >= 2 and extraction_model:
                try:
                    print(f"    Generating chunk summary ({chunk_memory_count} memories)...", flush=True)

                    # Get summary prompt name from config
                    prompt_name = "summary_v1"
                    if prompts_config:
                        prompt_name = prompts_config.get("summary", "summary_v1")

                    # Get committed memory nodes for this chunk
                    chunk_memory_nodes = []
                    for mem_id, _ in committed_memories:
                        mem_node = storage.get_memory(mem_id)
                        if mem_node:
                            chunk_memory_nodes.append(mem_node)

                    if chunk_memory_nodes:
                        # Generate summary from chunk memories
                        summary_result = generate_summary(
                            memories=chunk_memory_nodes,
                            model=extraction_model,
                            source_label=f"{path.name} (chunk {i})",
                            ingest_run_id=f"{path.name}_chunk_{i}",
                            prompt_name=prompt_name,
                        )

                        # Commit summary with chunk_id
                        summary_id = commit_summary(
                            summary_result=summary_result,
                            memory_ids=[m.id for m in chunk_memory_nodes],
                            artifact_ref=path.name,
                            source_label=f"{path.name} (chunk {i})",
                            storage=storage,
                            embedding_engine=embedding_engine,
                            event_storage=event_storage,
                            chunk_id=chunk_id,  # M5: Link summary to chunk
                        )

                        if verbose and summary_id:
                            print(f"    Summary: {summary_result.summary.title}", flush=True)

                except Exception as e:
                    print(f"    Warning: Summary generation failed for chunk: {e}", flush=True)

        except Exception as e:
            error_msg = f"Chunk {i} extraction failed: {e}"
            errors.append(error_msg)
            print(f"Error: {error_msg}")

    # Track entities created
    entities_after = len(storage.get_all_entities())
    entities_created = entities_after - entities_before

    # M5: Per-chunk summaries are generated in the chunk loop above (not here)
    # Removed old per-document summary generation

    return IngestionResult(
        document_path=document_path,
        chunks_processed=len(chunks),
        memories_extracted=memories_extracted,
        memories_committed=memories_committed,
        memories_deduplicated=memories_deduplicated,
        entities_created=entities_created,
        errors=errors,
    )
