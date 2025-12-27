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
    stability: str = "unknown"  # "static" | "dynamic" | "unknown"

    # Evidence for debugging
    extraction_method: str = "default"  # "jsonl_timestamp" | "file_mtime" | "filename_pattern" | "default"
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
            evidence="No temporal metadata available; using current time"
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
            evidence=f"Extracted from file modification time: {path.name}"
        )

    @classmethod
    def from_timestamp(cls, timestamp: str, evidence: str) -> "TemporalHints":
        """Create hints from explicit timestamp (e.g., JSONL event)."""
        return cls(
            t_valid=timestamp,
            stability="unknown",
            extraction_method="jsonl_timestamp",
            evidence=evidence
        )


@dataclass
class ExtractedMemory:
    """Memory extracted from document by LLM with temporal hints"""

    content: str
    confidence: float
    rationale: str
    entities: list[tuple[str, str, float, str]]  # (name, type, confidence, evidence)

    # Temporal hints (optional, None = use defaults)
    t_valid_hint: str | None = None           # When fact became true (ISO 8601)
    temporal_stability_hint: str | None = None # "static" | "dynamic" | "unknown"
    temporal_evidence: str | None = None       # Brief explanation of temporal extraction


# Pydantic schemas for LLM structured output
class EntitySchema(BaseModel):
    """Schema for an extracted entity"""

    name: str = Field(description="Entity name or identifier")
    type: str = Field(
        description="Entity type: PERSON, ORG, SYSTEM, PROJECT, PLACE, SKILL, TOOL, FILE, or CONCEPT"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    evidence: str = Field(
        description="Text snippet from memory that supports this entity"
    )


class MemorySchema(BaseModel):
    """Schema for a single memory with entities"""

    content: str = Field(
        description="The full memory text with enough context to be self-contained"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    rationale: str = Field(description="Brief explanation of why this is worth remembering")
    temporal_stability: str = Field(
        default="unknown",
        description="Temporal stability: static (permanent facts), dynamic (changing state), or unknown"
    )
    entities: list[EntitySchema] = Field(
        default_factory=list,
        description="Entities mentioned in this memory (people, orgs, systems, projects, places)",
    )


class MemoryExtractionResult(BaseModel):
    """Schema for memory extraction response"""

    memories: list[MemorySchema] = Field(description="List of extracted memories")


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


def chunk_text_by_chars(text: str, chunk_size: int = 20000, overlap: int = 500) -> list[str]:
    """
    Chunk text by character count with overlap.

    Simple chunking strategy that tries to break at paragraph boundaries.
    Uses character count as proxy for tokens (roughly 1 token = 4 chars).

    Args:
        text: Text to chunk
        chunk_size: Max characters per chunk (default 20k chars ≈ 5k tokens)
        overlap: Character overlap between chunks (for context continuity)

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Find end of chunk
        end = start + chunk_size

        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
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

        chunks.append(text[start:break_point])

        # Next chunk starts with overlap
        start = max(start + 1, break_point - overlap)

    return chunks


def extract_memories_from_chunk(
    chunk: str,
    model: str,
    min_confidence: float = 0.6,
    temporal_hints: TemporalHints | None = None,
) -> list[ExtractedMemory]:
    """
    Extract memories from a text chunk using LLM.

    Args:
        chunk: Text chunk to extract from
        model: LLM model to use
        min_confidence: Minimum confidence threshold
        temporal_hints: Optional temporal context for this chunk

    Returns:
        List of ExtractedMemory objects with temporal hints

    Raises:
        ValueError: If LLM returns invalid JSON
    """
    # Load and substitute prompt
    prompts = load_prompts()
    template = prompts.get("extract_memories_from_session")

    if not template:
        raise ValueError("'extract_memories_from_session' prompt not found in prompts.yaml")

    prompt = substitute_tokens(template, content=chunk)

    # Call LLM with schema for structured output
    try:
        result = call_llm(prompt, model=model, schema=MemoryExtractionResult)
    except Exception as e:
        raise ValueError(f"LLM call failed: {e}")

    # Convert schema response to ExtractedMemory objects
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

        # Extract entities from schema
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


def ingest_document(
    document_path: str,
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    extraction_model: str,
    event_storage: MemoryEventStorage | None = None,
    m4_config: dict[str, Any] | None = None,
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

    # Chunk text
    if resolved_format == "claude-session":
        chunks_with_hints = extract_claude_session_chunks(
            raw_text,
            format_config or {},
            chunk_size,
            chunk_overlap,
        )
        chunks = [chunk for chunk, _ in chunks_with_hints]
    else:
        chunks = chunk_text_by_chars(text, chunk_size=chunk_size, overlap=chunk_overlap)
        chunks_with_hints = [(chunk, document_temporal_hints) for chunk in chunks]

    print(f"Processing {path.name}: {len(text):,} chars → {len(chunks)} chunks")

    # Track results
    memories_extracted = 0
    memories_committed = 0
    memories_deduplicated = 0
    errors = []

    # Track entities before and after
    entities_before = len(storage.get_all_entities())

    # Process each chunk
    for i, (chunk, chunk_hints) in enumerate(chunks_with_hints, 1):
        print(f"  Chunk {i}/{len(chunks)}: ", end="", flush=True)

        try:
            # Extract memories from chunk
            extracted = extract_memories_from_chunk(
                chunk,
                model=extraction_model,
                min_confidence=min_confidence,
                temporal_hints=chunk_hints,
            )

            print(f"{len(extracted)} memories extracted", flush=True)
            memories_extracted += len(extracted)

            # Show extracted memories in verbose mode
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
                    if memory.entities:
                        print(f"      Entities ({len(memory.entities)}):")
                        for name, entity_type, conf, evidence in memory.entities:
                            evid_str = (
                                f', evidence="{evidence[:50]}..."'
                                if len(evidence) > 50
                                else f', evidence="{evidence}"'
                            )
                            print(f"        - {name} ({entity_type}, confidence={conf:.2f}{evid_str})")

            # Commit each memory
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
                            deduped_entities.append(
                                (name, entity_type, confidence, evidence)
                            )
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
                    )

                    if outcome.outcome == "INSERTED_NEW":
                        memories_committed += 1

                        # Show entities in verbose mode
                        if verbose:
                            # Get MENTIONS edges for this memory
                            edges = storage.get_edges_from_memory(
                                outcome.memory_id, edge_type="MENTIONS", include_expired=False
                            )
                            if edges:
                                print(f"    Memory {idx} - Entities committed ({len(edges)}):")
                                for edge in edges:
                                    entity = storage.get_entity(edge.to_node)
                                    if entity:
                                        conf_str = (
                                            f", confidence={edge.confidence:.2f}"
                                            if edge.confidence
                                            else ""
                                        )
                                        evid_str = (
                                            f', evidence="{edge.evidence[:50]}..."'
                                            if edge.evidence and len(edge.evidence) > 50
                                            else f', evidence="{edge.evidence}"'
                                            if edge.evidence
                                            else ""
                                        )
                                        print(
                                            f"      - {entity.canonical_name} ({entity.entity_type}{conf_str}{evid_str})"
                                        )
                    else:
                        memories_deduplicated += 1

                except Exception as e:
                    error_msg = f"Failed to commit memory: {e}"
                    errors.append(error_msg)
                    print(f"    Error: {error_msg}")

        except Exception as e:
            error_msg = f"Chunk {i} extraction failed: {e}"
            errors.append(error_msg)
            print(f"Error: {error_msg}")

    # Track entities created
    entities_after = len(storage.get_all_entities())
    entities_created = entities_after - entities_before

    return IngestionResult(
        document_path=document_path,
        chunks_processed=len(chunks),
        memories_extracted=memories_extracted,
        memories_committed=memories_committed,
        memories_deduplicated=memories_deduplicated,
        entities_created=entities_created,
        errors=errors,
    )
