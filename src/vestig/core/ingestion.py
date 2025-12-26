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
from vestig.core.storage import MemoryStorage


@dataclass
class ExtractedMemory:
    """Memory extracted from document by LLM"""

    content: str
    confidence: float
    rationale: str


# Pydantic schemas for LLM structured output
class MemorySchema(BaseModel):
    """Schema for a single memory"""

    content: str = Field(
        description="The full memory text with enough context to be self-contained"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    rationale: str = Field(description="Brief explanation of why this is worth remembering")


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
) -> list[ExtractedMemory]:
    """
    Extract memories from a text chunk using LLM.

    Args:
        chunk: Text chunk to extract from
        model: LLM model to use
        min_confidence: Minimum confidence threshold

    Returns:
        List of ExtractedMemory objects

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

        memories.append(
            ExtractedMemory(
                content=content,
                confidence=confidence,
                rationale=rationale,
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

    text = path.read_text(encoding="utf-8")

    # Chunk text
    chunks = chunk_text_by_chars(text, chunk_size=chunk_size, overlap=chunk_overlap)

    print(f"Processing {path.name}: {len(text):,} chars → {len(chunks)} chunks")

    # Track results
    memories_extracted = 0
    memories_committed = 0
    memories_deduplicated = 0
    errors = []

    # Track entities before and after
    entities_before = len(storage.get_all_entities())

    # Process each chunk
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}/{len(chunks)}: ", end="", flush=True)

        try:
            # Extract memories from chunk
            extracted = extract_memories_from_chunk(
                chunk,
                model=extraction_model,
                min_confidence=min_confidence,
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

            # Commit each memory
            for idx, memory in enumerate(extracted, 1):
                try:
                    outcome = commit_memory(
                        content=memory.content,
                        storage=storage,
                        embedding_engine=embedding_engine,
                        source=source,
                        event_storage=event_storage,
                        m4_config=m4_config,
                        artifact_ref=path.name,
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
                                print(f"      Entities extracted ({len(edges)}):")
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
                                            f"        - {entity.canonical_name} ({entity.entity_type}{conf_str}{evid_str})"
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
