"""CLI entry point for Vestig memory system"""

import argparse
import glob
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from vestig.core.commitment import commit_memory
from vestig.core.config import load_config
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.event_storage import MemoryEventStorage
from vestig.core.storage import MemoryStorage
from vestig.core.tracerank import TraceRankConfig


def get_version() -> str:
    """Get the installed version of vestig."""
    try:
        return version("vestig")
    except PackageNotFoundError:
        return "dev"


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate required config keys.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required keys are missing
    """
    required_keys = [
        ("embedding", "model"),
        ("embedding", "dimension"),
        ("embedding", "normalize"),
        ("storage", "db_path"),
    ]

    for *path, key in required_keys:
        current = config
        for part in path:
            if part not in current:
                raise ValueError(f"Missing required config key: {'.'.join(path)}.{key}")
            current = current[part]
        if key not in current:
            raise ValueError(f"Missing required config key: {'.'.join(path)}.{key}")


def _truncate(text: str, length: int = 80) -> str:
    if len(text) <= length:
        return text
    return text[: max(0, length - 3)] + "..."


def _resolve_node_label(storage: MemoryStorage, node_id: str, length: int = 80) -> str:
    if node_id.startswith("mem_"):
        memory = storage.get_memory(node_id)
        if memory:
            return _truncate(memory.content, length)
    if node_id.startswith("ent_"):
        entity = storage.get_entity(node_id)
        if entity:
            return f"{entity.canonical_name} ({entity.entity_type})"
    return node_id


def expand_ingest_paths(pattern: str, recursive: bool = False) -> list[str]:
    """Expand glob patterns for ingest paths.

    Args:
        pattern: File path or glob pattern (e.g., "*.txt", "**/*.md")
        recursive: Enable recursive globbing (allows ** to match subdirectories)

    Returns:
        List of matching file paths
    """
    expanded = os.path.expanduser(pattern)
    if glob.has_magic(expanded):
        return sorted(glob.glob(expanded, recursive=recursive))
    return [expanded]


def build_runtime(
    config: dict[str, Any],
) -> tuple[MemoryStorage, EmbeddingEngine, MemoryEventStorage, TraceRankConfig]:
    """
    Build storage, embedding engine, event storage, and TraceRank config from config.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (storage, embedding_engine, event_storage, tracerank_config)
    """
    embedding_engine = EmbeddingEngine(
        model_name=config["embedding"]["model"],
        expected_dimension=config["embedding"]["dimension"],
        normalize=config["embedding"]["normalize"],
        provider=config["embedding"].get("provider", "llm"),  # Default to llm
        max_length=config["embedding"].get("max_length"),  # Optional truncation
    )
    storage = MemoryStorage(config["storage"]["db_path"])
    event_storage = MemoryEventStorage(storage.conn)  # M3: Share DB connection

    # M3: Build TraceRank config
    m3_config = config.get("m3", {})
    tracerank_cfg = m3_config.get("tracerank", {})
    tracerank_config = TraceRankConfig(
        enabled=tracerank_cfg.get("enabled", True),
        tau_days=tracerank_cfg.get("tau_days", 21.0),
        cooldown_hours=tracerank_cfg.get("cooldown_hours", 24.0),
        burst_discount=tracerank_cfg.get("burst_discount", 0.2),
        k=tracerank_cfg.get("k", 0.35),
        graph_connectivity_enabled=tracerank_cfg.get("graph_connectivity_enabled", True),
        graph_k=tracerank_cfg.get("graph_k", 0.15),
        temporal_decay_enabled=tracerank_cfg.get("temporal_decay_enabled", True),
        dynamic_tau_days=tracerank_cfg.get("dynamic_tau_days", 90.0),
        ephemeral_tau_days=tracerank_cfg.get("ephemeral_tau_days"),
        static_boost=tracerank_cfg.get("static_boost", 1.0),
    )

    return storage, embedding_engine, event_storage, tracerank_config


def cmd_add(args):
    """Handle 'vestig memory add' command"""
    config = args.config_dict
    storage, embedding_engine, event_storage, _ = build_runtime(config)

    # Parse tags if provided
    tags = None
    if args.tags:
        tags = [tag.strip() for tag in args.tags.split(",")]

    try:
        outcome = commit_memory(
            content=args.content,
            storage=storage,
            embedding_engine=embedding_engine,
            source=args.source,
            hygiene_config=config.get("hygiene", {}),
            tags=tags,
            event_storage=event_storage,  # M3: Enable event logging
            m4_config=config.get("m4", {}),  # M4: Enable one-shot entity extraction
        )

        # Display outcome info
        if outcome.outcome == "EXACT_DUPE":
            print(f"Memory stored: {outcome.memory_id} (exact duplicate detected)")
        elif outcome.outcome == "NEAR_DUPE":
            print(
                f"Memory stored: {outcome.memory_id} "
                f"(near-duplicate of {outcome.matched_memory_id}, "
                f"score={outcome.query_score:.4f})"
            )
        else:  # INSERTED_NEW
            print(f"Memory stored: {outcome.memory_id}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        storage.close()


def cmd_search(args):
    """Handle 'vestig memory search' command"""
    from vestig.core.retrieval import format_search_results, search_memories

    config = args.config_dict
    storage, embedding_engine, event_storage, tracerank_config = build_runtime(config)

    try:
        results = search_memories(
            query=args.query,
            storage=storage,
            embedding_engine=embedding_engine,
            limit=args.limit,
            event_storage=event_storage,
            tracerank_config=tracerank_config,
            show_timing=getattr(args, 'timing', False),
        )
        print(format_search_results(results))
    finally:
        storage.close()


def cmd_recall(args):
    """Handle 'vestig memory recall' command"""
    from vestig.core.retrieval import (
        format_recall_results,
        format_recall_results_with_explanation,
        search_memories,
    )

    config = args.config_dict
    storage, embedding_engine, event_storage, tracerank_config = build_runtime(config)

    try:
        results = search_memories(
            query=args.query,
            storage=storage,
            embedding_engine=embedding_engine,
            limit=args.limit,
            event_storage=event_storage,
            tracerank_config=tracerank_config,
            show_timing=getattr(args, 'timing', False),
        )

        # Format with or without explanation
        if args.explain:
            output = format_recall_results_with_explanation(
                results, event_storage, storage, tracerank_config
            )
        else:
            output = format_recall_results(results)

        print(output)
    finally:
        storage.close()


def cmd_show(args):
    """Handle 'vestig memory show' command"""
    config = args.config_dict
    storage = MemoryStorage(config["storage"]["db_path"])

    try:
        if args.expand > 1:
            print("Error: Only expansion depth 0 or 1 is supported", file=sys.stderr)
            sys.exit(1)

        memory = storage.get_memory(args.id)
        if memory is None:
            print(f"Error: Memory not found: {args.id}", file=sys.stderr)
            sys.exit(1)

        # Display full memory details
        print(f"ID: {memory.id}")
        print(f"Created: {memory.created_at}")
        print(f"t_valid: {memory.t_valid}")
        print(f"t_invalid: {memory.t_invalid}")
        print(f"t_created: {memory.t_created}")
        print(f"t_expired: {memory.t_expired}")
        print(f"temporal_stability: {memory.temporal_stability}")
        print(f"last_seen_at: {memory.last_seen_at}")
        print(f"reinforce_count: {memory.reinforce_count}")
        print(f"Metadata: {json.dumps(memory.metadata, indent=2)}")
        print(f"Embedding: {len(memory.content_embedding)} dimensions")
        print(f"  First 5 values: {memory.content_embedding[:5]}")
        print(f"\nContent:\n{memory.content}")

        if args.expand == 1:
            mentions = storage.get_edges_from_memory(
                memory.id,
                edge_type="MENTIONS",
                include_expired=args.include_expired,
            )
            related = storage.get_edges_from_memory(
                memory.id,
                edge_type="RELATED",
                include_expired=args.include_expired,
            )

            print("\nMENTIONS:")
            if not mentions:
                print("  (none)")
            for edge in mentions:
                label = _resolve_node_label(storage, edge.to_node)
                conf = f"{edge.confidence:.2f}" if edge.confidence is not None else "n/a"
                print(f"  - {edge.to_node} | {label} | confidence={conf}")

            print("\nRELATED:")
            if not related:
                print("  (none)")
            for edge in related:
                label = _resolve_node_label(storage, edge.to_node)
                print(f"  - {edge.to_node} | {label} | weight={edge.weight:.2f}")
    finally:
        storage.close()


def cmd_memory_list(args):
    """Handle 'vestig memory list' command"""
    config = args.config_dict
    storage = MemoryStorage(config["storage"]["db_path"])

    try:
        query = "SELECT id, content, created_at, t_expired, metadata FROM memories "
        params = []
        if not args.include_expired:
            query += "WHERE t_expired IS NULL "
        query += "ORDER BY created_at DESC LIMIT ?"
        params.append(args.limit)

        cursor = storage.conn.execute(query, params)
        rows = cursor.fetchall()

        for row in rows:
            memory_id, content, created_at, t_expired, metadata_json = row
            metadata = json.loads(metadata_json)
            source = metadata.get("source", "unknown")
            status = "expired" if t_expired else "active"
            snippet = _truncate(content, args.snippet_len)
            print(f"{memory_id} | {created_at} | {source} | {status} | {snippet}")
    finally:
        storage.close()


def cmd_entity_list(args):
    """Handle 'vestig entity list' command"""
    config = args.config_dict
    storage = MemoryStorage(config["storage"]["db_path"])

    try:
        query = (
            "SELECT e.id, e.entity_type, e.canonical_name, e.created_at, "
            "e.expired_at, e.merged_into, "
            "COUNT(ed.edge_id) AS mentions "
            "FROM entities e "
            "LEFT JOIN edges ed ON ed.to_node = e.id "
            "AND ed.edge_type = 'MENTIONS' AND ed.t_expired IS NULL "
        )
        params = []
        if not args.include_expired:
            query += "WHERE e.expired_at IS NULL "
        query += "GROUP BY e.id ORDER BY mentions DESC, e.created_at DESC LIMIT ?"
        params.append(args.limit)

        cursor = storage.conn.execute(query, params)
        rows = cursor.fetchall()
        for row in rows:
            entity_id, entity_type, name, created_at, expired_at, merged_into, mentions = row
            status = "expired" if expired_at else "active"
            merged = f" -> {merged_into}" if merged_into else ""
            print(
                f"{entity_id} | {entity_type} | {name} | mentions={mentions} | "
                f"{created_at} | {status}{merged}"
            )
    finally:
        storage.close()


def cmd_entity_show(args):
    """Handle 'vestig entity show' command"""
    config = args.config_dict
    storage = MemoryStorage(config["storage"]["db_path"])

    try:
        if args.expand > 1:
            print("Error: Only expansion depth 0 or 1 is supported", file=sys.stderr)
            sys.exit(1)

        entity = storage.get_entity(args.id)
        if entity is None:
            print(f"Error: Entity not found: {args.id}", file=sys.stderr)
            sys.exit(1)

        print(f"ID: {entity.id}")
        print(f"Type: {entity.entity_type}")
        print(f"Name: {entity.canonical_name}")
        print(f"Created: {entity.created_at}")
        print(f"Expired: {entity.expired_at}")
        print(f"Merged into: {entity.merged_into}")

        if args.expand == 1:
            edges = storage.get_edges_to_entity(
                entity.id,
                include_expired=args.include_expired,
            )
            print("\nMENTIONED IN:")
            if not edges:
                print("  (none)")
            for edge in edges:
                label = _resolve_node_label(storage, edge.from_node)
                conf = f"{edge.confidence:.2f}" if edge.confidence is not None else "n/a"
                print(f"  - {edge.from_node} | {label} | confidence={conf}")
    finally:
        storage.close()


def cmd_entity_purge(args):
    """Handle 'vestig entity purge --force' command"""
    config = args.config_dict
    db_path = config["storage"]["db_path"]
    storage = MemoryStorage(db_path)

    try:
        if not args.force:
            print("Error: --force flag is required to purge all entities", file=sys.stderr)
            sys.exit(1)

        print(f"vestig v{get_version()}")
        print("Purging all entities and edges...")
        print(f"Database: {Path(db_path).absolute()}")
        print()

        # Count before deletion
        entity_count = storage.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        mentions_count = storage.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE edge_type = 'MENTIONS'"
        ).fetchone()[0]

        print(f"  Entities to delete: {entity_count}")
        print(f"  MENTIONS edges to delete: {mentions_count}")
        print()

        # Delete all MENTIONS edges first (referential integrity)
        storage.conn.execute("DELETE FROM edges WHERE edge_type = 'MENTIONS'")

        # Delete all entities
        storage.conn.execute("DELETE FROM entities")

        storage.conn.commit()

        print("✓ Purge completed successfully")
        print(f"  Deleted {entity_count} entities")
        print(f"  Deleted {mentions_count} MENTIONS edges")

    except KeyboardInterrupt:
        print("\n\nPurge interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError during purge: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        storage.close()


def cmd_entity_extract(args):
    """Handle 'vestig entity extract' command"""
    from vestig.core.entity_extraction import process_memories_for_entities

    config = args.config_dict
    db_path = config["storage"]["db_path"]
    storage = MemoryStorage(db_path)

    # Get model from config
    m4_config = config.get("m4", {})
    entity_config = m4_config.get("entity_extraction", {})
    llm_config = entity_config.get("llm", {})
    model = llm_config.get("model", "claude-haiku-4.5")

    try:
        print(f"vestig v{get_version()}")
        print("Extracting entities from memories...")
        print(f"Database: {Path(db_path).absolute()}")
        print(f"Model: {model}")
        print(f"Reprocess: {args.reprocess}")
        print(f"Batch size: {args.batch_size}")
        print(f"Verbose: {args.verbose}")
        print()

        # Process memories for entity extraction
        stats = process_memories_for_entities(
            storage=storage,
            config=config,
            reprocess=args.reprocess,
            batch_size=args.batch_size,
            verbose=args.verbose
        )

        print("\n✓ Entity extraction completed successfully")

    except KeyboardInterrupt:
        print("\n\nEntity extraction interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError during entity extraction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        storage.close()


def cmd_edge_list(args):
    """Handle 'vestig edge list' command"""
    config = args.config_dict
    storage = MemoryStorage(config["storage"]["db_path"])

    try:
        query = (
            "SELECT edge_id, from_node, to_node, edge_type, weight, confidence, "
            "t_created, t_expired FROM edges "
        )
        params = []
        if args.type != "ALL":
            query += "WHERE edge_type = ? "
            params.append(args.type)
            if not args.include_expired:
                query += "AND t_expired IS NULL "
        else:
            if not args.include_expired:
                query += "WHERE t_expired IS NULL "
        query += "ORDER BY t_created DESC LIMIT ?"
        params.append(args.limit)

        cursor = storage.conn.execute(query, params)
        rows = cursor.fetchall()
        for row in rows:
            edge_id, from_node, to_node, edge_type, weight, confidence, t_created, t_expired = row
            from_label = _resolve_node_label(storage, from_node, args.snippet_len)
            to_label = _resolve_node_label(storage, to_node, args.snippet_len)
            conf = f"{confidence:.2f}" if confidence is not None else "n/a"
            status = "expired" if t_expired else "active"
            print(
                f"{edge_id} | {edge_type} | {status} | {from_node} -> {to_node} | "
                f"weight={weight:.2f} | confidence={conf}"
            )
            print(f"  from: {from_label}")
            print(f"  to:   {to_label}")
    finally:
        storage.close()


def cmd_edge_show(args):
    """Handle 'vestig edge show' command"""
    config = args.config_dict
    storage = MemoryStorage(config["storage"]["db_path"])

    try:
        edge = storage.get_edge(args.id)
        if edge is None:
            print(f"Error: Edge not found: {args.id}", file=sys.stderr)
            sys.exit(1)

        print(f"ID: {edge.edge_id}")
        print(f"Type: {edge.edge_type}")
        print(f"From: {edge.from_node}")
        print(f"To: {edge.to_node}")
        print(f"Weight: {edge.weight}")
        print(f"Confidence: {edge.confidence}")
        print(f"Evidence: {edge.evidence}")
        print(f"t_valid: {edge.t_valid}")
        print(f"t_invalid: {edge.t_invalid}")
        print(f"t_created: {edge.t_created}")
        print(f"t_expired: {edge.t_expired}")

        from_label = _resolve_node_label(storage, edge.from_node)
        to_label = _resolve_node_label(storage, edge.to_node)
        print(f"\nFrom node: {from_label}")
        print(f"To node: {to_label}")
    finally:
        storage.close()


def cmd_deprecate(args):
    """Handle 'vestig memory deprecate' command"""
    from vestig.core.models import EventNode

    config = args.config_dict
    storage, _, event_storage, _ = build_runtime(config)

    try:
        # Verify memory exists
        memory = storage.get_memory(args.id)
        if memory is None:
            print(f"Error: Memory not found: {args.id}", file=sys.stderr)
            sys.exit(1)

        # Check if already deprecated
        if memory.t_expired:
            print(
                f"Warning: Memory {args.id} is already deprecated (t_expired={memory.t_expired})",
                file=sys.stderr,
            )
            sys.exit(1)

        # M3 FIX #5: Atomic transaction for event + deprecation
        with storage.conn:
            # Create DEPRECATE event
            event = EventNode.create(
                memory_id=args.id,
                event_type="DEPRECATE",
                source="manual",
                payload={
                    "reason": args.reason or "Manual deprecation",
                    "t_invalid": args.t_invalid,
                },
            )
            event_storage.add_event(event)

            # Mark memory as deprecated
            storage.deprecate_memory(args.id, t_invalid=args.t_invalid)

            # Transaction commits here automatically

        print(f"Memory {args.id} deprecated successfully")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        storage.close()


def cmd_regen_embeddings(args):
    """Handle 'vestig memory regen-embeddings' command"""
    import json

    config = args.config_dict

    # Override embedding config with new model if specified
    if args.model:
        config["embedding"]["model"] = args.model
        print(f"Using model: {args.model}")

    # Build runtime with new embedding engine
    storage, embedding_engine, _, _ = build_runtime(config)

    try:
        # Get all memories (includes both MEMORY and SUMMARY kinds)
        print("Loading all memories from database...")
        all_memories = storage.get_all_memories()

        # Apply limit if specified (for testing)
        if args.limit:
            all_memories = all_memories[:args.limit]
            print(f"Processing first {len(all_memories)} memories (--limit {args.limit})")
        else:
            print(f"Processing {len(all_memories)} memories")

        if not all_memories:
            print("No memories found in database")
            return

        # Process memories in batches
        batch_size = args.batch_size
        total_processed = 0
        total_errors = 0

        for i in range(0, len(all_memories), batch_size):
            batch = all_memories[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(all_memories) + batch_size - 1) // batch_size

            print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} memories)...")

            # Generate embeddings for batch
            texts = [m.content for m in batch]
            try:
                embeddings = embedding_engine.embed_batch(texts)
            except Exception as e:
                print(f"Error generating embeddings for batch {batch_num}: {e}", file=sys.stderr)
                total_errors += len(batch)
                continue

            # Update database with new embeddings
            with storage.conn:
                for memory, embedding in zip(batch, embeddings):
                    try:
                        embedding_json = json.dumps(embedding)
                        storage.conn.execute(
                            "UPDATE memories SET content_embedding = ? WHERE id = ?",
                            (embedding_json, memory.id)
                        )
                        total_processed += 1
                    except Exception as e:
                        print(f"Error updating memory {memory.id}: {e}", file=sys.stderr)
                        total_errors += 1

            # Show progress
            progress_pct = (i + len(batch)) / len(all_memories) * 100
            print(f"Progress: {total_processed}/{len(all_memories)} ({progress_pct:.1f}%)")

        print(f"\n{'='*60}")
        print("REGENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total processed: {total_processed}")
        print(f"Total errors: {total_errors}")
        print(f"Model: {config['embedding']['model']}")
        print(f"Dimension: {config['embedding']['dimension']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        storage.close()


def cmd_memory(args):
    """Handle noun subcommand routing"""
    # If we get here without a subcommand, show help
    parser = getattr(args, "noun_parser", None)
    if parser is None:
        parser = getattr(args, "memory_parser", None)
    if parser is not None:
        parser.print_help()
    sys.exit(1)


def cmd_ingest(args):
    """Handle 'vestig ingest' command"""
    from vestig.core.ingestion import ingest_document

    config = args.config_dict
    storage, embedding_engine, event_storage, _ = build_runtime(config)

    # Get M4 config for entity extraction
    m4_config = config.get("m4", {})

    # Get prompts config for prompt version selection
    prompts_config = config.get("prompts", {})

    # Get ingestion config with CLI overrides
    ingestion_config = config.get("ingestion", {})
    model = args.model if args.model else ingestion_config.get("model")
    chunk_size = args.chunk_size if args.chunk_size else ingestion_config.get("chunk_size", 20000)
    chunk_overlap = (
        args.chunk_overlap if args.chunk_overlap else ingestion_config.get("chunk_overlap", 500)
    )
    min_confidence = (
        args.min_confidence
        if args.min_confidence is not None
        else ingestion_config.get("min_confidence", 0.6)
    )
    source_format = args.format if args.format else ingestion_config.get("format", "auto")
    format_config = ingestion_config.get("claude_session", {})
    force_entities = ingestion_config.get("force_entities", [])
    if args.force_entity:
        force_entities = force_entities + args.force_entity

    if not model:
        raise ValueError(
            "Model must be specified in config (ingestion.model) or via --model argument"
        )

    paths = expand_ingest_paths(args.document, recursive=args.recurse)
    if not paths:
        print(f"No files match: {args.document}", file=sys.stderr)
        sys.exit(1)

    total = {
        "chunks": 0,
        "extracted": 0,
        "committed": 0,
        "deduped": 0,
        "entities": 0,
        "errors": 0,
    }
    failures = []

    for idx, document_path in enumerate(paths, 1):
        try:
            if len(paths) > 1:
                print("\n" + "=" * 70)
                print(f"INGESTING {idx}/{len(paths)}")
                print("=" * 70)

            result = ingest_document(
                document_path=document_path,
                storage=storage,
                embedding_engine=embedding_engine,
                event_storage=event_storage,
                m4_config=m4_config,
                prompts_config=prompts_config,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                extraction_model=model,
                min_confidence=min_confidence,
                source="document_ingest",
                source_format=source_format,
                format_config=format_config,
                force_entities=force_entities,
                verbose=args.verbose,
            )

            total["chunks"] += result.chunks_processed
            total["extracted"] += result.memories_extracted
            total["committed"] += result.memories_committed
            total["deduped"] += result.memories_deduplicated
            total["entities"] += result.entities_created
            total["errors"] += len(result.errors)

            # Print summary
            print("\n" + "=" * 70)
            print("INGESTION COMPLETE")
            print("=" * 70)
            print(f"Document: {result.document_path}")
            print(f"Chunks processed: {result.chunks_processed}")
            print(f"Memories extracted: {result.memories_extracted}")
            print(f"Memories committed: {result.memories_committed}")
            print(f"Duplicates skipped: {result.memories_deduplicated}")
            print(f"Entities created: {result.entities_created}")

            if result.errors:
                print(f"\nErrors: {len(result.errors)}")
                for error in result.errors[:5]:  # Show first 5 errors
                    print(f"  - {error}")
                if len(result.errors) > 5:
                    print(f"  ... and {len(result.errors) - 5} more")

            print("=" * 70)

        except FileNotFoundError as e:
            failures.append(str(e))
        except Exception as e:
            failures.append(f"{document_path}: {e}")

    if len(paths) > 1:
        print("\n" + "=" * 70)
        print("INGESTION SUMMARY")
        print("=" * 70)
        print(f"Documents: {len(paths)}")
        print(f"Chunks processed: {total['chunks']}")
        print(f"Memories extracted: {total['extracted']}")
        print(f"Memories committed: {total['committed']}")
        print(f"Duplicates skipped: {total['deduped']}")
        print(f"Entities created: {total['entities']}")
        print(f"Errors: {total['errors']}")
        print("=" * 70)

    if failures:
        print("\nErrors during ingestion:", file=sys.stderr)
        for error in failures[:5]:
            print(f"  - {error}", file=sys.stderr)
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="vestig",
        description="Vestig: LLM Agent Memory System",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # vestig ingest
    parser_ingest = subparsers.add_parser(
        "ingest", help="Ingest document by extracting memories with LLM"
    )
    parser_ingest.add_argument("document", help="Path to document file or glob pattern")
    parser_ingest.add_argument(
        "-r",
        "--recurse",
        action="store_true",
        help="Enable recursive globbing (allows ** to match subdirectories)",
    )
    parser_ingest.add_argument(
        "--format",
        choices=["auto", "plain", "claude-session"],
        help="Input format (default from config ingestion.format or auto)",
    )
    parser_ingest.add_argument(
        "--force-entity",
        action="append",
        help="Force entity on every memory (format TYPE:Name, repeatable)",
    )
    parser_ingest.add_argument(
        "--chunk-size",
        type=int,
        help="Characters per chunk (overrides config, default from config or 20000)",
    )
    parser_ingest.add_argument(
        "--chunk-overlap",
        type=int,
        help="Character overlap between chunks (overrides config, default from config or 500)",
    )
    parser_ingest.add_argument(
        "--model",
        help="LLM model for extraction (overrides config, required if not in config)",
    )
    parser_ingest.add_argument(
        "--min-confidence",
        type=float,
        help="Minimum confidence for extracted memories (overrides config, default from config or 0.6)",
    )
    parser_ingest.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed extraction output (memories, entities, confidence values)",
    )
    parser_ingest.set_defaults(func=cmd_ingest)

    # vestig memory
    parser_memory = subparsers.add_parser("memory", help="Memory operations")
    memory_subparsers = parser_memory.add_subparsers(dest="memory_command", help="Memory commands")

    # vestig memory add
    parser_add = memory_subparsers.add_parser("add", help="Add a new memory")
    parser_add.add_argument("content", help="Memory content")
    parser_add.add_argument("--source", default="manual", help="Memory source (default: manual)")
    parser_add.add_argument("--tags", help="Comma-separated tags (e.g., bug,auth,fix)")
    parser_add.set_defaults(func=cmd_add)

    # vestig memory search
    parser_search = memory_subparsers.add_parser(
        "search", help="Search memories by semantic similarity"
    )
    parser_search.add_argument("query", help="Search query")
    parser_search.add_argument(
        "--limit", type=int, default=5, help="Number of results (default: 5)"
    )
    parser_search.add_argument(
        "--timing",
        action="store_true",
        help="Show performance timing breakdown",
    )
    parser_search.set_defaults(func=cmd_search)

    # vestig memory recall
    parser_recall = memory_subparsers.add_parser(
        "recall", help="Recall memories formatted for agent context"
    )
    parser_recall.add_argument("query", help="Recall query")
    parser_recall.add_argument(
        "--limit", type=int, default=5, help="Number of results (default: 5)"
    )
    parser_recall.add_argument(
        "--explain",
        action="store_true",
        help="Show explanation for why each memory was retrieved",
    )
    parser_recall.add_argument(
        "--timing",
        action="store_true",
        help="Show performance timing breakdown",
    )
    parser_recall.set_defaults(func=cmd_recall)

    # vestig memory show
    parser_show = memory_subparsers.add_parser("show", help="Show memory details by ID")
    parser_show.add_argument("id", help="Memory ID")
    parser_show.add_argument(
        "--expand",
        type=int,
        default=0,
        help="Expansion depth (0 or 1)",
    )
    parser_show.add_argument(
        "--include-expired",
        action="store_true",
        help="Include expired edges in expansion",
    )
    parser_show.set_defaults(func=cmd_show)

    # vestig memory list
    parser_list = memory_subparsers.add_parser("list", help="List memories")
    parser_list.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of memories to show (default: 20)",
    )
    parser_list.add_argument(
        "--snippet-len",
        type=int,
        default=80,
        help="Snippet length for content preview (default: 80)",
    )
    parser_list.add_argument(
        "--include-expired",
        action="store_true",
        help="Include expired memories",
    )
    parser_list.set_defaults(func=cmd_memory_list)

    # vestig memory deprecate
    parser_deprecate = memory_subparsers.add_parser("deprecate", help="Mark memory as deprecated")
    parser_deprecate.add_argument("id", help="Memory ID to deprecate")
    parser_deprecate.add_argument(
        "--reason", help="Reason for deprecation (stored in event payload)"
    )
    parser_deprecate.add_argument(
        "--t-invalid", help="When fact became invalid (ISO 8601 timestamp)"
    )
    parser_deprecate.set_defaults(func=cmd_deprecate)

    # vestig memory regen-embeddings
    parser_regen = memory_subparsers.add_parser(
        "regen-embeddings", help="Regenerate all embeddings with a new model"
    )
    parser_regen.add_argument(
        "--model",
        help="New embedding model to use (overrides config, e.g., 'ollama/nomic-embed-text')",
    )
    parser_regen.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of memories to process per batch (default: 100)",
    )
    parser_regen.add_argument(
        "--limit",
        type=int,
        help="Only regenerate first N memories (for testing)",
    )
    parser_regen.set_defaults(func=cmd_regen_embeddings)

    # Set default for memory command (show help if no subcommand)
    parser_memory.set_defaults(func=cmd_memory, noun_parser=parser_memory)

    # vestig entity
    parser_entity = subparsers.add_parser("entity", help="Entity operations")
    entity_subparsers = parser_entity.add_subparsers(dest="entity_command", help="Entity commands")

    parser_entity_list = entity_subparsers.add_parser("list", help="List entities")
    parser_entity_list.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of entities to show (default: 20)",
    )
    parser_entity_list.add_argument(
        "--include-expired",
        action="store_true",
        help="Include expired entities",
    )
    parser_entity_list.set_defaults(func=cmd_entity_list)

    parser_entity_show = entity_subparsers.add_parser("show", help="Show entity details by ID")
    parser_entity_show.add_argument("id", help="Entity ID")
    parser_entity_show.add_argument(
        "--expand",
        type=int,
        default=0,
        help="Expansion depth (0 or 1)",
    )
    parser_entity_show.add_argument(
        "--include-expired",
        action="store_true",
        help="Include expired edges in expansion",
    )
    parser_entity_show.set_defaults(func=cmd_entity_show)

    # vestig entity extract
    parser_entity_extract = entity_subparsers.add_parser(
        "extract", help="Extract entities from memories"
    )
    parser_entity_extract.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-extract entities for all memories (not just unprocessed ones)",
    )
    parser_entity_extract.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of memories per batch (default: 1)",
    )
    parser_entity_extract.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output including memories and extracted entities",
    )
    parser_entity_extract.set_defaults(func=cmd_entity_extract)

    # vestig entity purge
    parser_entity_purge = entity_subparsers.add_parser(
        "purge", help="Delete all entities and their edges"
    )
    parser_entity_purge.add_argument(
        "--force",
        action="store_true",
        required=True,
        help="Required: confirm deletion of all entities and edges",
    )
    parser_entity_purge.set_defaults(func=cmd_entity_purge)

    parser_entity.set_defaults(func=cmd_memory, noun_parser=parser_entity)

    # vestig edge
    parser_edge = subparsers.add_parser("edge", help="Edge operations")
    edge_subparsers = parser_edge.add_subparsers(dest="edge_command", help="Edge commands")

    parser_edge_list = edge_subparsers.add_parser("list", help="List edges")
    parser_edge_list.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of edges to show (default: 20)",
    )
    parser_edge_list.add_argument(
        "--type",
        choices=["ALL", "MENTIONS", "RELATED"],
        default="ALL",
        help="Edge type filter (default: ALL)",
    )
    parser_edge_list.add_argument(
        "--snippet-len",
        type=int,
        default=60,
        help="Snippet length for node previews (default: 60)",
    )
    parser_edge_list.add_argument(
        "--include-expired",
        action="store_true",
        help="Include expired edges",
    )
    parser_edge_list.set_defaults(func=cmd_edge_list)

    parser_edge_show = edge_subparsers.add_parser("show", help="Show edge details by ID")
    parser_edge_show.add_argument("id", help="Edge ID")
    parser_edge_show.set_defaults(func=cmd_edge_show)

    parser_edge.set_defaults(func=cmd_memory, noun_parser=parser_edge)

    args = parser.parse_args()

    # Show help if no command provided
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load config with minimal error handling (fail hard on other errors)
    try:
        config = load_config(args.config)
        validate_config(config)
        args.config_dict = config
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Display version if not suppressed
    display_config = config.get("display", {})
    if display_config.get("show_version", True):
        print(f"vestig v{get_version()}")

    # Dispatch to subcommand handler
    args.func(args)


if __name__ == "__main__":
    main()
