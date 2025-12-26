"""CLI entry point for Vestig memory system"""

import argparse
import sys
from typing import Any

from vestig.core.commitment import commit_memory
from vestig.core.config import load_config
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.event_storage import MemoryEventStorage
from vestig.core.storage import MemoryStorage
from vestig.core.tracerank import TraceRankConfig


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


def build_runtime(config: dict[str, Any]) -> tuple[MemoryStorage, EmbeddingEngine, MemoryEventStorage, TraceRankConfig]:
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
        )
        print(format_search_results(results))
    finally:
        storage.close()


def cmd_recall(args):
    """Handle 'vestig memory recall' command"""
    from vestig.core.retrieval import format_recall_results, search_memories

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
        )
        print(format_recall_results(results))
    finally:
        storage.close()


def cmd_show(args):
    """Handle 'vestig memory show' command"""
    import json

    config = args.config_dict
    storage = MemoryStorage(config["storage"]["db_path"])

    try:
        memory = storage.get_memory(args.id)
        if memory is None:
            print(f"Error: Memory not found: {args.id}", file=sys.stderr)
            sys.exit(1)

        # Display full memory details
        print(f"ID: {memory.id}")
        print(f"Created: {memory.created_at}")
        print(f"Metadata: {json.dumps(memory.metadata, indent=2)}")
        print(f"Embedding: {len(memory.content_embedding)} dimensions")
        print(f"  First 5 values: {memory.content_embedding[:5]}")
        print(f"\nContent:\n{memory.content}")
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
            print(f"Warning: Memory {args.id} is already deprecated (t_expired={memory.t_expired})", file=sys.stderr)
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
                }
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


def cmd_memory(args):
    """Handle 'vestig memory' subcommand routing"""
    # This is just a router - actual work done by add/search/recall/show
    # If we get here without a memory subcommand, show help
    args.memory_parser.print_help()
    sys.exit(1)


def cmd_ingest(args):
    """Handle 'vestig ingest' command"""
    from vestig.core.ingestion import ingest_document

    config = args.config_dict
    storage, embedding_engine, event_storage, _ = build_runtime(config)

    # Get M4 config for entity extraction
    m4_config = config.get("m4", {})

    # Get ingestion config with CLI overrides
    ingestion_config = config.get("ingestion", {})
    model = args.model if args.model else ingestion_config.get("model")
    chunk_size = args.chunk_size if args.chunk_size else ingestion_config.get("chunk_size", 20000)
    chunk_overlap = args.chunk_overlap if args.chunk_overlap else ingestion_config.get("chunk_overlap", 500)
    min_confidence = args.min_confidence if args.min_confidence is not None else ingestion_config.get("min_confidence", 0.6)

    if not model:
        raise ValueError("Model must be specified in config (ingestion.model) or via --model argument")

    try:
        result = ingest_document(
            document_path=args.document,
            storage=storage,
            embedding_engine=embedding_engine,
            event_storage=event_storage,
            m4_config=m4_config,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extraction_model=model,
            min_confidence=min_confidence,
            source="document_ingest",
        )

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
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
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
    parser_ingest.add_argument("document", help="Path to document file")
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
    parser_ingest.set_defaults(func=cmd_ingest)

    # vestig memory
    parser_memory = subparsers.add_parser("memory", help="Memory operations")
    memory_subparsers = parser_memory.add_subparsers(
        dest="memory_command", help="Memory commands"
    )

    # vestig memory add
    parser_add = memory_subparsers.add_parser("add", help="Add a new memory")
    parser_add.add_argument("content", help="Memory content")
    parser_add.add_argument(
        "--source", default="manual", help="Memory source (default: manual)"
    )
    parser_add.add_argument(
        "--tags", help="Comma-separated tags (e.g., bug,auth,fix)"
    )
    parser_add.set_defaults(func=cmd_add)

    # vestig memory search
    parser_search = memory_subparsers.add_parser(
        "search", help="Search memories by semantic similarity"
    )
    parser_search.add_argument("query", help="Search query")
    parser_search.add_argument(
        "--limit", type=int, default=5, help="Number of results (default: 5)"
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
    parser_recall.set_defaults(func=cmd_recall)

    # vestig memory show
    parser_show = memory_subparsers.add_parser("show", help="Show memory details by ID")
    parser_show.add_argument("id", help="Memory ID")
    parser_show.set_defaults(func=cmd_show)

    # vestig memory deprecate
    parser_deprecate = memory_subparsers.add_parser(
        "deprecate", help="Mark memory as deprecated"
    )
    parser_deprecate.add_argument("id", help="Memory ID to deprecate")
    parser_deprecate.add_argument(
        "--reason", help="Reason for deprecation (stored in event payload)"
    )
    parser_deprecate.add_argument(
        "--t-invalid", help="When fact became invalid (ISO 8601 timestamp)"
    )
    parser_deprecate.set_defaults(func=cmd_deprecate)

    # Set default for memory command (show help if no subcommand)
    parser_memory.set_defaults(func=cmd_memory, memory_parser=parser_memory)

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

    # Dispatch to subcommand handler
    args.func(args)


if __name__ == "__main__":
    main()
