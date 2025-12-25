"""CLI entry point for Vestig memory system"""

import argparse
import sys
from typing import Dict, Any, Tuple

from vestig.core.commitment import commit_memory
from vestig.core.config import load_config
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.storage import MemoryStorage


def validate_config(config: Dict[str, Any]) -> None:
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


def build_runtime(config: Dict[str, Any]) -> Tuple[MemoryStorage, EmbeddingEngine]:
    """
    Build storage and embedding engine from config.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (storage, embedding_engine)
    """
    embedding_engine = EmbeddingEngine(
        model_name=config["embedding"]["model"],
        expected_dimension=config["embedding"]["dimension"],
        normalize=config["embedding"]["normalize"],
    )
    storage = MemoryStorage(config["storage"]["db_path"])
    return storage, embedding_engine


def cmd_add(args):
    """Handle 'vestig memory add' command"""
    config = args.config_dict
    storage, embedding_engine = build_runtime(config)

    try:
        memory_id = commit_memory(
            content=args.content,
            storage=storage,
            embedding_engine=embedding_engine,
            source="manual",
        )
        print(f"Memory stored: {memory_id}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        storage.close()


def cmd_search(args):
    """Handle 'vestig memory search' command"""
    from vestig.core.retrieval import format_search_results, search_memories

    config = args.config_dict
    storage, embedding_engine = build_runtime(config)

    try:
        results = search_memories(
            query=args.query,
            storage=storage,
            embedding_engine=embedding_engine,
            limit=args.limit,
        )
        print(format_search_results(results))
    finally:
        storage.close()


def cmd_recall(args):
    """Handle 'vestig memory recall' command"""
    from vestig.core.retrieval import format_recall_results, search_memories

    config = args.config_dict
    storage, embedding_engine = build_runtime(config)

    try:
        results = search_memories(
            query=args.query,
            storage=storage,
            embedding_engine=embedding_engine,
            limit=args.limit,
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


def cmd_memory(args):
    """Handle 'vestig memory' subcommand routing"""
    # This is just a router - actual work done by add/search/recall/show
    # If we get here without a memory subcommand, show help
    args.memory_parser.print_help()
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

    # vestig memory
    parser_memory = subparsers.add_parser("memory", help="Memory operations")
    memory_subparsers = parser_memory.add_subparsers(
        dest="memory_command", help="Memory commands"
    )

    # vestig memory add
    parser_add = memory_subparsers.add_parser("add", help="Add a new memory")
    parser_add.add_argument("content", help="Memory content")
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
