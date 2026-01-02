#!/usr/bin/env python3
"""
Create TraceRank configuration variants from base configs
Properly modifies YAML values for testing
"""

import sys
import yaml
from pathlib import Path


def create_variant(base_config_path: Path, suffix: str, k_value: float,
                   graph_enabled: bool, graph_k: float) -> Path:
    """
    Create a config variant with specific TraceRank settings

    Args:
        base_config_path: Path to base config file
        suffix: Suffix for variant (e.g., "full", "off", "no-graph")
        k_value: TraceRank k parameter value
        graph_enabled: Whether graph connectivity is enabled
        graph_k: Graph connectivity boost strength

    Returns:
        Path to created config file
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update TraceRank settings
    if 'm3' not in config:
        config['m3'] = {}
    if 'tracerank' not in config['m3']:
        config['m3']['tracerank'] = {}

    config['m3']['tracerank']['k'] = k_value
    config['m3']['tracerank']['graph_connectivity_enabled'] = graph_enabled
    config['m3']['tracerank']['graph_k'] = graph_k

    # Create output path
    output_path = base_config_path.parent / f"{base_config_path.stem}-{suffix}.yaml"

    # Write modified config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 create_tracerank_variants.py <base_config.yaml>")
        print("Creates three variants: -full, -no-graph, -off")
        sys.exit(1)

    base_config = Path(sys.argv[1])

    if not base_config.exists():
        print(f"Error: Config file not found: {base_config}")
        sys.exit(1)

    print(f"Creating TraceRank variants from: {base_config}")

    # Create three variants
    variants = [
        ("full", 0.35, True, 0.15, "Full TraceRank with graph connectivity"),
        ("no-graph", 0.35, False, 0.0, "TraceRank without graph boost"),
        ("off", 0.0, False, 0.0, "No TraceRank (pure embeddings)"),
    ]

    for suffix, k, graph_enabled, graph_k, description in variants:
        output = create_variant(base_config, suffix, k, graph_enabled, graph_k)
        print(f"  ✓ Created {output.name}: {description}")
        print(f"     k={k}, graph_enabled={graph_enabled}, graph_k={graph_k}")

    print("\nDone! Config variants created.")


if __name__ == "__main__":
    main()
